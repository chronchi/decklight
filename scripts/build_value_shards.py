"""Outcome shards for the learned value v2 (ADR 0007, Labellers v2).

Every finished game with a winner labels all of its states for free: z = +1 when the
acting player is the winner, else -1. Replays records (teacher / dagger / self-play
games: config + choices + winner) in the C++ env and keeps every stride-th decision
(offset random per game, seeded) plus always the first decision of each turn;
decisions with a single legal option are skipped (no choice, no signal).

Per kept state: the OBS_KEYS arrays (same conventions as the BC shards, so
load_shards-style loading works), z, turn (duel turn), game (global int id; the
shard's game_ids side table maps it to the game_id string), deck (the actor's deck
stem), deck0/deck1 (both seats' stems, for holdout splits by either deck), actor,
src (source dir name: teacher / dagger / selfplay ...) and hscore (the
position_score heuristic, for comparison). One .npz per record file.

    uv run python scripts/build_value_shards.py --records 'data/demos/teacher/*.jsonl' \\
        --out data/cache/value_shards --stride 4 --jobs 4
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

from decklight.model.bc_data import OBS_KEYS

EXTRA_KEYS = ("z", "turn", "game", "deck", "deck0", "deck1", "actor", "src", "hscore")

_ENGINE = None


def _init() -> None:
    global _ENGINE
    from decklight.engine.duel import Engine

    _ENGINE = Engine()


def _offset(game_id: str, stride: int) -> int:
    """Deterministic per-game stride offset in [0, stride)."""
    seed = hashlib.md5(game_id.encode()).hexdigest()
    return random.Random(seed).randrange(max(stride, 1))


def _shard_one(job: tuple) -> tuple[str, int, int, int]:
    """Replay a file's selected records; returns (out_file, games, states, z_pos)."""
    from decklight.demos.dataset import record_config
    from decklight.engine.cpp_env import CppEnv
    from decklight.search.window import position_score

    src_file, out_file, src_tag, rec_ids, game_ids, game_strs, stride = job
    import json

    recs = [json.loads(line) for line in Path(src_file).read_text().splitlines() if line.strip()]
    cols: dict[str, list] = {k: [] for k in OBS_KEYS}
    z, turn, game, deck, deck0, deck1, actor = [], [], [], [], [], [], []
    hscore = []
    game_str_of: dict[int, str] = {}
    n_games = 0
    for rec, gid, gstr in zip((recs[i] for i in rec_ids), game_ids, game_strs, strict=True):
        winner = rec.get("winner")
        if winner not in (0, 1):
            continue
        try:
            env = CppEnv(_ENGINE, record_config(rec), rec.get("auto_single", True))
            obs = env.reset()
            off = _offset(str(gstr), stride)
            seen_turns: set[int] = set()
            kept = 0
            for k, c in enumerate(rec["choices"]):
                if obs is None:
                    break
                me = int(env.player)
                t = int(obs["global"][2])
                n_legal = int(obs["action_mask"].sum())
                first = t not in seen_turns
                seen_turns.add(t)
                if n_legal >= 2 and ((k + off) % stride == 0 or first):
                    for key in OBS_KEYS:
                        cols[key].append(obs[key].copy())
                    z.append(1.0 if me == winner else -1.0)
                    turn.append(t)
                    game.append(gid)
                    decks = rec.get("decks") or []
                    d0 = str(decks[0]) if len(decks) > 0 else ""
                    d1 = str(decks[1]) if len(decks) > 1 else ""
                    deck.append(str(decks[me]) if me < len(decks) else "")
                    deck0.append(d0)
                    deck1.append(d1)
                    actor.append(me)
                    hscore.append(position_score(obs, me, me))
                    kept += 1
                obs = env.step(int(c))
            if kept:
                game_str_of[gid] = str(gstr)
                n_games += 1
        except Exception as e:  # noqa: BLE001 - one bad record must not kill a shard build
            print(f"value_shards: skipping {rec.get('game_id', rec.get('line_id'))} in {src_file}: {e!r}"[:300], flush=True)
            continue
    n_states = len(z)
    if not n_states:
        return out_file, n_games, 0, 0
    gids = sorted(game_str_of)
    np.savez_compressed(
        out_file,
        **{k: np.stack(v) for k, v in cols.items()},
        z=np.array(z, dtype=np.float32),
        turn=np.array(turn, dtype=np.int32),
        game=np.array(game, dtype=np.int32),
        deck=np.array(deck),
        deck0=np.array(deck0),
        deck1=np.array(deck1),
        actor=np.array(actor, dtype=np.int8),
        src=np.array([src_tag] * n_states),
        hscore=np.array(hscore, dtype=np.float32),
        game_ids=np.array([game_str_of[g] for g in gids]),
        game_row=np.array(gids, dtype=np.int32),
    )
    return out_file, n_games, n_states, int((np.array(z) > 0).sum())


def _flat_name(path: Path) -> str:
    rel = path.resolve()
    try:
        rel = rel.relative_to(Path(__file__).resolve().parents[1] / "data" / "demos")
    except ValueError:
        pass
    return "~".join(rel.with_suffix("").parts[-6:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help="record file globs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=None, help="use only the first N eligible games")
    args = ap.parse_args()

    files = sorted({f for g in args.records for f in glob.glob(g)})
    files = [f for f in files if Path(f).stat().st_size]
    print(f"{len(files)} record files", flush=True)

    import json

    # eligible games (winner in (0, 1)), in file order; global game ids assigned here
    work, gid, games_total = [], 0, 0
    args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        recs = [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
        sel = [(i, r) for i, r in enumerate(recs) if r.get("winner") in (0, 1)]
        if args.max_games is not None:
            sel = sel[: max(0, args.max_games - games_total)]
        if not sel:
            continue
        tag = Path(f).parent.name
        rec_ids = [i for i, _ in sel]
        game_ids = list(range(gid, gid + len(sel)))
        game_strs = [str(r.get("game_id", r.get("line_id", f"g{g}"))) for (_, r), g in zip(sel, game_ids, strict=True)]
        gid += len(sel)
        games_total += len(sel)
        out = str(args.out / f"{tag}__{_flat_name(Path(f))}.npz")
        work.append((f, out, tag, rec_ids, game_ids, game_strs, args.stride))
        if args.max_games is not None and games_total >= args.max_games:
            break
    print(f"{games_total} games selected", flush=True)

    total_games = total_states = total_pos = 0
    with mp.get_context("fork").Pool(max(1, args.jobs), initializer=_init) as pool:
        for out, n_games, n_states, n_pos in pool.imap_unordered(_shard_one, work):
            total_games += n_games
            total_states += n_states
            total_pos += n_pos
            print(f"{Path(out).name}: games {n_games} states {n_states} z+ {n_pos}", flush=True)
    neg = total_states - total_pos
    print(f"games {total_games} states {total_states} z balance +1:{total_pos} -1:{neg}", flush=True)


if __name__ == "__main__":
    main()
