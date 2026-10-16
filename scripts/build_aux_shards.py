"""Aux-target shards for the evaluator heads (engine plan N3, `src/ygo_zero/eval/aux.py`).

Replays game records (the gold/selfplay record format written by
`scripts/gen_selfplay.py`: config + choices + winner) in the C++ env and keeps
every stride-th decision (offset random per game, seeded) plus always the first
decision of each turn; decisions with a single legal option are skipped (no
choice, no signal) -- the same keep rule as `scripts/build_value_shards.py`.

Per kept state: the OBS_KEYS arrays (same conventions as the BC/value shards),
the aux targets + has_next mask from `aux.extract_targets` over the full
per-decision trace, turn, game (global int id; the shard's game_ids side table
maps it to the game_id string), deck (the actor's stem), deck0/deck1, actor and
src. Winner-3 (truncated) games are dropped: without a game end there is no
next-own-turn target and no `ends` label.

    uv run python scripts/build_aux_shards.py --records 'data/demos/teacher/*.jsonl' \\
        --out data/cache/aux_shards --stride 4 --jobs 4
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

from ygo_zero.eval.aux import extract_targets
from ygo_zero.model.bc_data import OBS_KEYS

EXTRA_KEYS = ("d_lp", "d_cards", "ends", "turns_left", "mask", "turn", "game", "deck", "deck0",
              "deck1", "actor", "src")

_ENGINE = None


def _init() -> None:
    global _ENGINE
    from ygo_zero.engine.duel import Engine

    _ENGINE = Engine()


def _offset(game_id: str, stride: int) -> int:
    """Deterministic per-game stride offset in [0, stride)."""
    seed = hashlib.md5(game_id.encode()).hexdigest()
    return random.Random(seed).randrange(max(stride, 1))


def _shard_one(job: tuple) -> tuple[str, int, int, int]:
    """Replay a file's selected records; returns (out_file, games, states, masked)."""
    from ygo_zero.demos.dataset import record_config
    from ygo_zero.engine.cpp_env import CppEnv

    src_file, out_file, src_tag, rec_ids, game_ids, game_strs, stride = job
    import json

    recs = [json.loads(line) for line in Path(src_file).read_text().splitlines() if line.strip()]
    cols: dict[str, list] = {k: [] for k in OBS_KEYS}
    turn, game, deck, deck0, deck1, actor = [], [], [], [], [], []
    tgt_lp, tgt_cards, tgt_ends, tgt_turns, tgt_mask = [], [], [], [], []
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
            g_list, a_list, keep_at = [], [], []
            kept_obs: list[dict] = []
            kept_meta: list[tuple] = []
            for k, c in enumerate(rec["choices"]):
                if obs is None:
                    break
                me = int(env.player)
                t = int(obs["global"][2])
                g_list.append(obs["global"].copy())
                a_list.append(me)
                n_legal = int(obs["action_mask"].sum())
                first = t not in seen_turns
                seen_turns.add(t)
                if n_legal >= 2 and ((k + off) % stride == 0 or first):
                    keep_at.append(len(g_list) - 1)
                    kept_obs.append({key: obs[key].copy() for key in OBS_KEYS})
                    decks = rec.get("decks") or []
                    d0 = str(decks[0]) if len(decks) > 0 else ""
                    d1 = str(decks[1]) if len(decks) > 1 else ""
                    kept_meta.append((t, me, str(decks[me]) if me < len(decks) else "", d0, d1))
                obs = env.step(int(c))
            if not keep_at:
                continue
            end_turn = rec.get("end_turn")
            if end_turn is None:
                end_turn = max(int(g[2]) for g in g_list) if g_list else 0
            tgts = extract_targets(np.stack(g_list), np.array(a_list, np.int64), int(winner), int(end_turn))
            for pos, ob, meta in zip(keep_at, kept_obs, kept_meta, strict=True):
                for key in OBS_KEYS:
                    cols[key].append(ob[key])
                tgt_lp.append(tgts["d_lp"][pos])
                tgt_cards.append(tgts["d_cards"][pos])
                tgt_ends.append(float(tgts["ends"][pos]))
                tgt_turns.append(float(tgts["turns_left"][pos]))
                tgt_mask.append(bool(tgts["mask"][pos]))
                t, me, d, d0, d1 = meta
                turn.append(t)
                game.append(gid)
                deck.append(d)
                deck0.append(d0)
                deck1.append(d1)
                actor.append(me)
            game_str_of[gid] = str(gstr)
            n_games += 1
        except Exception as e:  # noqa: BLE001 - one bad record must not kill a shard build
            print(f"aux_shards: skipping {rec.get('game_id', rec.get('line_id'))} in {src_file}: {e!r}"[:300], flush=True)
            continue
    n_states = len(turn)
    if not n_states:
        return out_file, n_games, 0, 0
    gids = sorted(game_str_of)
    np.savez_compressed(
        out_file,
        **{k: np.stack(v) for k, v in cols.items()},
        d_lp=np.stack(tgt_lp).astype(np.float32),
        d_cards=np.stack(tgt_cards).astype(np.float32),
        ends=np.array(tgt_ends, dtype=np.float32),
        turns_left=np.array(tgt_turns, dtype=np.float32),
        mask=np.array(tgt_mask, dtype=bool),
        turn=np.array(turn, dtype=np.int32),
        game=np.array(game, dtype=np.int32),
        deck=np.array(deck),
        deck0=np.array(deck0),
        deck1=np.array(deck1),
        actor=np.array(actor, dtype=np.int8),
        src=np.array([src_tag] * n_states),
        game_ids=np.array([game_str_of[g] for g in gids]),
        game_row=np.array(gids, dtype=np.int32),
    )
    return out_file, n_games, n_states, int(np.array(tgt_mask).sum())


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

    total_games = total_states = total_masked = 0
    with mp.get_context("fork").Pool(max(1, args.jobs), initializer=_init) as pool:
        for out, n_games, n_states, n_masked in pool.imap_unordered(_shard_one, work):
            total_games += n_games
            total_states += n_states
            total_masked += n_masked
            print(f"{Path(out).name}: games {n_games} states {n_states} has_next {n_masked}", flush=True)
    print(f"games {total_games} states {total_states} has_next {total_masked}", flush=True)


if __name__ == "__main__":
    main()
