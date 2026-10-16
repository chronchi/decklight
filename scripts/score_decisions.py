"""Score every decision of stored games with the value net (engine plan N2a).

Reads `games-*.jsonl` records (e.g. search_match output), replays each game
through the C++ env, collects the actor-view obs at every decision, batches
obs across games through the value net, and writes one line per game:

    {game_idx, seed, decks, s_seat, k: [...], actor: [...], turn: [...], v: [...]}

v is the actor's-view win estimate in [-1, 1], rounded to 4 decimals.

    taskset -c 0-3 uv run python scripts/score_decisions.py \\
        --games runs/<job>/hppo-p2 --value data/cache/value_p2/best.pt \\
        --out values.jsonl [--device cuda] [--max-games N] [--batch 2048] [--shard 0/4]

--shard i/n scores games i::n, so several processes split a box.
No model inference on the laptop per the brief: this script loads a real
checkpoint (vast GPU); unit tests use a fake value object instead.

Invariants (exit 2): 0 games read; a game with 0 decisions; a replay whose
winner diverges from its record; NaN in the values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import NoReturn

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def fail(msg: str) -> NoReturn:
    print(f"ALERT score_decisions: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def parse_shard(s: str) -> tuple[int, int]:
    try:
        i, n = s.split("/")
        i, n = int(i), int(n)
    except ValueError:
        fail(f"--shard {s!r}: want i/n")
    if not (0 <= i < n):
        fail(f"--shard {s!r}: need 0 <= i < n")
    return i, n


def iter_records(dirs: list[Path], max_games: int):
    n = 0
    for d in dirs:
        files = sorted(d.glob("games-*.jsonl"))
        if not files:
            print(f"ALERT score_decisions: no games-*.jsonl in {d}", file=sys.stderr, flush=True)
        for f in files:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                yield json.loads(line)
                n += 1
                if max_games and n >= max_games:
                    return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", required=True, type=Path)
    ap.add_argument("--value", required=True, type=Path, help="learned value checkpoint")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2048, help="max obs held / scored at once")
    ap.add_argument("--shard", default="0/1", help="i/n: score games i::n")
    a = ap.parse_args()

    for d in a.games:
        if not d.is_dir():
            fail(f"no such dir: {d}")
    si, sn = parse_shard(a.shard)

    from ygo_zero.eval.value import ValueNet

    t0 = time.time()
    value = ValueNet(a.value, a.emb, device=a.device, batch=min(a.batch, 512)).values

    import render_showcase as R

    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig, Engine

    engine = Engine()
    recs = [r for i, r in enumerate(iter_records(a.games, a.max_games)) if i % sn == si]
    if not recs:
        fail("0 games read (after --shard/--max-games)")
    for r in recs:
        if not r.get("choices"):
            fail(f"game {r.get('game_idx')}: 0 decisions")

    out_lines: list[str] = []
    pend_obs: list[dict] = []
    pend_pos: list[tuple[int, int]] = []  # (line idx, decision idx)
    per_game: list[dict] = []  # parallel to recs: k/actor/turn/v lists
    for r in recs:
        per_game.append(
            {
                "game_idx": r.get("game_idx"),
                "seed": r.get("seed"),
                "decks": list(r.get("decks", [])),
                "s_seat": r.get("s_seat"),
                "k": [],
                "actor": [],
                "turn": [],
                "v": [],
            }
        )

    def flush() -> None:
        if not pend_obs:
            return
        buf = {k: np.stack([o[k] for o in pend_obs]) for k in pend_obs[0]}
        v = value(buf)
        if len(v) != len(pend_obs):
            fail(f"value returned {len(v)} scores for {len(pend_obs)} obs")
        if bool(np.isnan(np.asarray(v)).any()):
            fail("NaN in value scores")
        for (li, di), x in zip(pend_pos, v, strict=True):
            per_game[li]["v"][di] = round(float(np.clip(x, -1, 1)), 4)
        pend_obs.clear()
        pend_pos.clear()

    n_dec = 0
    for li, rec in enumerate(recs):
        d0, d1 = rec["decks"]
        cfg = DuelConfig(load_ydk(R.deck_path(d0)), load_ydk(R.deck_path(d1)), seed=int(rec["seed"]))
        env = CppEnv(engine, cfg)
        obs = env.reset()
        g = per_game[li]
        for k, ch in enumerate(rec["choices"]):
            if obs is None:
                if rec.get("truncated"):
                    break
                fail(f"game {rec.get('game_idx')}: replay ended before choice {k}")
            actor = int(env.player)
            turn = int(obs["global"][2])
            g["k"].append(k)
            g["actor"].append(actor)
            g["turn"].append(turn)
            g["v"].append(0.0)  # placeholder, filled by flush
            pend_obs.append({k2: np.array(v2, copy=True) for k2, v2 in obs.items()})
            pend_pos.append((li, len(g["v"]) - 1))
            n_dec += 1
            if len(pend_obs) >= a.batch:
                flush()
            obs = env.step(int(ch))
        if not rec.get("truncated") and (obs is not None or int(env.winner) != int(rec["winner"])):
            fail(f"game {rec.get('game_idx')}: replay winner {env.winner} != record {rec['winner']}")
        if not g["k"]:
            fail(f"game {rec.get('game_idx')}: 0 decisions replayed")
        if (li + 1) % 100 == 0:
            dt = time.time() - t0
            print(
                f"{li + 1}/{len(recs)} games, {n_dec} decisions, "
                f"{(li + 1) / dt:.1f} games/s, {n_dec / dt:.0f} decisions/s",
                flush=True,
            )
    flush()

    for g in per_game:
        if any(not np.isfinite(x) for x in g["v"]):
            fail(f"game {g['game_idx']}: NaN in final values")
        out_lines.append(json.dumps(g))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(out_lines) + "\n")
    dt = time.time() - t0
    print(
        f"scored {len(per_game)} games, {n_dec} decisions in {dt:.0f} s "
        f"({len(per_game) / dt:.1f} games/s, {n_dec / dt:.0f} decisions/s) -> {a.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
