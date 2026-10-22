"""Context-behaviour band (retrain plan-v2 §Stage B first milestone): does the policy use quick effects from the hand
(hand traps: Maxx "C", Ash, Mulcharmy, ...) in the same contexts as the teacher?

On held-out teacher states (the train_bc val split: md5(base line id) % 1000 < 100, never trained on), at every
decision where a hand-trap activation is legal, the context cell is (phase x whose turn x T1/T2+). Per cell:
- teacher rate = share of those decisions where the teacher activated one;
- model rate = mean policy probability mass on those options (softmax over legal options).
A cell passes when the model rate is within max(2x, +-5 pp) of the teacher's: min(t/2, t-0.05) <= m <= max(2t, t+0.05).
Cells with fewer than --min-n decisions are reported, not judged. The flip test is reported only: over pairs of judged
cells whose teacher rates differ by >= 10 pp, the share where the model orders them the same way.
Exit 2 when a judged cell is out of band, or no cell can be judged (fail fast: the epoch-1 milestone stops the job).

    uv run python scripts/audit/ctx_band.py --ckpt runs/<run>/v3/ep1.pt --out runs/<run>/ctx_band-ep1.json
GPU box (vast) for the forward passes; the replay is engine-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from principles import LOC_HAND, HandTraps  # noqa: E402

from decklight.demos.dataset import iter_samples  # noqa: E402
from decklight.engine.actions import Kind  # noqa: E402
from decklight.engine.duel import Engine  # noqa: E402
from decklight.model.bc_data import base_line  # noqa: E402

PHASE = {0x1: "draw", 0x2: "standby", 0x4: "main1", 0x8: "battle", 0x10: "battle", 0x20: "battle", 0x40: "battle",
         0x80: "battle", 0x100: "main2", 0x200: "end"}


def in_val(line_id: str, frac: float = 0.1) -> bool:
    return int(hashlib.md5(base_line(line_id).encode()).hexdigest()[:8], 16) % 1000 < int(frac * 1000)


def in_band(t: float, m: float) -> bool:
    return min(t / 2, t - 0.05) <= m <= max(2 * t, t + 0.05)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--emb", default="data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--records", default="data/demos/teacher")
    ap.add_argument("--max-states", type=int, default=6000, help="stop after this many hand-trap-legal decisions")
    ap.add_argument("--max-seconds", type=float, default=1200)
    ap.add_argument("--min-n", type=int, default=40, help="judge only cells with at least this many decisions")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from decklight.eval.bc_policy import BCPolicy

    pol = BCPolicy(args.ckpt, args.emb, device=args.device, threads=4, loop_guard=False)
    engine = Engine()
    ht = HandTraps(engine)
    files = sorted(Path(args.records).glob("*.jsonl"))
    random.Random(args.seed).shuffle(files)
    cells: dict[str, dict] = defaultdict(lambda: {"n": 0, "teacher": 0.0, "model": 0.0})
    n_states, t0 = 0, time.time()
    pending: list = []

    def flush() -> None:
        keys = [k for k in pending[0][0] if k in ("cards", "card_mask", "global", "chain", "chain_mask", "actions",
                                                    "action_mask", "history", "history_mask", "hist_long",
                                                    "hist_long_mask")]  # fmt: skip
        batch = {k: np.stack([o[k] for o, *_ in pending]) for k in keys}
        lg = pol.logits(batch)
        for j, (obs, opts, fired, cell) in enumerate(pending):
            mask = obs["action_mask"].astype(bool)
            z = lg[j][: len(mask)].astype(np.float64)
            z[~mask] = -np.inf
            p = np.exp(z - z.max())
            p /= p.sum()
            c = cells[cell]
            c["n"] += 1
            c["teacher"] += float(fired)
            c["model"] += float(p[opts].sum())
        pending.clear()

    for f in files:
        if n_states >= args.max_states or time.time() - t0 > args.max_seconds:
            break
        for obs, a, _w, meta in iter_samples(engine, f, skip=lambda r: not in_val(r["line_id"])):
            acts, mask = obs["actions"], obs["action_mask"].astype(bool)
            opts = [i for i in np.flatnonzero(mask) if int(acts[i, 0]) == Kind.ACTIVATE and int(acts[i, 7]) == LOC_HAND
                    and ht(int(acts[i, 2]))]  # fmt: skip
            if not opts:
                continue
            g = obs["global"]
            cell = f"{PHASE.get(int(g[3]), str(int(g[3])))}|{'own' if int(g[5]) == 1 else 'opp'}|" \
                   f"{'T1' if int(g[2]) == 1 else 'T2+'}"  # fmt: skip
            pending.append((obs, opts, a in opts, cell))
            n_states += 1
            if len(pending) >= 256:
                flush()
    if pending:
        flush()

    rows, judged, bad = {}, [], []
    for cell, c in sorted(cells.items()):
        t, m = c["teacher"] / c["n"], c["model"] / c["n"]
        ok = in_band(t, m) if c["n"] >= args.min_n else None
        rows[cell] = {"n": c["n"], "teacher": round(t, 4), "model": round(m, 4), "in_band": ok}
        if ok is not None:
            judged.append(cell)
            if not ok:
                bad.append(cell)
    pairs = [(a, b) for i, a in enumerate(judged) for b in judged[i + 1:]
             if abs(rows[a]["teacher"] - rows[b]["teacher"]) >= 0.10]  # fmt: skip
    same = sum((rows[a]["teacher"] > rows[b]["teacher"]) == (rows[a]["model"] > rows[b]["model"]) for a, b in pairs)
    out = {"ckpt": str(args.ckpt), "states": n_states, "seconds": round(time.time() - t0), "cells": rows,
           "judged": len(judged), "out_of_band": bad,
           "flip_test": {"pairs": len(pairs), "same_order": same, "share": round(same / len(pairs), 3) if pairs else None}}
    args.out.write_text(json.dumps(out, indent=1))
    for cell, r in rows.items():
        print(f"{cell:22s} n={r['n']:5d} teacher {r['teacher']:.3f} model {r['model']:.3f} band {r['in_band']}")
    print(f"ctx_band: {n_states} states, {len(judged)} cells judged, out of band {bad or 'none'}; "
          f"flip test {same}/{len(pairs)} pairs ordered as the teacher")
    if not judged:
        print("ALERT ctx_band: no cell has enough decisions to judge")
        return 2
    if bad:
        print(f"ALERT ctx_band: out of band {bad}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
