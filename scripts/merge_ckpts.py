"""Weighted average of policy checkpoints (Kolibri §5 checkpoint merging / model soup), fp16 weights-only out.

    uv run python scripts/merge_ckpts.py --out data/cache/<name>/best.pt a.pt:0.5 b.pt:0.5

Weights are normalised to sum 1. All inputs must have the same keys and shapes (fail loud otherwise). The first
input's metrics.jsonl (architecture args) is copied next to the output so BCPolicy can load it. Exits 2 if the merge
equals one of its inputs (a weight of ~1 or identical inputs).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("inputs", nargs="+", help="path[:weight] (default weight 1)")
    args = ap.parse_args()
    paths, ws = [], []
    for s in args.inputs:
        p, _, w = s.partition(":")
        paths.append(Path(p))
        ws.append(float(w) if w else 1.0)
    if len(paths) < 2 or min(ws) < 0 or sum(ws) <= 0:
        raise SystemExit("need >= 2 inputs with non-negative weights")
    ws = [w / sum(ws) for w in ws]
    sds = []
    for p in paths:
        sd = torch.load(p, map_location="cpu")
        sds.append(sd.get("model", sd))
    keys = set(sds[0])
    for p, sd in zip(paths, sds, strict=True):
        if set(sd) != keys or any(sd[k].shape != sds[0][k].shape for k in keys):
            raise SystemExit(f"{p}: keys/shapes differ from {paths[0]}")
    out = {}
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            out[k] = sum(w * sd[k].float() for w, sd in zip(ws, sds, strict=True)).half()
        else:
            out[k] = sds[0][k]
    for p, sd in zip(paths, sds, strict=True):
        if all(torch.equal(out[k], sd[k].half() if sd[k].is_floating_point() else sd[k]) for k in out):
            print(f"ALERT merge equals input {p}")
            raise SystemExit(2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    m = paths[0].parent / "metrics.jsonl"
    if m.is_file():
        shutil.copy(m, args.out.parent / "metrics.jsonl")
    print(f"{args.out}: " + ", ".join(f"{p} x{w:.3f}" for p, w in zip(paths, ws, strict=True)))


if __name__ == "__main__":
    main()
