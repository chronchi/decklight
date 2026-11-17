#!/usr/bin/env python3
"""Score a checkpoint's behaviour on fixed decision windows against X (seconds, laptop).

    uv run python scripts/weightedit/score.py --ckpt data/cache/weightedit/<agent>/edit.pt --windows 'data/cache/weightedit/windows/dev-*.npz'

Reports, per window kind:
- attack (my battle step with an attack legal): probability mass on ATTACK options, and how often the argmax attacks;
- interrupt (opponent's turn, chain window with an activation legal): the same for ACTIVATE; interrupt_t1: the same
  on turn 1 only (the opponent's opening combo), every such window;
- other (every other decision): KL(X || edit) and argmax agreement with X: how much the rest of play changed.

Also the number of changed parameters. Exit 3: 0 windows, or NaN in the edited net's output.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from decklight.eval.weightedit import changed_params, load_policy, load_windows, score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", default="data/cache/weightedit/windows/dev-*.npz")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-per-kind", type=int, default=1500, help="score a fixed random subset of each kind (0 = all)")
    ap.add_argument("--quick", action="store_true", help="300 windows per kind (~20 s on one core)")
    a = ap.parse_args()
    paths = sorted(glob.glob(a.windows))
    if not paths:
        print(f"ALERT score: no window files match {a.windows}", flush=True)
        return 3
    w = load_windows([Path(p) for p in paths], x_probs=True)
    cap, rng, keep = (300 if a.quick else a.max_per_kind), np.random.default_rng(0), []
    for kind in ("attack", "interrupt", "battle", "other"):  # the same subset every run
        idx = np.flatnonzero(w["meta_kind"] == kind)
        keep.append(rng.choice(idx, cap, replace=False) if cap and len(idx) > cap else idx)
    t1 = np.flatnonzero((w["meta_kind"] == "interrupt") & (w["global"][:, 2] == 1))  # all of them (few: ~300 on dev)
    w1 = {k: v[t1] for k, v in w.items()}
    w = {k: v[np.sort(np.concatenate(keep))] for k, v in w.items()}
    pol = load_policy(a.ckpt, threads=1)
    res = score(pol, None, w)
    res["interrupt_t1"] = score(pol, None, w1)["interrupt_t1"]  # the opponent's opening turn, every window
    res["params"] = changed_params(a.ckpt)
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    flat = [v for d in res.values() if isinstance(d, dict) for v in d.values() if isinstance(v, float)]
    if not len(w["meta_kind"]) or any(math.isnan(v) for v in flat):
        print("ALERT score: 0 windows or NaN", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
