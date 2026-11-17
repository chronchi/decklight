"""How often the demonstration corpora interrupt on the opponent's turn, by chain length (NOTEBOOK 2026-11-05 05:45).

For every sample of a demo directory (`iter_samples`, engine replay only, no model) where it is NOT the acting
player's turn and an activation is legal: bucket by chain length at the decision (0 = open game state, 1, 2+) and
count how often the demonstrated move is an activation.

    uv run python scripts/audit/opp_window_prior.py data/demos/teacher data/demos/gold --max-records 300
Exit 3: a directory with 0 opponent-turn windows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from decklight.demos.dataset import iter_samples
from decklight.engine.actions import Kind
from decklight.engine.duel import Engine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--max-samples", type=int, default=200_000)
    a = ap.parse_args()
    engine, out, bad = Engine(), {}, 0
    for d in a.dirs:
        c: Counter = Counter()
        for obs, act, _w, _m in itertools.islice(iter_samples(engine, d), a.max_samples):
            if int(obs["global"][5]) == 1:  # the acting player's own turn
                continue
            legal = np.flatnonzero(np.asarray(obs["action_mask"]).astype(bool))
            if not any(int(obs["actions"][i][0]) == Kind.ACTIVATE for i in legal):
                continue
            ch = min(int(np.asarray(obs["chain_mask"]).sum()), 2)
            c[(ch, "n")] += 1
            c[(ch, "act")] += int(int(obs["actions"][int(act)][0]) == Kind.ACTIVATE)
        r = {f"chain{ch}{'+' if ch == 2 else ''}": {"n": c[(ch, "n")], "activate": c[(ch, "act")],
             "rate": round(c[(ch, "act")] / max(c[(ch, "n")], 1), 3)} for ch in (0, 1, 2)}  # fmt: skip
        out[str(d)] = r
        print(d, json.dumps(r), flush=True)
        bad += sum(v["n"] for v in r.values()) == 0
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
