"""Stage L milestone / final summary of decisive-label search output (scripts/vast_stage_l.sh).

Reads `<dir>/p*/*.jsonl` (gen_window records with --ho 1) and prints one line: games, searched decisions per kind,
decisive-label rate, the fire rate among window labels (activation chosen), PASS-label share, labels per game.
Exit 2 (fail fast) when there are 0 labels, or the window fire rate among labels is > --max-fire (sw-v fired at 0.60
of windows, ygo-agent takes 0.24 of opponent-turn windows; NOTEBOOK 2026-10-11 21:20).

    python scripts/audit/stage_l_milestone.py runs/<name>/hw [--max-fire 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--max-fire", type=float, default=0.5)
    args = ap.parse_args()
    games, st, fired = 0, Counter(), Counter()
    for f in sorted(args.dir.glob("p*/*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            games += 1
            for e in json.loads(line).get("search", []):
                h = e.get("ho")
                if h is None:
                    continue
                st[(e["kind"], h["status"])] += 1
                if h["status"] == "label":
                    fired[e["kind"]] += bool(e["fired"])
    lab = {k: st[(k, "label")] for k in ("window", "own")}
    n = {k: st[(k, "label")] + st[(k, "undecided")] for k in ("window", "own")}
    tot_lab = sum(lab.values())
    fire_w = fired["window"] / lab["window"] if lab["window"] else 0.0
    line = (f"games {games}; searched window {n['window']} own {n['own']}; label rate window "
            f"{lab['window'] / max(n['window'], 1):.3f} own {lab['own'] / max(n['own'], 1):.3f}; window labels fire "
            f"{fire_w:.3f} (pass {1 - fire_w:.3f}); own labels activate {fired['own'] / max(lab['own'], 1):.3f}; "
            f"labels/game {tot_lab / max(games, 1):.1f}")
    print(line)
    if tot_lab == 0:
        print(f"ALERT stage_l: 0 decisive labels in {games} games")
        return 2
    if fire_w > args.max_fire:
        print(f"ALERT stage_l: window labels fire {fire_w:.3f} > {args.max_fire}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
