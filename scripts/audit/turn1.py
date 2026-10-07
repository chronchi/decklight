"""Turn-1 activity of the player going first, policy vs its opponent, from rendered game.txt files (engine-free).
For each game: who went first (P0), how many non-PASS actions P0 took in turn 1 (summons/activations/sets), and
whether P0 lost by turn 2 (killed on the opponent's first turn).
Usage: python scripts/audit/turn1.py <games dir>/<match>"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    stats = {"policy": [], "opponent": []}
    for f in sorted(Path(sys.argv[1]).glob("*/game.txt")):
        lines = f.read_text().splitlines()
        seat = int(re.search(r"policy seat P(\d)", lines[0]).group(1))
        win = re.search(r"winner P(\d)", lines[0])
        turns = int(re.search(r"turns (\d+)", lines[1]).group(1))
        t1 = [ln for ln in lines if ln.startswith("T1 P0 ")]
        acts = [ln for ln in t1 if re.match(r"T1 P0 (SUMMON|SPSUMMON|ACTIVATE|SSET|MSET|SET)\b", ln)]
        who = "policy" if seat == 0 else "opponent"
        p0_lost_by_t2 = bool(win) and win.group(1) == "1" and turns <= 2
        stats[who].append((len(acts), p0_lost_by_t2, bool(win) and win.group(1) == "0"))
    for who, v in stats.items():
        if not v:
            continue
        acts = sorted(a for a, _, _ in v)
        print(f"{who:>8} goes first: n={len(v)}  turn-1 actions median {acts[len(acts) // 2]} "
              f"(p25 {acts[len(acts) // 4]}, p75 {acts[3 * len(acts) // 4]})  "
              f"killed by turn 2: {sum(k for _, k, _ in v) / len(v):.2f}  first player wins: {sum(w for *_, w in v) / len(v):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
