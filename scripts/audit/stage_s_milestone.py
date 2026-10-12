"""Stage S first-milestone / progress check (retrain plan v2): ho games so far vs the stored hppo games.

Pairs by (decks, seed, s_seat), prints one PROGRESS line: n paired, win diff (ho - hppo) with paired SE, override rate,
ho status counts, branches per search (rollout cost). Exits 2 when branches per search > --max-cost x hppo's (the plan's
kill rule: adaptive sampling must not cost more than 1.4x hppo), or when a paired-games invariant breaks.

Usage: python scripts/audit/stage_s_milestone.py <ho games dir> <hppo games dir> [--min-games 150] [--max-cost 1.4]
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("games-*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[(tuple(r["decks"]), int(r["seed"]), int(r["s_seat"]))] = r
    return out


def win(r: dict) -> float:
    w = r.get("winner")
    return 1.0 if w == r["s_seat"] else 0.0 if w in (0, 1) else 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ho", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--min-games", type=int, default=150)
    ap.add_argument("--max-cost", type=float, default=1.4)
    a = ap.parse_args()
    ho, ref = load(a.ho), load(a.ref)
    keys = [k for k in ho if k in ref]
    if len(ho) < a.min_games:
        print(f"milestone: only {len(ho)} ho games")
        return
    if len(keys) < 0.9 * len(ho):
        print(f"ALERT stage_s: only {len(keys)}/{len(ho)} ho games pair with the hppo reference")
        sys.exit(2)
    d = [win(ho[k]) - win(ref[k]) for k in keys]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")

    def cost(recs):
        s = sum(r["n_searched"] for r in recs)
        return sum(r["n_branches"] for r in recs) / max(s, 1), sum(r["n_override"] for r in recs) / max(s, 1), s

    c_ho, ov_ho, s_ho = cost([ho[k] for k in keys])
    c_ref, ov_ref, _ = cost([ref[k] for k in keys])
    st = {}
    for k in keys:
        for e in ho[k].get("search", []):
            h = e.get("ho")
            if h:
                st[h["status"]] = st.get(h["status"], 0) + 1
    if s_ho and not st:
        print("ALERT stage_s: searched decisions carry no ho log (is --ho 1 set?)")
        sys.exit(2)
    print(f"milestone n={n} ho-hppo {m:+.3f} (SE {se:.3f}); override {ov_ho:.3f} vs hppo {ov_ref:.3f}; "
          f"branches/search {c_ho:.1f} vs {c_ref:.1f} ({c_ho / max(c_ref, 1e-9):.2f}x); ho {st}")
    if c_ho > a.max_cost * c_ref:
        print(f"ALERT stage_s: branches/search {c_ho:.1f} > {a.max_cost}x hppo {c_ref:.1f}")
        sys.exit(2)


if __name__ == "__main__":
    main()
