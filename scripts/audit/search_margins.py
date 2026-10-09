"""Override margins and rollout noise of a searched search_match arm (per-sample logs, NOTEBOOK 2026-10-09 08:40).

    python scripts/audit/search_margins.py runs/<job>/hppo [--margins 0.05,0.1,0.2,0.3]

Per searched decision: margin = best score - the policy move's score (when the policy move is a candidate), and the
noise SE of that gap from the per-sample scores (sqrt(var_best/n + var_pol/n)). Prints the margin distribution of
overrides, the share with margin >= z*SE, and how many overrides each candidate --margin would keep. Also an
offline budget curve: how often the best candidate with the first s samples equals the best with all samples.
Exits 1 if no searched decision carries samples (the arm was not run with per-sample logging).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", help="search_match --out dir (games-*.jsonl with 'search')")
    ap.add_argument("--margins", default="0.02,0.05,0.1,0.2,0.3")
    a = ap.parse_args()
    dec = []
    for f in sorted(glob.glob(f"{a.arm}/games-*.jsonl")):
        with open(f) as fh:
            rows = [json.loads(line) for line in fh]
        for r in rows:
            for e in r.get("search", []):
                if e.get("samples") and all(e["samples"]):
                    dec.append((r, e))
    if not dec:
        print(f"no searched decisions with samples in {a.arm}")
        sys.exit(1)
    over, gaps, zs = 0, [], []
    for _, e in dec:
        c, sm = e["cands"], e["samples"]
        means = [sum(x) / len(x) for x in sm]
        best = max(range(len(c)), key=lambda j: (means[j], -j))
        if e["played"] not in c or c[best] == e["played"]:
            continue
        over += 1
        p = c.index(e["played"])
        g = means[best] - means[p]
        n = len(sm[best])
        se = math.sqrt((st.pvariance(sm[best]) + st.pvariance(sm[p])) / max(n, 1))
        gaps.append(g)
        zs.append(g / se if se > 0 else float("inf"))
    n = len(dec)
    print(f"searched decisions {n}, overrides {over} ({over / n:.1%})")
    if gaps:
        q = st.quantiles(gaps, n=10)
        print(f"override margin: median {st.median(gaps):.3f}, deciles {[round(x, 3) for x in q]}")
        print(f"overrides with gap >= 1 SE {sum(z >= 1 for z in zs) / over:.1%}, >= 2 SE {sum(z >= 2 for z in zs) / over:.1%}")
        for m in (float(x) for x in a.margins.split(",")):
            k = sum(g >= m for g in gaps)
            print(f"  --margin {m}: keeps {k}/{over} overrides ({k / over:.1%}), {k / n:.1%} of searched decisions")
    # Offline budget curve: best with the first s samples vs best with all.
    smax = min(len(e["samples"][0]) for _, e in dec)
    for s in range(1, smax + 1):
        agree = 0
        for _, e in dec:
            full = [sum(x) / len(x) for x in e["samples"]]
            part = [sum(x[:s]) / s for x in e["samples"]]
            fb = max(range(len(full)), key=lambda j: (full[j], -j))
            pb = max(range(len(part)), key=lambda j: (part[j], -j))
            agree += fb == pb
        print(f"samples {s}/{smax}: same best as {smax} samples in {agree / n:.1%}")


if __name__ == "__main__":
    main()
