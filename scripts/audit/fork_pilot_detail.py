"""Detail of fork_pilot rows (M1, NOTEBOOK 2026-10-20): where does the fork signal come from?

Per category:
  decided  share of rows where every candidate gives the same result on every sample (the choice cannot matter here);
  flat     share of rows where the candidates' mean results are all equal (no gap at all);
  |gap|    distribution of best-vs-played gaps (all worlds, end units) and of the paired SE;
  sig+     rows where the best alternative beats played by >= 2 SE; sig- where played beats every alternative by >= 2 SE;
  base     mean result of the played option (how often the actor wins from here).
    python scripts/audit/fork_pilot_detail.py runs/<forkpilot>/rows
Exits 2 on 0 usable rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ygo_zero.search.honest import paired_gap


def main() -> int:
    stem = Path(sys.argv[1])
    rows = [
        json.loads(x)
        for f in sorted(stem.parent.glob(f"{stem.name}-*.jsonl"))
        for x in f.read_text().splitlines()
        if x.strip()
    ]
    rows = [r for r in rows if "error" not in r]
    if not rows:
        print("ALERT fork_pilot_detail: 0 usable rows")
        return 2
    for cat in ("ht", "place", "t1", "any"):
        rs = [r for r in rows if cat in r["tags"]]
        if not rs:
            continue
        decided = flat = sigp = sigm = 0
        gaps, ses, base = [], [], []
        for r in rs:
            end = [np.asarray(c, float) for c in r["end"]]
            means = [float(c.mean()) for c in end]
            decided += all(np.array_equal(end[0], c) for c in end) and len(set(end[0].tolist())) == 1
            flat += len(set(means)) == 1
            base.append(means[0])
            best = max(range(1, len(end)), key=lambda j: means[j]) if len(end) > 1 else 0
            m, se, _ = paired_gap(end[best].tolist(), end[0].tolist())
            gaps.append(abs(m))
            ses.append(se if np.isfinite(se) else 0.0)
            sigp += bool(np.isfinite(se) and se > 0 and m >= 2 * se)
            worst = [paired_gap(end[0].tolist(), end[j].tolist()) for j in range(1, len(end))]
            sigm += bool(worst) and all(np.isfinite(s) and s > 0 and g >= 2 * s for g, s, _ in worst)
        n = len(rs)
        q = np.percentile(gaps, [50, 75, 90]).round(3).tolist()
        print(
            f"{cat:6s} n {n:4d}  decided {decided / n:.2f}  flat {flat / n:.2f}  |gap| p50/75/90 {q}  "
            f"se p50 {np.median(ses):.3f}  sig+ {sigp / n:.3f}  sig- {sigm / n:.3f}  base {np.mean(base):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
