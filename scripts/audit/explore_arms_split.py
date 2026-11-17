"""Split `explore_arms.py` results as pre-registered (NOTEBOOK 2026-11-05 15:55 amendment): windows with one legal
activation (rand = "activate that card") vs >= 2. The shard rows do not store the number of legal activations, so a
window counts as >= 2 when its rand draws hit more than one distinct option (8 draws: a 2-option window looks single
with p = 2 * 0.5**8 < 1%). Also splits by whether search's move is an activation (search index among the draws).

    uv run python scripts/audit/explore_arms_split.py runs/explore-arms-20261105/s*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_alt import sign_p


def diff_stats(d: list[float], seed: int = 0) -> dict:
    x = np.asarray(d, float)
    if not len(x):
        return {"n": 0}
    boot = x[np.random.default_rng(seed).integers(0, len(x), (2000, len(x)))].mean(1)
    b, w = int((x > 0).sum()), int((x < 0).sum())
    return {"n": len(x), "mean": round(float(x.mean()), 4),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "pos": b, "neg": w, "sign_p_pos": round(sign_p(b, w), 4), "sign_p_neg": round(sign_p(w, b), 4)}  # fmt: skip


def main() -> int:
    rows = [r for f in sys.argv[1:] for r in json.loads(Path(f).read_text())["rows"] if r.get("variants")]
    if not rows:
        print("ALERT explore_arms_split: no rows with variants", flush=True)
        return 3
    groups: dict[str, list[dict]] = {}
    for r in rows:
        vs = r["variants"]
        idx = {d["index"] for v in vs for d in v["rand"]}
        multi = "acts>=2" if len(idx) > 1 else "acts=1"
        s_act = "search_activates" if r["search_index"] in idx else "search_other"
        w = {arm: float(np.mean([v[arm]["won"] for v in vs])) for arm in ("p", "search")}
        w["rand"] = float(np.mean([d["won"] for v in vs for d in v["rand"]]))
        f = {arm: float(np.mean([v[arm]["opp_field_end_turn"] for v in vs])) for arm in ("p", "search")}
        f["rand"] = float(np.mean([d["opp_field_end_turn"] for v in vs for d in v["rand"]]))
        item = {"w": w, "f": f}
        for g in ("all", multi, s_act, f"{multi}|{s_act}"):
            groups.setdefault(g, []).append(item)
    out = {}
    for g, items in sorted(groups.items()):
        out[g] = {"windows": len(items),
                  "won": {a: round(float(np.mean([i["w"][a] for i in items])), 4) for a in ("p", "search", "rand")},
                  "won_rand_minus_p": diff_stats([i["w"]["rand"] - i["w"]["p"] for i in items]),
                  "won_search_minus_p": diff_stats([i["w"]["search"] - i["w"]["p"] for i in items]),
                  "won_search_minus_rand": diff_stats([i["w"]["search"] - i["w"]["rand"] for i in items]),
                  "field_rand_minus_p": diff_stats([i["f"]["rand"] - i["f"]["p"] for i in items]),
                  "field_search_minus_p": diff_stats([i["f"]["search"] - i["f"]["p"] for i in items])}  # fmt: skip
        print(g, json.dumps(out[g]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
