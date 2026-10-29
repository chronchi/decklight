"""Forge F0 / E1: the specialist's gain over the release by fine-tune iteration (scripts/vast_forge_f0.sh card).

Arms in <card>/: R-<arch> (the release, D) and S<it>-<arch> (the specialist snapshot at iteration it), all on the
same deals (decks/forge/f0/pairs/arch-<arch>.pairs, same seeds), opponent D raw. Per archetype and iteration: S - R
paired (all, train lists, held-out list). Pooled over archetypes per iteration. Pre-registered budget rule (NOTEBOOK
2026-10-29 "Forge F0"): the budget is the smallest iteration whose pooled gain is >= 90% of the best iteration's
pooled gain and within 2 pp of it; the same rule per archetype is reported too. Writes <card>/../curve_card.json.
Exit 2 (via x4b_spec_card.paired) when an arm is missing, games do not pair, or a snapshot plays R's moves.

    python scripts/audit/forge_curve_card.py runs/<job>/card --split decks/forge/f0/split.json --its 20 40 60 80 120 160
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x4b_spec_card import games, paired, stats


def knee(gains: dict[int, float | None]) -> int | None:
    ok = {i: g for i, g in gains.items() if g is not None}
    if not ok:
        return None
    best = max(ok.values())
    if best <= 0:
        return None
    return min(i for i, g in ok.items() if g >= 0.9 * best and g >= best - 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("card", type=Path)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--its", type=int, nargs="+", required=True)
    a = ap.parse_args()
    split = json.loads(a.split.read_text())
    res: dict = {"archetypes": {}, "pooled": {}}
    pooled: dict[int, list[float]] = {i: [] for i in a.its}
    for arch in split["archs"]:
        if not (a.card / f"R-{arch}").is_dir():
            print(f"ALERT forge_curve_card: R-{arch} missing; archetype skipped")
            continue
        r = games(a.card / f"R-{arch}")
        held = split["heldout"][arch]
        row = {"R_wr": None, "its": {}}
        for it in a.its:
            d = a.card / f"S{it}-{arch}"
            if not d.is_dir():
                print(f"ALERT forge_curve_card: {d.name} missing")
                continue
            s = games(d)
            keys, diffs = paired(r, s, f"{arch} it{it}")
            row["R_wr"] = round(sum(r[k][0] for k in keys) / len(keys), 4)
            row["its"][it] = {"S_wr": round(sum(s[k][0] for k in keys) / len(keys), 4), "all": stats(diffs),
                              "train": stats([x for k, x in zip(keys, diffs, strict=True) if r[k][1] != held]),
                              "heldout": stats([x for k, x in zip(keys, diffs, strict=True) if r[k][1] == held])}
            pooled[it] += diffs
        row["knee"] = knee({it: v["all"]["diff_pp"] for it, v in row["its"].items()})
        res["archetypes"][arch] = row
        print(f"{arch}: R {row['R_wr']}  " + "  ".join(
            f"it{it} {v['all']['diff_pp']:+.1f} [{v['all']['ci_pp'][0]:+.1f},{v['all']['ci_pp'][1]:+.1f}]"
            f" (held {v['heldout']['diff_pp'] if v['heldout']['n'] else '-'})" for it, v in row["its"].items())
            + f"  knee it{row['knee']}")
    for it, d in pooled.items():
        res["pooled"][it] = stats(d)
    res["budget"] = knee({it: v["diff_pp"] for it, v in res["pooled"].items()})
    print("pooled: " + "  ".join(f"it{it} {v['diff_pp']:+.1f} (n={v['n']})" for it, v in res["pooled"].items()
                                 if v["n"]))
    print(f"BUDGET it{res['budget']} (smallest iteration with >= 90% of the best pooled gain and within 2 pp)")
    (a.card.parent / "curve_card.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
