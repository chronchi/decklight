"""Pick the best fine-tune snapshot from a validation deck_card card.json.

    python scripts/audit/ft_select.py --card runs/x/card-val/card.json --base base --out runs/x/select.json

Reads the card's real keys (arms, diff_vs_first, seats; base = --base, default the card's
first arm). Lists every non-base arm's paired diff vs base with SE/CI, chooses the arm with
the largest diff (tie -> the earlier iteration; itNNNN sorts by NNNN, final counts as last),
writes {"best": arm, "diffs": {...}} and prints one line per arm plus one PROGRESS line.

Fails loud (non-zero): no non-base arms, a NaN diff, or all diffs identical.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import NoReturn


def fail(msg: str) -> NoReturn:
    print(f"ALERT ft_select: {msg}", flush=True)
    raise SystemExit(1)


def arm_order(name: str) -> float:
    """itNNNN -> NNNN; final -> +inf (last); anything else -> +inf (ties against final)."""
    if name == "final":
        return math.inf
    if name.startswith("it"):
        try:
            return int(name[2:])
        except ValueError:
            return math.inf
    return math.inf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", type=Path, required=True)
    ap.add_argument("--base", type=str, default=None,
                    help="base arm name (default: the card's diff_vs_first.first)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        card = json.loads(args.card.read_text())
    except (OSError, ValueError) as e:
        fail(f"cannot read {args.card}: {e}")
    diffs_all = card.get("diff_vs_first")
    if not isinstance(diffs_all, dict) or "first" not in diffs_all:
        fail(f"{args.card}: no diff_vs_first.first (keys: {sorted(card)} if dict)")
    base = args.base or diffs_all["first"]
    if base != diffs_all["first"]:
        # --base overrides: diffs are still vs the card's first arm; require base == first.
        fail(f"--base {base!r} != card first arm {diffs_all['first']!r} (paired diffs are vs first)")
    if base not in card.get("arms", {}):
        fail(f"base arm {base!r} not in arms {sorted(card.get('arms', {}))}")
    # diff_vs_first holds {"first": base, arm: {...}}; non-base arms are every other key.
    arms = [k for k in diffs_all if k != "first"]
    if not arms:
        fail(f"{args.card}: no non-base arms (only {diffs_all['first']!r})")

    diffs: dict[str, dict] = {}
    for name in arms:
        d = diffs_all[name]
        for k in ("diff_pp", "se_pp"):
            v = d.get(k)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                fail(f"arm {name}: {k}={v!r} is NaN/non-finite")
        diffs[name] = d

    vals = [float(diffs[a]["diff_pp"]) for a in arms]
    if len(set(vals)) == 1 and len(arms) > 1:
        fail(f"all {len(arms)} diffs identical ({vals[0]})")
    if len(arms) == 1:
        best = arms[0]
    else:
        top = max(vals)
        tied = [a for a in arms if float(diffs[a]["diff_pp"]) == top]
        best = min(tied, key=arm_order)

    for name in sorted(arms, key=arm_order):
        d = diffs[name]
        print(f"{name}: {d['diff_pp']:+.2f} pp [{d['ci_pp'][0]:+.2f}, {d['ci_pp'][1]:+.2f}] "
              f"(SE {d['se_pp']:.2f}) n={d.get('n')}", flush=True)
    print(f"PROGRESS select best={best} diff={diffs[best]['diff_pp']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"best": best, "diffs": diffs}, indent=1))


if __name__ == "__main__":
    main()
