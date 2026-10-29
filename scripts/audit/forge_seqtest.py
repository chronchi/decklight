"""Audit a specialist arm vs a release arm with the forge group-sequential test.

    uv run python scripts/audit/forge_seqtest.py <release arm dir> <specialist arm dir>
        [--look 256] [--K 6] [--margin 0.03] [--alpha 0.025] [--order-seed 0] [--json out.json]

Loads both arms with x4b_spec_card.games, pairs by (decks, seed, seat) in
deal_order, and runs decklight.forge.seqtest.decide on d = S - R.
Exit 2 when an arm has no games or < 90% of the release arm's keys pair up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x4b_spec_card

from decklight.forge.seqtest import deal_order, decide


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("specialist")
    ap.add_argument("--look", type=int, default=256)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rel = x4b_spec_card.games(Path(a.release))
    spe = x4b_spec_card.games(Path(a.specialist))
    if not rel:
        print("ALERT forge_seqtest: release arm has no games")
        return 2
    if not spe:
        print("ALERT forge_seqtest: specialist arm has no games")
        return 2
    common = set(rel) & set(spe)
    if len(common) < 0.9 * len(rel):
        print(f"ALERT forge_seqtest: only {len(common)} of {len(rel)} release games pair up")
        return 2
    order = deal_order(list(common), a.order_seed)
    diffs = [spe[k][0] - rel[k][0] for k in order]
    res = decide(diffs, a.look, a.K, margin=a.margin, alpha=a.alpha)
    for row in res["looks"]:
        lo, hi = row["rci_pp"]
        print(f"look {row['look']} n {row['n']} mean {row['mean_pp']:+.1f} pp "
              f"se {row['se_pp']:.2f} pp z {row['z']:.2f} b {row['boundary']:.2f} "
              f"rci [{lo:+.1f}, {hi:+.1f}] {row['decision']}")
    lo, hi = res["rci_pp"]
    print(f"DECISION {res['decision']} look {res['look']} n {res['n']} "
          f"mean {res['mean_pp']:+.1f} pp rci [{lo:+.1f}, {hi:+.1f}]")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
