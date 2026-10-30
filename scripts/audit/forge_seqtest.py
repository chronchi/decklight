"""Audit a specialist arm vs a release arm with the forge group-sequential test.

Single-block (original) form: one release arm dir and one specialist arm dir as
positionals; every paired game is used, in deal_order(keys, order_seed).

Multi-block (sequential looks on fresh deals) form: one arm dir per look, comma-separated:

    forge_seqtest.py --release R_L1,R_L2,... --specialist S_L1,S_L2,... --look <L> [...]

Block k pairs R_Lk against S_Lk, orders the keys with deal_order(keys, order_seed + k)
(k = 1, 2, ...), and takes the first L of them. Blocks concatenate in look order
(block 1's L diffs, then block 2's L, ...); decide(diffs, look=L, K, margin, alpha)
then evaluates look k on exactly blocks 1..k, so re-evaluating at look k+1 reproduces
the decision path of looks 1..k. Loads both arms with x4b_spec_card.games, paired by
(decks, seed, seat). Exit 2 when an arm has no games or < 90% of the release arm's keys
pair up (per block), or when a block holds fewer than L paired games.

    uv run python scripts/audit/forge_seqtest.py <release arm dir> <specialist arm dir>
        [--look 256] [--K 6] [--margin 0.03] [--alpha 0.025] [--order-seed 0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x4b_spec_card

from decklight.forge.seqtest import deal_order, decide


def block_diffs(rel_dir: Path, spe_dir: Path, look: int, seed: int, tag: str) -> list[float]:
    """Paired S - R diffs for one block: first `look` diffs in deal order."""
    rel = x4b_spec_card.games(rel_dir)
    spe = x4b_spec_card.games(spe_dir)
    if not rel:
        print(f"ALERT forge_seqtest: {tag}: release arm {rel_dir} has no games")
        raise SystemExit(2)
    if not spe:
        print(f"ALERT forge_seqtest: {tag}: specialist arm {spe_dir} has no games")
        raise SystemExit(2)
    common = set(rel) & set(spe)
    if len(common) < 0.9 * len(rel):
        print(f"ALERT forge_seqtest: {tag}: only {len(common)} of {len(rel)} release games pair up")
        raise SystemExit(2)
    if len(common) < look:
        print(f"ALERT forge_seqtest: {tag}: block has {len(common)} paired games, fewer than look {look}")
        raise SystemExit(2)
    order = deal_order(list(common), seed)
    return [spe[k][0] - rel[k][0] for k in order[:look]]


def report(res: dict, blocks: int, json_path: str | None) -> int:
    for row in res["looks"]:
        lo, hi = row["rci_pp"]
        print(f"look {row['look']} n {row['n']} mean {row['mean_pp']:+.1f} pp "
              f"se {row['se_pp']:.2f} pp z {row['z']:.2f} b {row['boundary']:.2f} "
              f"rci [{lo:+.1f}, {hi:+.1f}] {row['decision']}")
    lo, hi = res["rci_pp"]
    print(f"DECISION {res['decision']} look {res['look']} n {res['n']} "
          f"mean {res['mean_pp']:+.1f} pp rci [{lo:+.1f}, {hi:+.1f}] blocks {blocks}")
    if json_path:
        Path(json_path).write_text(json.dumps({**res, "blocks": blocks}, indent=1, default=str) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", help="release arm dir, specialist arm dir (one-block form)")
    ap.add_argument("--release", default=None, help="comma-separated release arm dirs (block form)")
    ap.add_argument("--specialist", default=None, help="comma-separated specialist arm dirs (block form)")
    ap.add_argument("--look", type=int, default=256)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--order-seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.arms:
        if len(a.arms) != 2 or a.release or a.specialist:
            print("ALERT forge_seqtest: give either two arm dirs or --release/--specialist lists")
            return 2
        rel = x4b_spec_card.games(Path(a.arms[0]))
        spe = x4b_spec_card.games(Path(a.arms[1]))
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
        return report(res, 1, a.json)

    if not a.release or not a.specialist:
        print("ALERT forge_seqtest: give either two arm dirs or --release/--specialist lists")
        return 2
    rel_dirs = [p for p in a.release.split(",") if p]
    spe_dirs = [p for p in a.specialist.split(",") if p]
    if len(rel_dirs) != len(spe_dirs) or not rel_dirs:
        print("ALERT forge_seqtest: --release and --specialist need the same non-zero number of dirs")
        return 2
    diffs: list[float] = []
    for k, (r, s) in enumerate(zip(rel_dirs, spe_dirs, strict=True), start=1):
        diffs += block_diffs(Path(r), Path(s), a.look, a.order_seed + k, f"block {k}")
    res = decide(diffs, a.look, a.K, margin=a.margin, alpha=a.alpha)
    return report(res, len(rel_dirs), a.json)


if __name__ == "__main__":
    sys.exit(main())
