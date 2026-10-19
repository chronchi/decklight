"""Merge scored primer summary caches into one, for ``build_primers.py --from-summaries`` (NOTEBOOK 2026-10-19).

Drops the summaries whose deal seed is in ``--drop-seed`` (the v1 lobby primer games all used seed 3000, so each deck
met ~one opening hand) and refuses unscored summaries or an empty result.

Usage: uv run python scripts/primer_merge_summaries.py --out data/primers/summaries.json.gz \
           --drop-seed 3000 runs/primer-impact-20261018-0756/summaries.json.gz runs/<v2>/score-*/summaries.json.gz
Exit 2 on an unscored summary, a game already in an earlier cache, or 0 summaries out.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path


def main(argv=None) -> int:
    import ygo_zero.analysis.primer as P

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("caches", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drop-seed", type=int, action="append", default=[])
    args = ap.parse_args(argv)

    out, seen = [], set()
    for p in args.caches:
        summ = json.loads(gzip.decompress(p.read_bytes()))
        kept = [s for s in summ if s["seed"] not in args.drop_seed]
        if any(not P._is_scored(s) for s in kept):
            print(f"ALERT primer_merge_summaries: unscored summaries in {p}", flush=True)
            return 2
        # duplicates are checked across caches only: one cache may hold runs on the same schedule (hppo-p, hppo-p2)
        keys = {(s["seed"], tuple(s["decks"]), s.get("game_idx")) for s in kept}
        if keys & seen:
            print(f"ALERT primer_merge_summaries: {len(keys & seen)} games of {p} already merged", flush=True)
            return 2
        seen |= keys
        out.extend(kept)
        print(f"{p}: {len(summ)} summaries, kept {len(kept)}", flush=True)
    if not out:
        print("ALERT primer_merge_summaries: 0 summaries out", flush=True)
        return 2
    seeds = Counter(s["seed"] for s in out)
    print(
        f"out: {len(out)} summaries, {len(seeds)} distinct seeds, most common seed x{seeds.most_common(1)[0][1]}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(gzip.compress(json.dumps(out).encode()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
