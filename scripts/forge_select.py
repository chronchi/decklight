"""Forge F2 archetype selection by popularity x novelty (docs/plans/2026-10-29-deck-forge.md).

Ranks tournament archetypes by popularity (recent lists) times novelty (share of the
archetype's signature cards not yet covered), picks greedily, and proposes a held-out
"reserve" set of unrelated archetypes for the release card. Laptop only.

    taskset -c 0-3 uv run python scripts/forge_select.py --n 11 --covered-archs branded-light-and-darkness-ritual invoked \\
        radiant-typhoon sacred-beasts atlantean-mermail elfnote sky-striker mitsurugi fiendsmith-yummy \\
        [--pages 40] [--days 60] [--min-lists 4] [--reserve 20] --out decks/forge/f2/selection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_lists

from decklight.forge import engines

ROOT = Path(__file__).resolve().parents[1]


def parse_day(s: str) -> date | None:
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=11)
    ap.add_argument("--covered-archs", nargs="*", default=[])
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--min-lists", type=int, default=4)
    ap.add_argument("--reserve", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "decks/forge/f2/selection.json")
    args = ap.parse_args()

    by_arch, _rejected, _fetched, _today = forge_lists.gather_candidates(args.pages, None, None)

    # Newest list in the data; popularity = lists dated within --days of it.
    dates = [d for cands in by_arch.values() for c in cands if (d := parse_day(c["date"])) is not None]
    if not dates:
        print("forge_select: no dated lists in the data", file=sys.stderr)
        return 1
    newest = max(dates)
    cutoff = (newest - timedelta(days=args.days)).isoformat()

    total = {a: len(cands) for a, cands in by_arch.items()}
    popularity = {a: sum(1 for c in cands if c["date"] >= cutoff) for a, cands in by_arch.items()}
    eligible = {a for a, n in total.items() if n >= args.min_lists}

    arch_lists: dict[str, list[Counter]] = {a: [c["ms"] for c in cands] for a, cands in by_arch.items() if cands}
    from decklight.engine.resources import load_cards

    cards = load_cards()
    named = {c for ms_list in arch_lists.values() for ms in ms_list for c in ms
             if c in cards and any(cards[c].setcodes)}
    staples = engines.staples(arch_lists, never=named)
    sigs: dict[str, set[int]] = {a: engines.signature([c["ms"] for c in cands], staples)
                                 for a, cands in by_arch.items() if cands}

    covered_archs: list[str] = list(args.covered_archs)
    covered_sigs: dict[str, set[int]] = {}
    for a in covered_archs:
        if not by_arch.get(a):
            print(f"forge_select: warning: covered archetype {a!r} has no lists in the data", file=sys.stderr)
            continue
        covered_sigs[a] = sigs.get(a, set())
    covered_union = set().union(*covered_sigs.values()) if covered_sigs else set()

    cands = {a: (popularity[a], sigs[a]) for a in sorted(eligible) if a not in set(covered_archs)}
    picked = engines.greedy_select(cands, covered_union, args.n)
    picked_archs = [p["arch"] for p in picked]
    picked_set = set(picked_archs)
    picked_sigs = {a: sigs[a] for a in picked_archs}

    # Reserve: neither picked nor covered, overlap < 0.2 with every picked and covered signature.
    ref_sigs: dict[str, set[int]] = {**covered_sigs, **picked_sigs}
    reserve_pool = []
    for a in sorted(by_arch):
        if a in picked_set or a in set(covered_archs) or not by_arch[a]:
            continue
        if all(engines.overlap(sigs.get(a, set()), s) < 0.2 for s in ref_sigs.values()):
            reserve_pool.append(a)
    reserve_pool.sort(key=lambda a: (popularity.get(a, 0), a))
    reserve = []
    for a in reserve_pool[: max(args.reserve, 0)]:
        stem, meta = max(((c["stem"], c["meta"]) for c in by_arch[a]),
                        key=lambda t: (t[1]["date"], t[0]))
        reserve.append({"arch": a, "stem": stem, "url": meta["url"], "date": meta["date"]})
    reserve_set = {r["arch"] for r in reserve}

    # Ranked table of the top 30 non-covered archetypes (score vs the original covered union).
    table: list[dict[str, Any]] = []
    for a in sorted(set(by_arch) - set(covered_archs)):
        if not by_arch[a]:
            continue
        sig = sigs.get(a, set())
        nov = engines.novelty(sig, covered_union)
        top2 = sorted(((c, engines.overlap(sig, s)) for c, s in covered_sigs.items()),
                      key=lambda t: (-t[1], t[0]))[:2]
        table.append({"arch": a, "lists": total[a], "popularity": popularity.get(a, 0),
                      "novelty": nov, "score": popularity.get(a, 0) * nov,
                      "top2": [{"arch": c, "overlap": o} for c, o in top2],
                      "picked": a in picked_set, "reserve": a in reserve_set})
    table.sort(key=lambda r: (-r["score"], r["arch"]))
    table = table[:30]

    print(f"newest {newest} cutoff {cutoff} staples {len(staples)} "
          f"archetypes {len(by_arch)} eligible {len(eligible)}")
    print(f"{'arch':44} {'lists':>5} {'pop':>4} {'nov':>5} {'score':>6}  "
          f"{'top2-overlaps':44} flags")
    for r in table:
        top = ",".join(f"{t['arch'][:20]}:{t['overlap']:.2f}" for t in r["top2"])
        flags = "+".join([s for s, on in (("picked", r["picked"]), ("reserve", r["reserve"])) if on])
        print(f"{r['arch']:44} {r['lists']:>5} {r['popularity']:>4} {r['novelty']:>5.2f} "
              f"{r['score']:>6.2f}  {top:44} {flags}")
    print("picked: " + ", ".join(picked_archs))
    print("reserve: " + ", ".join(r["arch"] for r in reserve))

    out = {"params": {"n": args.n, "covered_archs": covered_archs, "pages": args.pages,
                      "days": args.days, "min_lists": args.min_lists, "reserve": args.reserve,
                      "newest": newest.isoformat(), "cutoff": cutoff},
           "staples_n": len(staples), "covered": covered_archs, "picked": picked,
           "reserve": reserve, "table": table}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
