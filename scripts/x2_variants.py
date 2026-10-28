"""X2 (docs/plans/2026-10-27-experiments-and-rust.md): fresh tournament lists of archetypes P trained on.

Fetches the newest YGOPRODeck "Tournament Meta Decks" (own cache, data/cache/meta_decks_x2/), keeps lists whose
archetype slug matches a TRAIN pool deck but whose list is in no pool file, validates them like fetch_meta_decks
(rules + engine smoke), and writes:

  decks/variants/<slug>-v<k>.ydk + decks/variants/SOURCES.md   the fresh lists (variant V)
  decks/variants/x2-V.pairs / x2-T.pairs                        search_match --pairs files, same order and opponents:
                                                                V:O in one, T:O in the other (T = the archetype's
                                                                training list), so `--seed-per-pair` deals pair them

    taskset -c 0-3 uv run python scripts/x2_variants.py --pages 8 --per-archetype 3 --opponents 8

Exits 1 when fewer than --min-variants lists survive (the diagnostic would be too small to read).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_meta_decks as fm

from decklight.decks import load_ydk
from decklight.demos.names import CardNames
from decklight.engine.duel import Engine

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "decks" / "pool"
OUT = ROOT / "decks" / "variants"


def multiset(main, extra) -> Counter:
    return Counter(main) + Counter(extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--per-archetype", type=int, default=3)
    ap.add_argument("--opponents", type=int, default=8, help="fixed train pool decks every list meets")
    ap.add_argument("--min-variants", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2710)
    args = ap.parse_args()

    hold = set((POOL / "HOLDOUT").read_text().split()) | {"kewl-tune"}
    train = sorted(p.stem for p in POOL.glob("*.ydk") if p.stem not in hold)
    by_arch: dict[str, list[str]] = {}
    for s in train:
        by_arch.setdefault(fm._base_slug(s), []).append(s)
    pool_lists = {s: multiset(d.main, d.extra) for s in train for d in [load_ydk(POOL / f"{s}.ydk")]}
    pool_urls = set(re.findall(r"ygoprodeck\.com/deck/([\w-]+)", (POOL / "SOURCES.md").read_text()))

    fm.CACHE = ROOT / "data" / "cache" / "meta_decks_x2"  # fresh pages, never the pool's old cache
    fm.CACHE.mkdir(parents=True, exist_ok=True)
    entries, fetched = fm.fetch_pages(args.pages, 50, "Tournament Meta Decks")
    today = datetime.now(UTC).date()
    engine, names = Engine(), CardNames()
    rejected: Counter = Counter()
    cands: list[dict] = []
    seen: set = set()
    for e in entries:
        rec = fm.normalise(e, today)
        if rec is None or rec["deck_num"] in seen:
            rejected["parse/dup"] += 1
            continue
        seen.add(rec["deck_num"])
        if rec["slug"] not in by_arch:
            rejected["archetype not trained"] += 1
            continue
        if rec["url"].rsplit("/", 1)[-1] in pool_urls:
            rejected["in pool"] += 1
            continue
        ok, why = fm.validate_record(rec, engine, names)
        if not ok:
            rejected[why] += 1
            continue
        ms = multiset(rec["main"], rec["extra"])
        if any(ms == pool_lists[s] for s in by_arch[rec["slug"]]):
            rejected["same list as pool"] += 1
            continue
        cands.append(rec)

    picked = fm.select(cands, args.per_archetype)
    if len(picked) < args.min_variants:
        print(f"x2_variants: only {len(picked)} fresh lists (need {args.min_variants}); rejected {dict(rejected)}",
              file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.ydk"):
        old.unlink()
    rng = random.Random(args.seed)
    archs = sorted({r["slug"] for r in picked})
    others = [s for s in train if fm._base_slug(s) not in archs]
    opps = sorted(rng.sample(others, args.opponents))
    run_day = today.isoformat()
    vp, tp, src = [], [], ["# X2 fresh lists (scripts/x2_variants.py)", "",
                           f"Fetched {run_day}; {fetched} pages from the network; opponents: {', '.join(opps)}.", "",
                           "| variant | control (training list) | cards differing | source |", "|---|---|---|---|"]
    k_of: Counter = Counter()
    for rec in sorted(picked, key=lambda r: (r["slug"], r["url"])):
        k_of[rec["slug"]] += 1
        stem = f"{rec['slug']}-v{k_of[rec['slug']]}"
        fm.write_ydk(OUT / f"{stem}.ydk", rec, rec["main"], rec["extra"], run_day)
        ctrl = min(by_arch[rec["slug"]], key=lambda s: ("-meta" not in s, s))  # a tournament list when the pool has one
        diff = sum((multiset(rec["main"], rec["extra"]) - pool_lists[ctrl]).values())
        src.append(f"| {stem} | {ctrl} | {diff} | {rec['url']} ({rec['tournament']}, {rec['placement']}, "
                   f"{rec['date']}) |")
        for o in opps:
            vp.append(f"{stem}:{o}")
            tp.append(f"{ctrl}:{o}")
    (OUT / "SOURCES.md").write_text("\n".join(src) + "\n")
    (OUT / "x2-V.pairs").write_text("\n".join(vp) + "\n")
    (OUT / "x2-T.pairs").write_text("\n".join(tp) + "\n")
    print(f"x2_variants: {len(picked)} fresh lists of {len(archs)} archetypes, {len(vp)} pairs per side, "
          f"rejected {dict(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
