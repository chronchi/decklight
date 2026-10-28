"""X4b (docs/plans/2026-10-28-specialists-distill.md): tournament deck lists, split, eval sets.

Only tournament lists (YGOPRODeck "Tournament Meta Decks": pool `-meta<k>` lists, decks/variants, fresh fetches);
generated lists (pool files without `-meta`, decks/newcards) are never used. Writes decks/x4b/:
  train/<arch>/<stem>.ydk   the specialist's lists (3 per archetype)
  heldout/<stem>.ydk        one list per archetype, newest tournament list, never trained on
  related/<stem>.ydk        other archetypes sharing an engine with a fine-tuned one (RELATED tokens)
  unrelated/<stem>.ydk      everything else: fresh lists P never saw + the pool HOLDOUT meta lists
  opponents.txt             8 fixed pool train meta lists (unrelated), the O side of every card
  SOURCES.md, split.json    URL, event, placing, date of every list; the split

    taskset -c 0-3 uv run python scripts/x4b_lists.py [--pages 16] [--unrelated 14]
Exit 1 when an archetype has fewer than 4 lists or the unrelated fresh set has fewer than 10.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
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
POOL, VAR, OUT = ROOT / "decks/pool", ROOT / "decks/variants", ROOT / "decks/x4b"
FT = ["branded-light-and-darkness-ritual", "invoked", "radiant-typhoon", "sacred-beasts", "atlantean-mermail"]
# Shared engines with a fine-tuned archetype (user plan: "related" vs "unrelated" decks).
RELATED = ("branded", "invoked", "radiant", "typhoon", "sacred-beasts", "mermail", "atlantean", "light-and-darkness")


def multiset(d) -> Counter:
    return Counter(d.main) + Counter(d.extra)


def is_related(slug: str) -> bool:
    return slug not in FT and any(t in slug for t in RELATED)


def header_source(path: Path) -> dict | None:
    """{url, event, placing, date} from a fetch_meta_decks .ydk header (#source / #tournament ... | placement ... |
    date ...), None for a list without one (not a tournament list)."""
    head = path.read_text(errors="replace").splitlines()[:6]
    url = next((h.split(" ", 1)[1].strip() for h in head if h.startswith("#source ")), None)
    t = next((h for h in head if h.startswith("#tournament ")), None)
    m = re.match(r"#tournament (.*) \| placement (.*) \| date (\d{4}-\d\d-\d\d)", t or "")
    if not url or "ygoprodeck.com/deck/" not in url or not m:
        return None
    return {"url": url, "event": m.group(1).strip(), "placing": m.group(2).strip(), "date": m.group(3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=16)
    ap.add_argument("--unrelated", type=int, default=14, help="fresh unrelated lists (one per archetype)")
    ap.add_argument("--seed", type=int, default=2810)
    args = ap.parse_args()

    hold = set((POOL / "HOLDOUT").read_text().split())
    meta = {p.stem: p for p in POOL.glob("*-meta*.ydk")}
    var = {p.stem: p for p in VAR.glob("*.ydk")}
    src = {s: m for s, p in {**meta, **var}.items() if (m := header_source(p)) is not None}
    missing = sorted(s for s in [*meta, *var] if s not in src)
    if missing:
        print(f"x4b_lists: no tournament source for {missing[:5]}", file=sys.stderr)
    known = {s: multiset(load_ydk(p)) for s, p in {**meta, **var}.items()}
    used_urls = {v["url"].rsplit("/", 1)[-1] for v in src.values()}

    fm.CACHE = ROOT / "data/cache/meta_decks_x4b"
    fm.CACHE.mkdir(parents=True, exist_ok=True)
    entries, fetched = fm.fetch_pages(args.pages, 50, "Tournament Meta Decks")
    today = datetime.now(UTC).date()
    engine, names = Engine(), CardNames()
    rejected: Counter = Counter()
    fresh, seen = [], set()
    for e in entries:
        rec = fm.normalise(e, today)
        if rec is None or rec["deck_num"] in seen:
            rejected["parse/dup"] += 1
            continue
        seen.add(rec["deck_num"])
        if rec["url"].rsplit("/", 1)[-1] in used_urls:
            rejected["already have"] += 1
            continue
        ok, why = fm.validate_record(rec, engine, names)
        if not ok:
            rejected[why] += 1
            continue
        if any(Counter(rec["main"]) + Counter(rec["extra"]) == m for m in known.values()):
            rejected["same list"] += 1
            continue
        fresh.append(rec)

    # Fine-tuned archetypes: every tournament list we have, held-out = newest (fresh fetches fill a short archetype).
    lists: dict[str, list[tuple[str, Path | dict, dict]]] = {a: [] for a in FT}
    for s, p in {**meta, **var}.items():
        a = re.sub(r"-(meta|v)\d+$", "", s)
        if a in lists and s in src:
            lists[a].append((s, p, src[s]))
    for a in FT:
        extra = fm.select([r for r in fresh if r["slug"] == a], max(0, 4 - len(lists[a])))
        for k, r in enumerate(extra, 1):
            lists[a].append((f"{a}-x{k}", r, {"url": r["url"], "event": r["tournament"], "placing": r["placement"],
                                              "date": r["date"]}))
    short = [a for a in FT if len(lists[a]) < 4]
    if short:
        print(f"x4b_lists: fewer than 4 tournament lists for {short}", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    run_day = today.isoformat()
    split: dict = {"train": {}, "heldout": {}, "related": [], "unrelated_fresh": [], "unrelated_holdout": []}
    rows = ["# X4b tournament lists (scripts/x4b_lists.py)", "", f"Built {run_day}; {fetched} pages fetched. Only "
            "tournament lists. Held-out = the newest list of each fine-tuned archetype.", "",
            "| set | list | source |", "|---|---|---|"]

    def put(kind: str, stem: str, item, meta_: dict, sub: str = "") -> None:
        d = OUT / kind / sub if sub else OUT / kind
        d.mkdir(parents=True, exist_ok=True)
        if isinstance(item, Path):
            shutil.copy(item, d / f"{stem}.ydk")
        else:
            fm.write_ydk(d / f"{stem}.ydk", item, item["main"], item["extra"], run_day)
        rows.append(f"| {kind}{'/' + sub if sub else ''} | {stem} | {meta_['url']} ({meta_['event']}, "
                    f"{meta_['placing']}, {meta_['date']}) |")

    for a in FT:
        ordered = sorted(lists[a], key=lambda t: (t[2]["date"], t[0]))
        held, train = ordered[-1], ordered[:-1][-3:]  # newest held out; the 3 next newest train
        put("heldout", held[0], held[1], held[2])
        split["heldout"][a] = held[0]
        split["train"][a] = []
        for s, item, m in train:
            put("train", s, item, m, a)
            split["train"][a].append(s)

    # Related: every other tournament list sharing an engine (pool train/holdout meta, variants, fresh).
    for s, p in sorted({**meta, **var}.items()):
        a = re.sub(r"-(meta|v)\d+$", "", s)
        if is_related(a) and s in src:
            put("related", s, p, src[s])
            split["related"].append(s)
    # Unrelated: fresh lists P never saw (variants of other archetypes + new fetches, one newest per archetype),
    # then the unrelated pool HOLDOUT meta lists.
    cand = {}
    for s, p in sorted(var.items()):
        a = re.sub(r"-v\d+$", "", s)
        if a not in FT and not is_related(a) and s in src:
            cand.setdefault(a, []).append((s, p, src[s]))
    for r in fm.select([r for r in fresh if r["slug"] not in FT and not is_related(r["slug"])], 1):
        cand.setdefault(r["slug"], []).append((f"{r['slug']}-x1", r, {"url": r["url"], "event": r["tournament"],
                                                                       "placing": r["placement"], "date": r["date"]}))
    newest = sorted((max(v, key=lambda t: t[2]["date"]) for v in cand.values()), key=lambda t: t[2]["date"],
                    reverse=True)
    picks, per_engine = [], Counter()  # at most 2 lists per lead engine (first slug word): one new engine (e.g.
    for t in newest:                    # Ars Magna hybrids) must not dominate the transfer number
        lead = re.sub(r"-(v|x)\d+$", "", t[0]).split("-")[0]
        if per_engine[lead] < 2 and len(picks) < args.unrelated:
            per_engine[lead] += 1
            picks.append(t)
    for s, item, m in sorted(picks):
        put("unrelated", s, item, m)
        split["unrelated_fresh"].append(s)
    for s in sorted(hold):
        a = re.sub(r"-meta\d+$", "", s)
        if s in meta and s in src and not is_related(a) and a not in FT:
            put("unrelated", s, meta[s], src[s])
            split["unrelated_holdout"].append(s)
    if len(split["unrelated_fresh"]) < 10:
        print(f"x4b_lists: only {len(split['unrelated_fresh'])} fresh unrelated lists", file=sys.stderr)
        return 1

    # Opponents: 8 fixed pool *train* meta lists of unrelated archetypes not used anywhere above.
    taken = {re.sub(r"-(meta|v|x)\d+$", "", s) for v in split.values() for s in (v if isinstance(v, list) else
             [x for y in v.values() for x in (y if isinstance(y, list) else [y])])}
    opp_pool = sorted(s for s in meta if s not in hold and s in src and not is_related(re.sub(r"-meta\d+$", "", s))
                      and re.sub(r"-meta\d+$", "", s) not in taken | set(FT))
    split["opponents"] = sorted(random.Random(args.seed).sample(opp_pool, 8))
    (OUT / "opponents.txt").write_text("\n".join(split["opponents"]) + "\n")
    rows += ["", "Opponents (pool train meta lists, the O side of every card): " + ", ".join(split["opponents"]), ""]
    (OUT / "SOURCES.md").write_text("\n".join(rows) + "\n")
    (OUT / "split.json").write_text(json.dumps(split, indent=1) + "\n")
    print(f"x4b_lists: train {sum(len(v) for v in split['train'].values())}, heldout {len(split['heldout'])}, "
          f"related {len(split['related'])}, unrelated fresh {len(split['unrelated_fresh'])} + holdout "
          f"{len(split['unrelated_holdout'])}; rejected {dict(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
