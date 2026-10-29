"""Forge F0 (docs/plans/2026-10-29-deck-forge.md): tournament lists + split + pairs for N archetypes.

A reusable version of scripts/x4b_lists.py + scripts/x4b_pairs.py for any list of archetypes.
One specialist per archetype trains on ``--n-train`` tournament lists and holds out the newest one.

Only tournament lists (YGOPRODeck "Tournament Meta Decks"): pool ``-meta`` lists, decks/variants,
fresh fetches. The X4b evaluation sets stay clean: a candidate whose URL deck id or card multiset
matches any list under decks/x4b/ or any --opponents list is dropped. Writes <out>/:
  train/<arch>/<stem>.ydk   the specialist's lists (--n-train per archetype)
  heldout/<stem>.ydk        one list per archetype, its newest tournament list, never trained on
  SOURCES.md, split.json    URL, event, placing, date of every list; the split
  pairs/arch-<arch>.pairs   each of the archetype's lists x its opponents (<S stem>:<opp stem>)

    taskset -c 0-3 uv run python scripts/forge_lists.py --archs elfnote sky-striker mitsurugi fiendsmith-yummy \\
        --out decks/forge/f0 [--pages 40] [--opponents decks/x4b/opponents.txt] [--n-train 3]

Exit 1 when an archetype has fewer than n-train + 1 lists.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_meta_decks as fm
import x4b_lists

from decklight.decks import load_ydk
from decklight.demos.names import CardNames
from decklight.engine.duel import Engine

ROOT = Path(__file__).resolve().parents[1]
POOL, VAR, X4B = ROOT / "decks/pool", ROOT / "decks/variants", ROOT / "decks/x4b"


def split_lists(cands: list[tuple[str, str]], n_train: int) -> tuple[list[str], str]:
    """Split (stem, date) candidates: newest held out, the next `n_train` newest train.

    Train comes back oldest-first (as in x4b_lists). Raises ValueError when short.
    """
    if len(cands) < n_train + 1:
        raise ValueError(f"need {n_train + 1} lists, have {len(cands)}")
    ordered = sorted(cands, key=lambda t: (t[1], t[0]))
    return [s for s, _ in ordered[:-1][-n_train:]], ordered[-1][0]


def opponents_for(arch: str, opps: list[str]) -> list[str]:
    """Opponents usable for `arch`: drop any stem containing the archetype string."""
    return [o for o in opps if arch not in o]


def multiset_key(ms: Counter) -> tuple[tuple[int, int], ...]:
    """Hashable form of a card multiset (for set membership)."""
    return tuple(sorted(Counter(ms).items()))


def is_excluded(deck_id: str, ms: Counter, excluded_ids: set[str], excluded_keys: set) -> bool:
    """True when a candidate matches an excluded list by URL deck id or card multiset."""
    return deck_id in excluded_ids or multiset_key(ms) in excluded_keys


def deck_id_of(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", required=True, help="archetype slugs (exact match)")
    ap.add_argument("--out", type=Path, default=ROOT / "decks/forge/f0")
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--opponents", type=Path, default=X4B / "opponents.txt")
    ap.add_argument("--n-train", type=int, default=3)
    args = ap.parse_args()
    archs: list[str] = args.archs
    OUT: Path = args.out

    opps = [l.strip() for l in args.opponents.read_text().splitlines() if l.strip()]
    opp_paths = [POOL / f"{s}.ydk" for s in opps]
    missing = [str(p) for p in opp_paths if not p.is_file()]
    if missing:
        print(f"forge_lists: opponent lists missing from {POOL}: {missing[:5]}", file=sys.stderr)
        return 1

    # Exclusion sets: every X4b list plus every --opponents list (URL deck id + card multiset).
    excluded_ids: set[str] = set()
    excluded_keys: set = set()
    for p in [*X4B.rglob("*.ydk"), *opp_paths]:
        m = x4b_lists.header_source(p)
        if m is not None:
            excluded_ids.add(deck_id_of(m["url"]))
        excluded_keys.add(multiset_key(x4b_lists.multiset(load_ydk(p))))

    # Pool / variant tournament candidates whose archetype equals the archetype exactly.
    meta = {p.stem: p for p in POOL.glob("*-meta*.ydk")}
    var = {p.stem: p for p in VAR.glob("*.ydk")}
    src = {s: m for s, p in {**meta, **var}.items() if (m := x4b_lists.header_source(p)) is not None}
    known = {s: x4b_lists.multiset(load_ydk(p)) for s, p in {**meta, **var}.items()}

    # Fresh fetches, validated exactly as x4b_lists.main does.
    fm.CACHE = ROOT / "data/cache/meta_decks_forge"
    fm.CACHE.mkdir(parents=True, exist_ok=True)
    entries, fetched = fm.fetch_pages(args.pages, 50, "Tournament Meta Decks")
    today = datetime.now(UTC).date()
    engine, names = Engine(), CardNames()
    rejected: Counter = Counter()
    fresh: list[dict] = []
    seen = set()
    for e in entries:
        rec = fm.normalise(e, today)
        if rec is None or rec["deck_num"] in seen:
            rejected["parse/dup"] += 1
            continue
        seen.add(rec["deck_num"])
        if rec["slug"] not in archs:
            rejected["other archetype"] += 1
            continue
        ok, why = fm.validate_record(rec, engine, names)
        if not ok:
            rejected[why] += 1
            continue
        fresh.append(rec)

    # Per archetype: filter, dedupe (same multiset keeps the newer), split.
    final: dict[str, dict[str, tuple[object, dict]]] = {}  # arch -> stem -> (Path | rec, source meta)
    counts: dict[str, Counter] = {}
    for arch in archs:
        cands: dict[str, tuple[object, dict, str, Counter, str]] = {}  # stem -> (item, meta, kind, ms, date)
        for s, p in {**meta, **var}.items():
            if re.sub(r"-(meta|v)\d+$", "", s) != arch or s not in src:
                continue
            ms = known[s]
            if is_excluded(deck_id_of(src[s]["url"]), ms, excluded_ids, excluded_keys):
                rejected[f"excluded {arch}"] += 1
                continue
            cands[s] = (p, src[s], "pool" if p.parent == POOL else "variant", ms, src[s]["date"])
        arch_fresh = sorted((r for r in fresh if r["slug"] == arch), key=lambda r: (r["date"], r["url"]))
        for k, r in enumerate(arch_fresh, 1):
            ms = Counter(r["main"]) + Counter(r["extra"])
            if is_excluded(deck_id_of(r["url"]), ms, excluded_ids, excluded_keys):
                rejected[f"excluded {arch}"] += 1
                continue
            stem = f"{arch}-f{k}"
            cands[stem] = (r, {"url": r["url"], "event": r["tournament"], "placing": r["placement"],
                               "date": r["date"]}, "fresh", ms, r["date"])
        # Same multiset inside one archetype: keep the newer (date, stem).
        best: dict = {}
        for stem, v in sorted(cands.items(), key=lambda kv: (kv[1][4], kv[0])):
            best[multiset_key(v[3])] = (stem, v)
        kept = sorted(best.values(), key=lambda t: (t[1][4], t[0]))
        final[arch] = {stem: (v[0], v[1]) for stem, v in kept}
        counts[arch] = Counter(v[2] for _, v in kept)

    short = {a: len(final[a]) for a in archs if len(final[a]) < args.n_train + 1}
    if short:
        detail = ", ".join(f"{a} has {n}" for a, n in short.items())
        print(f"forge_lists: fewer than {args.n_train + 1} tournament lists for {sorted(short)} "
              f"({detail}; excluded/dup/rejected {dict(rejected)})", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    run_day = today.isoformat()
    split: dict = {"archs": archs, "train": {}, "heldout": {}, "opponents": {}}
    rows = ["# Forge tournament lists (scripts/forge_lists.py)", "",
            (f"Built {run_day}; {fetched} pages fetched. Only tournament lists. "
             f"Held-out = the newest list of each archetype ({args.n_train} train)."), "",
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

    for arch in archs:
        cands_sorted = sorted(final[arch], key=lambda s: (final[arch][s][1]["date"], s))
        train, held = split_lists([(s, final[arch][s][1]["date"]) for s in cands_sorted], args.n_train)
        item, m = final[arch][held]
        put("heldout", held, item, m)
        split["heldout"][arch] = held
        split["train"][arch] = []
        for s in train:
            item, m = final[arch][s]
            put("train", s, item, m, arch)
            split["train"][arch].append(s)
        split["opponents"][arch] = opponents_for(arch, opps)
        pairs = OUT / "pairs"
        pairs.mkdir(parents=True, exist_ok=True)
        decks = [*train, held]
        (pairs / f"arch-{arch}.pairs").write_text(
            "".join(f"{d}:{o}\n" for d in decks for o in split["opponents"][arch]))
        c = counts[arch]
        print(f"{arch}: found pool={c.get('pool', 0)} variant={c.get('variant', 0)} fresh={c.get('fresh', 0)} "
              f"train={','.join(train)} heldout={held} opponents={len(split['opponents'][arch])}")

    rows += ["", "Opponents (--opponents minus stems containing the archetype): "
                 + "; ".join(f"{a}: {','.join(split['opponents'][a])}" for a in archs), ""]
    (OUT / "SOURCES.md").write_text("\n".join(rows) + "\n")
    (OUT / "split.json").write_text(json.dumps(split, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
