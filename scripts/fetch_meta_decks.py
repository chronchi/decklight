"""Fetch real tournament-topping decklists (YGOPRODeck Tournament Meta Decks) into the pool.

    nice -n 19 taskset -c 0-3 uv run python scripts/fetch_meta_decks.py --pages 10 --per-page 50

Fetches pages politely (cached under data/cache/meta_decks/), validates each list with the
same rules as build_decks (official script + engine DB, alias-grouped <=3, main 40-60,
extra <=15, engine smoke), keeps the newest --per-archetype valid lists per archetype slug
(preferring better placements), and writes decks/pool/<slug>-meta<k>.ydk plus GENERATED.md /
SOURCES.md / HOLDOUT rows. Real TCG lists (no banlist check, like build_decks).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_decks import EXTRA_T, smoke, validate_codes

from decklight.decks import Deck
from decklight.demos.names import CardNames
from decklight.engine.duel import Engine

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "decks" / "pool"
CACHE = ROOT / "data" / "cache" / "meta_decks"
API = "https://ygoprodeck.com/api/decks/getDecks.php"
UA = "Mozilla/5.0 (ygo-zero research)"
GAP = 1.5  # min seconds between HTTP requests

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        1,
    )
}


def slugify(name: str) -> str:
    """deck_name -> pool slug: lowercase, non-alphanumerics to '-', collapsed."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-+", "-", s)


def parse_passcodes(raw: object) -> list[int] | None:
    """JSON-encoded list of passcode strings -> ints; None when any entry is invalid."""
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(items, list):
        return None
    out = []
    for it in items:
        s = str(it).strip()
        if not s.isdigit() or int(s) <= 0:
            return None
        out.append(int(s))
    return out


def placement_rank(placement: str) -> int:
    """Tournament placement -> sort rank (lower is better); unknown -> large."""
    p = placement.lower()
    if re.search(r"\b1st\b|\bwinner\b|\bchampion\b", p):
        return 1
    if re.search(r"\brunner-?up\b|\b2nd\b", p):
        return 2
    m = re.search(r"\b([34])\s*(?:st|nd|rd|th)\b", p)
    if m:
        return int(m.group(1))
    m = re.search(r"\btop\s*(\d+)", p)
    if m:
        return int(m.group(1))
    return 9999


def parse_desc(desc_html: str) -> tuple[str, str, str]:
    """deck_description HTML -> (tournament, placement, iso-date); "" when absent."""
    text = re.sub(r"<[^>]+>", "\n", html.unescape(desc_html or ""))
    tournament, placement, iso = "", "", ""
    m = re.search(r"Tournament:\s*(.+?)\s*[–—-]\s*([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})", text)
    if m:
        tournament = re.sub(r"\s+", " ", m.group(1)).strip()
        iso = parse_day(m.group(2))
    else:
        m2 = re.search(r"Tournament:\s*([^\n]+)", text)
        if m2:
            tournament = re.sub(r"\s+", " ", m2.group(1)).strip()
    m3 = re.search(r"Placement:\s*([^\n]+)", text)
    if m3:
        placement = re.sub(r"\s+", " ", m3.group(1)).strip()
    return tournament, placement, iso


def parse_day(s: str) -> str:
    """'September 27th 2026' -> '2026-09-27'; '' when unparseable."""
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})", s.strip())
    if not m or m.group(1).lower()[:3] not in {k[:3] for k in MONTHS}:
        return ""
    mon = MONTHS.get(m.group(1).lower(), 0) or next(
        v for k, v in MONTHS.items() if k.startswith(m.group(1).lower())
    )
    try:
        return date(int(m.group(3)), mon, int(m.group(2))).isoformat()
    except ValueError:
        return ""


def relative_day(s: str, today: date) -> str:
    """'6 days ago' -> iso date; '' when unparseable (fallback only)."""
    s = (s or "").lower().strip()
    m = re.match(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", s)
    if not m:
        return ""
    n, unit = int(m.group(1)), m.group(2)
    days = {"minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30, "year": 365}[unit]
    return (today - timedelta(days=n * days)).isoformat()


def normalise(entry: dict, today: date) -> dict | None:
    """Raw API deck -> normalised record; None when unparseable."""
    name = str(entry.get("deck_name") or "").strip()
    slug = slugify(name)
    if not name or not slug:
        return None
    main = parse_passcodes(entry.get("main_deck"))
    extra = parse_passcodes(entry.get("extra_deck"))
    side = parse_passcodes(entry.get("side_deck") or "[]")
    if main is None or extra is None or side is None:
        return None
    tournament, placement, iso = parse_desc(str(entry.get("deck_description") or ""))
    tournament = str(entry.get("tournamentName") or tournament or "").strip()
    placement = str(entry.get("tournamentPlacement") or placement or "").strip()
    if not iso:
        iso = relative_day(str(entry.get("submit_date") or ""), today)
    pretty = str(entry.get("pretty_url") or "").strip()
    deck_num = entry.get("deckNum")
    url = (
        f"https://ygoprodeck.com/deck/{pretty}"
        if pretty
        else (f"https://ygoprodeck.com/deck/-{deck_num}" if deck_num else "")
    )
    return {
        "deck_num": deck_num,
        "name": name,
        "slug": slug,
        "main": main,
        "extra": extra,
        "side": side,
        "tournament": tournament,
        "placement": placement,
        "date": iso,
        "url": url,
    }


def resplit(main: list[int], extra: list[int], engine: Engine) -> tuple[list[int], list[int]]:
    """Re-split main/extra by card type (extra-deck monsters -> extra), like build_decks."""
    new_main, new_extra = [], []
    for code in (*main, *extra):
        cd = engine.cards.get(code)
        if cd is not None and cd.type & EXTRA_T:
            new_extra.append(code)
        else:
            new_main.append(code)
    return new_main, new_extra


def check_side(side: list[int], engine: Engine, names: CardNames) -> str | None:
    """Validate the side deck; return an error string or None."""
    if len(side) > 15:
        return f"side deck has {len(side)} cards (max 15)"
    per_card: Counter = Counter()
    for code in side:
        if engine.cards.get(code) is None:
            return f"side: {code} missing from the engine card DB"
        per_card[names.alias.get(code, 0) or code] += 1
    for canon, n in per_card.items():
        if n > 3:
            return f"side: {names.names.get(canon, canon)}: {n} copies (max 3)"
    return None


def validate_record(rec: dict, engine: Engine, names: CardNames) -> tuple[bool, str]:
    """Resplit + validate + smoke one record; return (ok, reject_reason)."""
    main, extra = resplit(rec["main"], rec["extra"], engine)
    problems = validate_codes(main, extra, engine, names, min_extra=0)
    side_err = check_side(rec["side"], engine, names)
    if side_err:
        problems = [*problems, side_err]
    if problems:
        return False, reason_of(problems)
    deck = Deck(tuple(main), tuple(extra), tuple(rec["side"]), rec["slug"])
    if smoke(engine, deck):
        return False, "smoke_fail"
    rec["main"], rec["extra"] = main, extra
    return True, ""


def fetch_pages(pages: int, per_page: int, fmt: str) -> tuple[list[dict], int]:
    """Fetch (or reuse cached) raw pages; return (entries, n_fetched_from_network).

    The API returns a fixed page size regardless of `limit`, so offsets step by the
    number of decks actually received; cache files are keyed by that true offset.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    fetched = 0
    last_req = 0.0
    offset = 0
    page_size = 0
    for _ in range(pages):
        cache_file = CACHE / f"page-{offset}.json"
        if cache_file.exists():
            raw = cache_file.read_text()
        else:
            wait = GAP - (time.time() - last_req)
            if wait > 0:
                time.sleep(wait)
            qs = urllib.parse.urlencode({"format": fmt, "limit": per_page, "offset": offset})
            req = urllib.request.Request(f"{API}?{qs}", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8", "replace")
            last_req = time.time()
            cache_file.write_text(raw)
            fetched += 1
        try:
            batch = json.loads(raw)
        except json.JSONDecodeError:
            batch = []
        if not isinstance(batch, list) or not batch:
            break
        entries.extend(batch)
        offset += len(batch)
        if not page_size:
            page_size = len(batch)  # the API fixes its own page size, ignoring `limit`
        if len(batch) < page_size:
            break
    return entries, fetched


def select(cands: list[dict], per_archetype: int) -> list[dict]:
    """Newest `per_archetype` valid lists per slug, preferring better placements."""
    by_slug: dict[str, list[dict]] = {}
    for c in cands:
        by_slug.setdefault(c["slug"], []).append(c)
    out = []
    # newest first, then stable by placement rank (better placements win over novelty)
    for slug in sorted(by_slug):
        newest_first = sorted(by_slug[slug], key=lambda c: (c["date"], c["deck_num"] or 0), reverse=True)
        ranked = sorted(newest_first, key=lambda c: placement_rank(c["placement"]))
        out.extend(ranked[: max(per_archetype, 0)])
    return out


def _base_slug(stem: str) -> str:
    return re.sub(r"-meta\d+$", "", stem)


def assign_stems(
    picked: list[dict], pool_dir: Path, existing: set[str], holdout_set: set[str], frac: float
) -> tuple[list[tuple[dict, str, str]], int]:
    """Assign <slug>-meta<k>.ydk stems (+ train/holdout split), skipping existing files."""
    rows: list[tuple[dict, str, str]] = []
    skipped = 0
    for rec in picked:
        hold = is_holdout(rec["slug"], existing, holdout_set, frac)
        split = "holdout" if hold else "train"
        k = 1
        stem = f"{rec['slug']}-meta{k}"
        while (pool_dir / f"{stem}.ydk").exists():
            k += 1
            skipped += 1
            stem = f"{rec['slug']}-meta{k}"
        while stem in existing:  # taken by an earlier pick of this run (not a file skip)
            k += 1
            stem = f"{rec['slug']}-meta{k}"
        rows.append((rec, stem, split))
        existing.add(stem)  # later picks of the same slug take the next k
        if hold:
            holdout_set.add(stem)  # later picks of this archetype stay held out
    return rows, skipped


def is_holdout(slug: str, existing: set[str], holdout: set[str], frac: float) -> bool:
    """Per-archetype holdout: consistent with existing pool files, else md5(slug) < frac."""
    if slug in holdout or _base_slug(slug) in holdout:
        return True
    related = [
        s
        for s in existing
        if (b := _base_slug(s)) == slug or b.startswith(slug + "-") or slug.startswith(b + "-")
    ]
    if related:
        return any(r in holdout or _base_slug(r) in holdout for r in related)
    if frac <= 0:
        return False
    if frac >= 1:
        return True
    h = int(hashlib.md5(slug.encode()).hexdigest(), 16) / 2**128
    return h < frac


def write_ydk(path: Path, rec: dict, main: list[int], extra: list[int], run_day: str) -> None:
    lines = [
        f"#created by ygo-zero fetch_meta_decks (ygoprodeck meta, {run_day}): {rec['name']}",
        f"#source {rec['url']}",
        f"#tournament {rec['tournament']} | placement {rec['placement']} | date {rec['date']}",
        "#main",
        *[str(c) for c in main],
        "#extra",
        *[str(c) for c in extra],
        "!side",
        *[str(c) for c in rec["side"]],
        "",
    ]
    path.write_text("\n".join(lines))


def reason_of(problems: list[str]) -> str:
    for p in problems:
        if "missing from the engine card DB" in p:
            return "unknown_card"
    for p in problems:
        if "no engine script" in p:
            return "no_script"
    for p in problems:
        if "copies (max 3)" in p:
            return "copy_limit"
    return "bad_size"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=10)
    ap.add_argument("--per-page", type=int, default=50)
    ap.add_argument("--per-archetype", type=int, default=2)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--format", default="Tournament Meta Decks")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    today = datetime.now(UTC).date()
    run_day = today.isoformat()

    entries, fetched = fetch_pages(args.pages, args.per_page, args.format)
    seen_nums: set = set()
    rejected: Counter = Counter()
    cands: list[dict] = []
    for e in entries:
        num = e.get("deckNum")
        if num in seen_nums:
            rejected["duplicate"] += 1
            continue
        seen_nums.add(num)
        rec = normalise(e, today)
        if rec is None:
            rejected["parse"] += 1
            continue
        cands.append(rec)

    engine, names = Engine(), CardNames()
    valid: list[dict] = []
    for rec in cands:
        ok, reason = validate_record(rec, engine, names)
        if not ok:
            rejected[reason] += 1
            continue
        valid.append(rec)

    picked = select(valid, args.per_archetype)
    existing = {p.stem for p in POOL.glob("*.ydk")}
    holdout_set = set()
    holdout_file = POOL / "HOLDOUT"
    if holdout_file.exists():
        holdout_set = set(holdout_file.read_text().split())
    gen_file = POOL / "GENERATED.md"
    src_file = POOL / "SOURCES.md"

    written_train = written_hold = 0
    rows, skipped = assign_stems(picked, POOL, existing, holdout_set, args.holdout_frac)

    if not args.dry_run:
        for rec, stem, split in rows:
            write_ydk(POOL / f"{stem}.ydk", rec, rec["main"], rec["extra"], run_day)
            if split == "holdout":
                with holdout_file.open("a") as f:
                    f.write(stem + "\n")
            plan = f"{rec['tournament']}; {rec['placement']} (ygoprodeck meta)".replace("|", "/")
            arch = rec["name"].replace("|", "/")
            with gen_file.open("a") as f:
                f.write(
                    f"| {stem} | {arch} | {rec['date']} | {split} "
                    f"| {len(rec['main'])} | {len(rec['extra'])} | {plan[:160]} |\n"
                )
        for rec, stem, split in rows:
            if split == "holdout":
                written_hold += 1
            else:
                written_train += 1
        with src_file.open("a") as f:
            f.write(
                f"\n## ygoprodeck meta fetch — {run_day} ({len(rows)} decks)\n\n"
                f"- Endpoint: {API}?format={urllib.parse.quote(args.format)} "
                f"(pages {args.pages} x {args.per_page}). Raw cache: `data/cache/meta_decks/`.\n"
                f"- Selection: newest <={args.per_archetype} valid lists per archetype slug, "
                "preferring better placements. Validation: official script + engine card DB, "
                "alias-grouped <=3, main 40-60, extra <=15, side <=15, engine smoke (2 self-play games).\n"
                "- Decks:\n"
            )
            for rec, stem, split in rows:
                f.write(
                    f"  - `decks/pool/{stem}.ydk` "
                    f"({len(rec['main'])} + {len(rec['extra'])} + {len(rec['side'])} side): "
                    f"{rec['name']} — {rec['tournament']}, {rec['placement']}, {rec['date']} — "
                    f"{rec['url']} [{split}]\n"
                )

    print(f"pages fetched from network: {fetched} (cache: {CACHE})", flush=True)
    print(
        f"decks seen: {len(entries)} unique: {len(seen_nums)} "
        f"valid: {len(valid)} written: train {written_train} holdout {written_hold} "
        f"existing-file skips: {skipped}",
        flush=True,
    )
    print(f"rejected by reason: {dict(rejected)}", flush=True)
    if args.dry_run:
        print(
            f"(dry run: would write {len(rows)} decks: "
            + ", ".join(f"{s} [{sp}]" for _, s, sp in rows)
            + ")",
            flush=True,
        )


if __name__ == "__main__":
    main()
