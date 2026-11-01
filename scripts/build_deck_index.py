"""Deck browser index for the web lobby (user 2026-10-31: "we have no idea how to properly filter based on decklist or
any other stuff, in particular I have no idea about their decklists"; choices: "Full-screen sheet", "Forge tag only").

Writes src/decklight/web/deck_index.json, served at /deck_index.json:
  {"version": 1, "decks": {id: {"arch", "style", "about", "origin", "year", "mech": [...], "counts": [m, s, t],
   "sig": [3 codes], "main": [[code, n], ...], "extra": [[code, n], ...]}}, "names": {code: name}}

Sources, all in git: decks/pool/GENERATED.md (archetype, era, plan), decks/pool/SOURCES.md (site/event, list date),
decks/newcards/README.md (one line per deck), the card DB (types), primers.json.gz (key cards by win-chance impact).
No strength number: the primers' self-play win rate came from P and mixes deck power with play skill (user chose the
Forge tag only).

    uv run python scripts/build_deck_index.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_deck_coverage import LOBBY_DIRS, STAPLE_SHARE, read_ydk

OUT = ROOT / "src" / "decklight" / "web" / "deck_index.json"
MOCK_OUT = ROOT / "web" / "mock" / "deck-index.json"  # for scripts/web_screens.py: the mock lobby's ids
PRIMERS = ROOT / "src" / "decklight" / "web" / "primers.json.gz"
STYLES = ("combo", "control", "midrange", "otk", "stun", "burn", "turbo", "ritual", "stall", "beatdown")
# A summon type counts as the deck's own when it is more than a staple splash (measured on the 189 lobby decks).
MECH_MIN = {"Fusion": 4, "Synchro": 4, "Xyz": 4, "Link": 6, "Ritual": 2, "Pendulum": 4}


def parse_generated(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) == 7 and re.fullmatch(r"[a-z0-9-]+", cells[0]) and cells[0] != "deck":
            out[cells[0]] = {"arch": cells[1], "era": cells[2], "plan": cells[6]}
    return out


def parse_sources(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    text = path.read_text()
    # Meta fetch lines: `decks/pool/x.ydk` (...): Name — Event, Placement, YYYY-MM-DD — URL [split]
    for m in re.finditer(r"`decks/pool/([a-z0-9-]+)\.ydk` \([^)]*\): ([^—\n]+) — ([^\n]+?), (\d{4}-\d\d-\d\d) — ", text):
        out[m.group(1)] = {"name": m.group(2).strip(), "origin": f"{m.group(3).strip()}, {m.group(4)}",
                           "year": int(m.group(4)[:4])}
    # Hand-sourced sections: ## x → `decks/pool/x.ydk` ... Site: ... List date: YYYY-MM-DD
    for sec in re.split(r"\n(?=## )", text):
        m = re.match(r"## [^\n]*`decks/pool/([a-z0-9-]+)\.ydk`", sec)
        site = re.search(r"Site: ([^.(\n]+)", sec)
        date = re.search(r"List date: (\d{4}-\d\d-\d\d)", sec)
        if m and date:
            out[m.group(1)] = {"origin": f"{site.group(1).strip() if site else 'web guide'}, {date.group(1)}",
                               "year": int(date.group(1)[:4])}
    return out


def parse_newcards(path: Path) -> dict[str, dict]:
    out = {}
    for m in re.finditer(r"^- `([a-z0-9-]+)\.ydk` — ([^,]+), ([^.]+)\. (.+?)\. List from", path.read_text(), re.MULTILINE):
        out[m.group(1)] = {"arch": m.group(2), "style": m.group(3), "plan": m.group(4)}
    return out


def styles_in(text: str) -> list[str]:
    t = text.lower()
    return [s for s in STYLES if re.search(rf"\b{s}\b", t)]


def build() -> dict:
    from decklight.analysis.primer import hash_of
    from decklight.engine import constants as C
    from decklight.engine.resources import load_cards
    from decklight.web.view import DB

    cards, names = load_cards(), DB.names()
    gen = parse_generated(ROOT / "decks" / "pool" / "GENERATED.md")
    src = parse_sources(ROOT / "decks" / "pool" / "SOURCES.md")
    newc = parse_newcards(ROOT / "decks" / "newcards" / "README.md")
    primers = json.loads(gzip.decompress(PRIMERS.read_bytes()))["decks"]
    lobby: dict[str, Path] = {}
    for d in LOBBY_DIRS:
        for p in sorted(d.glob("*.ydk")):
            lobby.setdefault(p.stem, p)
    lists = {k: read_ydk(p) for k, p in lobby.items()}
    df = collections.Counter(c for m, e in lists.values() for c in set(m) | set(e))
    staple = {c for c, n in df.items() if n >= STAPLE_SHARE * len(lists)}
    out, used = {}, set()
    for k, (main, extra) in lists.items():
        g, s, n = gen.get(k, {}), src.get(k, {}), newc.get(k, {})
        arch_long = g.get("arch") or n.get("arch") or s.get("name") or k.replace("-", " ").title()
        arch = re.sub(r"\s*\(.*\)$", "", arch_long).strip()
        ty = lambda c: cards[c].type if c in cards else 0
        cnt = collections.Counter()
        for c in main:
            t = ty(c)
            cnt["m" if t & C.TYPE_MONSTER else "s" if t & C.TYPE_SPELL else "t" if t & C.TYPE_TRAP else "?"] += 1
            if t & C.TYPE_MONSTER and t & C.TYPE_RITUAL:
                cnt["Ritual"] += 1
            if t & C.TYPE_PENDULUM:
                cnt["Pendulum"] += 1
        for c in extra:
            t = ty(c)
            for name, bit in (("Fusion", C.TYPE_FUSION), ("Synchro", C.TYPE_SYNCHRO), ("Xyz", C.TYPE_XYZ),
                              ("Link", C.TYPE_LINK)):
                if t & bit:
                    cnt[name] += 1
        if cnt["?"]:
            raise SystemExit(f"ALERT build_deck_index: {k} has {cnt['?']} main-deck codes missing from the card DB")
        mech = [m for m, lo in MECH_MIN.items() if cnt[m] >= lo]
        pr = primers.get(hash_of({"main": main, "extra": extra}))
        sig = [c["code"] for c in (pr or {}).get("key_cards", [])[:3]]
        if len(sig) < 3:  # no primer: the most-copied distinctive monsters
            mons = collections.Counter(c for c in main if c not in staple and ty(c) & C.TYPE_MONSTER)
            sig += [c for c, _ in mons.most_common() if c not in sig][: 3 - len(sig)]
        era = g.get("era", "")
        year = s.get("year") or (int(m.group(0)) if (m := re.search(r"\d{4}", era)) else None)
        origin = s.get("origin") or (f"Assembled for decklight from {era} builds" if era else
                                     "Assembled for decklight (new-cards test deck)" if n else "Assembled for decklight")
        style = ", ".join(styles_in(n.get("style", "") or arch_long + " " + g.get("plan", ""))[:2])
        about = g.get("plan") or n.get("plan") or ""
        if "ygoprodeck meta" in about:  # meta lists' plan cell only repeats the event (shown as origin)
            about = ""
        out[k] = {"arch": arch, "style": style, "about": about, "origin": origin,
                  "year": year, "mech": mech, "counts": [cnt["m"], cnt["s"], cnt["t"]], "sig": sig,
                  "main": [[c, x] for c, x in collections.Counter(main).items()],
                  "extra": [[c, x] for c, x in collections.Counter(extra).items()]}
        used |= set(main) | set(extra)
    return {"version": 1, "decks": out, "names": {str(c): names.get(c, f"#{c}") for c in sorted(used)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    res = build()
    d = res["decks"]
    bad = [k for k, v in d.items() if len(v["sig"]) < 3]
    print(f"decks {len(d)}, names {len(res['names'])}, with a plan {sum(bool(v['about']) for v in d.values())}, "
          f"< 3 signature cards: {bad}")
    print("mechanics:", collections.Counter(m for v in d.values() for m in v["mech"]))
    print("styles:", collections.Counter(s for v in d.values() for s in v["style"].split(", ") if s))
    print("years:", sorted(collections.Counter(v["year"] for v in d.values() if v["year"]).items()))
    if len(d) < 100 or any(n.startswith("#") for n in res["names"].values()):
        print("ALERT build_deck_index: too few decks or unnamed cards", flush=True)
        return 2
    a.out.write_text(json.dumps(res, separators=(",", ":")) + "\n")
    print(f"wrote {a.out} ({a.out.stat().st_size // 1024} KiB)")
    write_mock(res)
    return 0


def write_mock(res: dict) -> None:
    """The mock lobby's decks (pool-00 'Mitsurugi', ...) mapped to the real index entry of the same name."""
    mock = json.loads((ROOT / "web" / "mock" / "01-lobby.json").read_text())["decks"]
    by_name = {k.replace("-", " ").lower(): k for k in res["decks"]}
    decks = {}
    for d in mock:
        real = by_name.get(d["name"].replace("-", " ").lower())
        if real:
            decks[d["id"]] = res["decks"][real]
    used = {str(c) for v in decks.values() for c, _n in v["main"] + v["extra"]}
    MOCK_OUT.write_text(json.dumps({"version": 1, "decks": decks,
                                    "names": {c: n for c, n in res["names"].items() if c in used}}) + "\n")
    print(f"wrote {MOCK_OUT} ({len(decks)} of {len(mock)} mock decks)")


if __name__ == "__main__":
    sys.exit(main())
