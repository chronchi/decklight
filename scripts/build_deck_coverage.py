"""Which lobby decks the Forge trained on (user 2026-10-31: "we should somehow inform if that decklist has been fine
tuned or the similar decklist has been fine tuned, because different decks might have different powers").

Writes src/decklight/web/deck_coverage.json, read by the web server's deck list:
  {"version": 1, "release": "r1", "decks": {deck_id: {"level": "trained" | "related", "arch": ..., "overlap": 0.87,
   "release": "r0" | "r1", "gain_pp": ..., "gain_ci": [lo, hi], "vs": "P" | "r0"}}}

- trained: the deck's main + extra is identical (as a multiset) to a list a specialist trained on.
- related: at least RELATED share of its non-staple distinct cards appear in one archetype's training lists (staple =
  in >= 8% of lobby decks: hand traps, generic extra deck). Its own gain was never measured.
- other decks get no entry: the release only kept itself close to the earlier model on them.
Gains are the raw-policy release tables of docs/results.md (r0 - P, r1 - r0), one per archetype.

    uv run python scripts/build_deck_coverage.py            # writes the JSON, prints the table
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOBBY_DIRS = [ROOT / "decks" / "pool", ROOT / "decks" / "newcards"]  # = server.DECK_DIRS
OUT = ROOT / "src" / "decklight" / "web" / "deck_coverage.json"
# (folder glob of training lists, release, opponent the gain is measured against)
TRAIN_SETS = [("decks/x4b/train/*/*.ydk", "r0", "P"), ("decks/forge/f*/train/*/*.ydk", "r1", "r0")]
# docs/results.md "Release r0" (r0 - P) and "Release r1" (r1 - r0) tables, raw policy, 95% CI.
GAINS = {
    "branded-light-and-darkness-ritual": (6.6, 4.5, 8.7), "invoked": (17.6, 14.8, 20.4),
    "radiant-typhoon": (22.4, 19.4, 25.4), "sacred-beasts": (13.1, 10.6, 15.5),
    "atlantean-mermail": (19.1, 16.4, 21.9),
    "blitzclique": (7.9, 5.7, 10.2), "elfnote": (12.3, 9.6, 15.0), "fiendsmith-yummy": (33.7, 31.0, 36.5),
    "ignister-maliss": (19.5, 16.9, 22.2), "kewl-tune": (16.9, 14.1, 19.7), "lunalight": (11.9, 9.0, 14.7),
    "magistus-invoked-fairy-tail": (19.1, 16.4, 21.9), "mitsurugi": (11.4, 9.0, 13.8),
    "power-patron-artmage": (13.2, 10.8, 15.7), "sky-striker": (16.3, 13.6, 19.1), "trains": (13.8, 11.0, 16.5),
}
RELATED = 0.5
STAPLE_SHARE = 0.08


def read_ydk(p: Path) -> tuple[list[int], list[int]]:
    main: list[int] = []
    extra: list[int] = []
    sec = None
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("#main"):
            sec = main
        elif line.startswith("#extra"):
            sec = extra
        elif line.startswith("!side"):
            sec = None
        elif sec is not None and re.fullmatch(r"\d+", line):
            sec.append(int(line))
    return main, extra


def build() -> dict:
    lobby: dict[str, Path] = {}
    for d in LOBBY_DIRS:
        for p in sorted(d.glob("*.ydk")):
            lobby.setdefault(p.stem, p)  # first dir wins, as in the server
    if not lobby:
        raise SystemExit("ALERT build_deck_coverage: no lobby decks found")
    arch_cards: dict[str, set[int]] = collections.defaultdict(set)
    arch_release: dict[str, tuple[str, str]] = {}
    exact: dict[tuple, str] = {}
    for pattern, release, vs in TRAIN_SETS:
        for p in sorted(ROOT.glob(pattern)):
            a = p.parent.name
            main, extra = read_ydk(p)
            arch_cards[a] |= set(main) | set(extra)
            arch_release[a] = (release, vs)
            exact[(tuple(sorted(main)), tuple(sorted(extra)))] = a
    missing = set(arch_cards) ^ set(GAINS)
    if missing:
        raise SystemExit(f"ALERT build_deck_coverage: archetypes without a gain row or training lists: {missing}")
    decks = {k: read_ydk(p) for k, p in lobby.items()}
    df = collections.Counter(c for m, e in decks.values() for c in set(m) | set(e))
    staple = {c for c, n in df.items() if n >= STAPLE_SHARE * len(decks)}
    out: dict[str, dict] = {}
    for k, (m, e) in decks.items():
        a = exact.get((tuple(sorted(m)), tuple(sorted(e))))
        cards = (set(m) | set(e)) - staple
        share, best = max((len(cards & (arch_cards[x] - staple)) / max(1, len(cards)), x) for x in arch_cards)
        if a is None and share < RELATED:
            continue
        a = a or best
        release, vs = arch_release[a]
        g = GAINS[a]
        out[k] = {"level": "trained" if (tuple(sorted(m)), tuple(sorted(e))) in exact else "related", "arch": a,
                  "overlap": round(share, 2), "release": release, "vs": vs, "gain_pp": g[0], "gain_ci": [g[1], g[2]]}
    return {"version": 1, "release": "r1", "related_threshold": RELATED, "decks": dict(sorted(out.items()))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    res = build()
    n = collections.Counter(v["level"] for v in res["decks"].values())
    if not n["trained"]:
        print("ALERT build_deck_coverage: no lobby deck matches a training list exactly", flush=True)
        return 2
    for k, v in res["decks"].items():
        print(f"{v['level']:8s} {v['overlap']:.2f} {k:55s} {v['arch']} ({v['release']} {v['gain_pp']:+.1f})")
    print(f"trained {n['trained']}, related {n['related']}")
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
