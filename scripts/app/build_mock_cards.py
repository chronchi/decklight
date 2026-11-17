"""Mock card data for the deck builder screenshots (web/mock/deckbuilder.json).

Takes the cards of a few pool decks from BabelCDB and writes, per card: code, name, kind, stat line, extra-deck flag,
type bits and effect text, plus two sample user decks built from those lists. The mock backend in
web/features/deckbuilder/mock.js answers cards_search / card_get / deck_* from this file, so the UI can be designed and
screenshotted without a server.

    uv run python scripts/app/build_mock_cards.py
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECKS = ["snake-eye", "ryzeal-mitsurugi-meta1", "tenpai-dragon"]
ATTR = {1: "EARTH", 2: "WATER", 4: "FIRE", 8: "WIND", 0x10: "LIGHT", 0x20: "DARK", 0x40: "DIVINE"}
RACES = ["Warrior", "Spellcaster", "Fairy", "Fiend", "Zombie", "Machine", "Aqua", "Pyro", "Rock", "Winged Beast",
         "Plant", "Insect", "Thunder", "Dragon", "Beast", "Beast-Warrior", "Dinosaur", "Fish", "Sea Serpent", "Reptile",
         "Psychic", "Divine-Beast", "Creator God", "Wyrm", "Cyberse", "Illusion"]
EXTRA = 0x40 | 0x2000 | 0x800000 | 0x4000000


def ydk(name: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"main": [], "extra": [], "side": []}
    sec = "main"
    for ln in (ROOT / "decks/pool" / f"{name}.ydk").read_text().splitlines():
        ln = ln.strip()
        if ln in ("#main", "#extra", "!side"):
            sec = ln[1:]
        elif re.fullmatch(r"\d+", ln):
            out[sec].append(int(ln))
    return out


def line(t: int, atk: int, df: int, lvl: int, race: int, attr: int) -> tuple[str, str]:
    if t & 1:
        r = next((RACES[i] for i in range(len(RACES)) if race >> i & 1), "")
        a = ATTR.get(attr, "")
        lv = lvl & 0xFF
        if t & 0x4000000:
            return "monster", f"Link-{lv} · {a} · {r} · {atk}"
        head = f"Rank {lv}" if t & 0x800000 else f"Level {lv}"
        return "monster", f"{head} · {a} · {r} · {atk}/{df}"
    sub = {0x80: "Ritual", 0x10000: "Quick-Play", 0x20000: "Continuous", 0x40000: "Equip", 0x80000: "Field",
           0x100000: "Counter"}
    s = next((v for k, v in sub.items() if t & k), "Normal")
    kind = "spell" if t & 2 else "trap"
    return kind, f"{s} {kind.capitalize()}" if s != "Normal" else f"Normal {kind.capitalize()}"


def main() -> None:
    con = sqlite3.connect(ROOT / "third_party/BabelCDB/cards.cdb")
    decks = {d: ydk(d) for d in DECKS}
    codes = sorted({c for d in decks.values() for s in d.values() for c in s})
    cards = []
    for c in codes:
        row = con.execute("SELECT d.type, d.atk, d.def, d.level, d.race, d.attribute, t.name, t.desc "
                          "FROM datas d JOIN texts t ON t.id = d.id WHERE d.id = ?", (c,)).fetchone()
        if not row:
            continue
        t, atk, df, lvl, race, attr, name, desc = row
        kind, ln = line(t, atk, df, lvl, race, attr)
        cards.append({"code": c, "name": name, "kind": kind, "line": ln, "extra": bool(t & EXTRA and t & 1),
                      "type": t, "text": desc.replace("\r\n", "\n")})
    if len(cards) < 50:
        print(f"ALERT build_mock_cards: only {len(cards)} cards", file=sys.stderr)
        sys.exit(2)
    user = [
        {"id": "my-snake-eye-3fa21c", "name": "My Snake-Eye", "updated": "2026-11-12T21:40:00",
         **decks["snake-eye"]},
        {"id": "ryzeal-test-8b01d2", "name": "Ryzeal test", "updated": "2026-11-10T18:05:00",
         "main": decks["ryzeal-mitsurugi-meta1"]["main"][:36], "extra": decks["ryzeal-mitsurugi-meta1"]["extra"],
         "side": []},
    ]
    out = ROOT / "web/mock/deckbuilder.json"
    out.write_text(json.dumps({"cards": cards, "decks": user}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {out}: {len(cards)} cards, {len(user)} decks, {out.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
