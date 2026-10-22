"""Print one deck's card texts once each: `Name [type/stats] — full text`, Main then Extra.

Replaces many `txt.py` calls at the start of a hand-played game: the player reads both sheets once
instead of fetching each card text mid-game. `--brief` prints only name, type/stats and the first
300 chars of text per card (agents can still `txt.py` a card for full text).

    uv run python scripts/deck_sheet.py snake-eye [--brief]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk
from decklight.demos.game import _effect_text
from decklight.demos.names import CardNames
from decklight.engine import constants as C
from decklight.engine.resources import load_cards

_ATTR = {
    C.ATTRIBUTE_EARTH: "EARTH", C.ATTRIBUTE_WATER: "WATER", C.ATTRIBUTE_FIRE: "FIRE",
    C.ATTRIBUTE_WIND: "WIND", C.ATTRIBUTE_LIGHT: "LIGHT", C.ATTRIBUTE_DARK: "DARK",
    C.ATTRIBUTE_DIVINE: "DIVINE",
}  # fmt: skip
_RACE = {v: k.removeprefix("RACE_").title() for k, v in vars(C).items() if k.startswith("RACE_")}
_RACE.update({"Wingedbeast": "Winged Beast", "Beastwarrior": "Beast-Warrior",
              "Seaserpent": "Sea Serpent", "Creatorgod": "Creator God"})


def _stats(code: int, cards: dict) -> str:
    """Compact `[type/stats]` bracket for one card code."""
    d = cards.get(code)
    if d is None:
        return "[?]"
    t = d.type
    if t & C.TYPE_MONSTER:
        kind = "/".join(
            x
            for x, b in (
                ("Tuner", C.TYPE_TUNER), ("Fusion", C.TYPE_FUSION), ("Ritual", C.TYPE_RITUAL),
                ("Synchro", C.TYPE_SYNCHRO), ("Xyz", C.TYPE_XYZ), ("Link", C.TYPE_LINK),
                ("Pendulum", C.TYPE_PENDULUM), ("Spirit", C.TYPE_SPIRIT), ("Toon", C.TYPE_TOON),
                ("Union", C.TYPE_UNION), ("Gemini", C.TYPE_GEMINI), ("Flip", C.TYPE_FLIP),
            )
            if t & b
        )
        kind = kind or ("Normal" if t & C.TYPE_NORMAL else "Effect")
        if t & C.TYPE_XYZ:
            lvl = f"R{d.level}"
        elif t & C.TYPE_LINK:
            lvl = f"Link-{d.level}"
        else:
            lvl = f"L{d.level}"
        atk = "?" if d.attack < 0 else str(d.attack)
        dfn = "LINK" if t & C.TYPE_LINK else ("?" if d.defense < 0 else str(d.defense))
        race = _RACE.get(d.race, "")
        attr = _ATTR.get(d.attribute, "")
        return f"[{attr} {race}/{kind}/{lvl}/{atk}/{dfn}]"
    if t & C.TYPE_SPELL:
        prop = next(
            (x for x, b in (
                ("Quick-Play", C.TYPE_QUICKPLAY), ("Continuous", C.TYPE_CONTINUOUS),
                ("Equip", C.TYPE_EQUIP), ("Field", C.TYPE_FIELD), ("Ritual", C.TYPE_RITUAL),
            ) if t & b),
            "Normal",
        )
        return f"[Spell/{prop}]"
    prop = next(
        (x for x, b in (("Continuous", C.TYPE_CONTINUOUS), ("Counter", C.TYPE_COUNTER)) if t & b),
        "Normal",
    )
    return f"[Trap/{prop}]"


def sheet(deck_name: str, brief: bool = False) -> str:
    """The whole sheet as text (also used by tests). With `brief`, only the first 300 chars of text."""
    deck = load_ydk(ROOT / "decks" / "pool" / f"{deck_name}.ydk")
    names, cards = CardNames(), load_cards()
    lines = []
    for title, section in (("Main", deck.main), ("Extra", deck.extra)):
        lines.append(f"== {title} ({len(section)} cards) ==")
        seen: set[int] = set()
        for code in section:
            if code in seen:
                continue
            seen.add(code)
            nm = names.names.get(code, f"#{code}")
            text = _effect_text(code)
            if brief:
                text = text[:300]
            lines.append(f"{nm} {_stats(code, cards)} — {text}")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--brief"):
        print("usage: deck_sheet.py <deck> [--brief]", file=sys.stderr)
        sys.exit(2)
    print(sheet(sys.argv[1], brief=len(sys.argv) == 3))


if __name__ == "__main__":
    main()
