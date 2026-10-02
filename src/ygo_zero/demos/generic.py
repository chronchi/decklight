"""Generic card descriptions in guides ("a Branded Spell/Trap", "Bystial", "a FIRE monster") → the cards of a
given deck that fit. Deterministic: archetype names come from CardScripts' setcode constants, type/attribute words
from a fixed table. A description that yields no constraint (e.g. "lock monster") resolves to nothing.
"""

import re
from collections.abc import Iterable, Mapping
from functools import cache

from ygo_zero.engine import constants as C
from ygo_zero.engine.resources import SCRIPTS_DIR, CardData

_TYPE_WORDS = [  # (pattern, required type bits); longest first
    (r"spells? or traps?|spell/traps?|s/t", C.TYPE_SPELL | C.TYPE_TRAP),
    (r"spells?", C.TYPE_SPELL),
    (r"traps?", C.TYPE_TRAP),
    (r"monsters?", C.TYPE_MONSTER),
    (r"tuners?", C.TYPE_TUNER),
    (r"link", C.TYPE_LINK),
    (r"fusions?", C.TYPE_FUSION),
    (r"synchros?", C.TYPE_SYNCHRO),
    (r"xyz", C.TYPE_XYZ),
    (r"continuous", C.TYPE_CONTINUOUS),
    (r"quick-?play", C.TYPE_QUICKPLAY),
    (r"field", C.TYPE_FIELD),
]
_ATTRS = {
    "light": C.ATTRIBUTE_LIGHT, "dark": C.ATTRIBUTE_DARK, "fire": C.ATTRIBUTE_FIRE, "water": C.ATTRIBUTE_WATER,
    "earth": C.ATTRIBUTE_EARTH, "wind": C.ATTRIBUTE_WIND,
}  # fmt: skip
# Type words that combine with OR (an "S/T" is a spell or a trap); all others must all hold.
_ANY_OF = C.TYPE_SPELL | C.TYPE_TRAP


@cache
def archetypes() -> dict[str, int]:
    """Normalized archetype name ("sky striker") → setcode, from archetype_setcode_constants.lua."""
    out: dict[str, int] = {}
    text = (SCRIPTS_DIR / "archetype_setcode_constants.lua").read_text(errors="replace")
    for m in re.finditer(r"^SET_([A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+|\d+)", text, re.MULTILINE):
        out[m.group(1).lower().replace("_", " ")] = int(m.group(2), 0)
    return out


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9/ -]+", " ", name.lower().replace('"', " ")).replace("-", " ")


def _in_set(cd: CardData, set_code: int) -> bool:
    base, sub = set_code & 0xFFF, set_code & 0xF000
    return any((sc & 0xFFF) == base and (sc & 0xF000 & sub) == sub for sc in cd.setcodes)


def parse(name: str) -> tuple[int | None, int, int, int]:
    """(setcode or None, all-of type bits, any-of type bits, attribute bits) described by `name`."""
    text = f" {_norm(name)} "
    setcode = None
    for arch in sorted(archetypes(), key=len, reverse=True):
        if f" {arch} " in text:
            setcode = archetypes()[arch]
            text = text.replace(f" {arch} ", " ")
            break
    all_of = any_of = 0
    for pat, bits in _TYPE_WORDS:
        if re.search(rf"\b(?:{pat})\b", text):
            text = re.sub(rf"\b(?:{pat})\b", " ", text)
            if bits == _ANY_OF:
                any_of |= bits
            else:
                all_of |= bits
    attr = 0
    for word, bit in _ATTRS.items():
        if re.search(rf"\b{word}\b", text):
            attr |= bit
    return setcode, all_of, any_of, attr


def resolve_generic(name: str, deck_codes: Iterable[int], cards: Mapping[int, CardData]) -> list[int]:
    """Distinct codes of the deck matching the description; [] if it names no archetype/attribute constraint."""
    setcode, all_of, any_of, attr = parse(name)
    if setcode is None and not attr:
        return []  # "lock monster", "Spell": too vague to verify against
    out = []
    for code in sorted(set(deck_codes)):
        cd = cards.get(code)
        if cd is None:
            continue
        if setcode is not None and not _in_set(cd, setcode):
            continue
        if all_of and (cd.type & all_of) != all_of:
            continue
        if any_of and not cd.type & any_of:
            continue
        if attr and not cd.attribute & attr:
            continue
        out.append(code)
    return out
