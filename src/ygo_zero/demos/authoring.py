"""Hand-authoring combo lines (Claude / Sonnet agents, ADR 0003 second source).

A line is a list of step dicts built with the helpers below plus the expected end board, keyed by the index of a
sampled hand (`plan.sample_hands(deck, n, seed)`, the same hands `scripts/deck_sheet.py` prints):

    from ygo_zero.demos.authoring import *
    L = {0: ([ns("Ash Blossom & Joyous Spring"), sset("Infinite Impermanence")], ["Ash Blossom & Joyous Spring", ...])}
    write_lines("snake-eye", L, seed=0, source="sonnet")   # -> data/demos/sonnet/lines/snake-eye.jsonl

Card names must be exact card-DB names; `write_lines` reports unresolved names. Verify with
`scripts/verify_lines.py --lines <file> --out <verified> [--remote]`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ygo_zero.decks import load_ydk
from ygo_zero.demos.extract import _fill_codes
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.plan import sample_hands
from ygo_zero.demos.schema import ComboLine

ROOT = Path(__file__).resolve().parents[3]

__all__ = [
    "S",
    "act",
    "attack",
    "fus",
    "link",
    "mset",
    "ns",
    "rit",
    "ss",
    "sset",
    "syn",
    "write_lines",
    "xyz",
]


def S(action: str, card: str, results=(), mats=(), targets=(), text: str = "") -> dict:
    """One step: action (see schema.ACTIONS), the acting card, what it produces, materials/costs, targets."""
    return {"action": action, "card": card, "results": list(results), "materials": list(mats), "targets": list(targets),
                "effect": text, "text": text}  # fmt: skip


def act(card, results=(), mats=(), targets=(), text=""):
    """Activate a card or effect. results = cards it adds/summons/sets; mats = costs/discards/tributes."""
    return S("activate", card, results, mats, targets, text)


def ns(card):
    return S("normal_summon", card, [card])


def mset(card):
    return S("set_monster", card, [card])


def ss(card, text=""):
    """A procedure Special Summon (from the card's own condition, e.g. 'if you control no monsters')."""
    return S("special_summon", card, [card], text=text)


def syn(card, mats):
    return S("synchro_summon", card, [card], mats)


def xyz(card, mats):
    return S("xyz_summon", card, [card], mats)


def link(card, mats):
    return S("link_summon", card, [card], mats)


def fus(card, mats):
    return S("fusion_summon", card, [card], mats)


def rit(card, mats):
    return S("ritual_summon", card, [card], mats)


def sset(card):
    return S("set_spelltrap", card, [card])


def attack(card, target=""):
    return S("attack", card, targets=[target] if target else [])


def write_lines(
    deck: str, L: dict, seed: int = 0, source: str = "sonnet", n_hands: int = 20, append: bool = False
) -> Path:
    """Write {hand_index: (steps, end_board)} as ComboLines for `decks/pool/<deck>.ydk` hands sampled with `seed`."""
    names = CardNames()
    d = load_ydk(ROOT / "decks" / "pool" / f"{deck}.ydk")
    hands = sample_hands(d, n_hands, seed)
    out = []
    for k, (steps, end) in sorted(L.items()):
        steps = [dict(s, i=i) for i, s in enumerate(steps)]
        ln = ComboLine.from_json({
            "id": f"{deck}/{source}-s{seed}-h{k}", "deck": deck, "source_file": source, "title": f"{source} plan, seed {seed} hand {k}",
            "hand": [names.names[c] for c in hands[k]], "requires": [], "steps": steps, "end_board": list(end), "going": "first",
        })  # fmt: skip
        _fill_codes(ln, names)
        if ln.unresolved_names:
            print(f"hand {k}: UNRESOLVED {ln.unresolved_names}")
        out.append(json.dumps(ln.to_json()))
    p = ROOT / "data" / "demos" / source / "lines" / f"{deck}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a" if append else "w") as f:
        f.write("\n".join(out) + "\n")
    print(deck, len(out), "lines ->", p.relative_to(ROOT))
    return p
