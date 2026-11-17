"""Rank verified going-first lines by the interruptions they leave, to pick step-2 combo opponents
(docs/plans/2026-11-03-opp-turn-interaction.md; NOTEBOOK 2026-11-03 21:00 "Verified lines read directly").

Each record is replayed to the end of turn 1 (PyEnv; the field and hand are read from the engine). A field interruption
is a card player 0 controls that can act on the opponent's turn (Trap, Quick-Play Spell, or timed text such as "Quick
Effect" / "during your opponent's turn") and whose text disrupts the opponent (negates; destroys / banishes / returns
cards the opponent controls or on the field; "your opponent cannot"). Hand interruptions (`h_int`) are disrupting cards
that can act from the hand (hand traps, Impermanence), counted apart: hand luck, not the combo's work.
`acts` counts player 0's chosen activations and summons: the windows an opponent of this line gets to respond in,
which is what step 2 trains (a trap pile with few acts gives our bot little to interrupt).
Per (deck, opening hand) the best line is the one with the most field interruptions, then hand ones, then field size.
This selects training opponents only; nothing here enters our bot's play (memory: no hand-encoded play).

    uv run python scripts/audit/rank_lines.py --decks branded,agents --out /tmp/rank.jsonl
    uv run python scripts/audit/rank_lines.py --all --out data/cache/combo_lines/ranked.jsonl

Writes one JSON line per record (deck, line_id, hand key, field/hand interruption names, best_for_hand) and prints a
per-deck summary. Exits 3 when no line replays.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import playtest as PT  # noqa: E402

import hand_play as H  # noqa: E402

from decklight.demos.dataset import record_config  # noqa: E402
from decklight.demos.game import _opts  # noqa: E402
from decklight.engine import constants as C  # noqa: E402
from decklight.engine.env import PyEnv  # noqa: E402
from decklight.engine.duel import Engine  # noqa: E402

TYPE_SPELL, TYPE_TRAP, TYPE_QUICKPLAY = 0x2, 0x4, 0x10000
HAND_LOC = 2
TIMING = re.compile(r"quick effect|during your opponent's turn|during either player's turn|when your opponent|"
                    r"if your opponent (?:normal or special |special )?summons|when a card or effect is activated", re.I)
ACTS = {"ACTIVATE", "SPSUMMON", "SUMMON"}  # player-0 choices that open a response window for the opponent
# an effect on the opponent: negation, or removal / lock aimed at their cards (not self-protection or a drawback)
DISRUPT = re.compile(r"negate|(?:destroy|banish|return|shuffle|send|change|take control)[^.]{0,80}?"
                     r"(?:your opponent controls|your opponent's|on the field|in your opponent's)|"
                     r"(?:your opponent controls|on the field|your opponent's)[^.;]{0,40}; [^.]{0,20}?"
                     r"(?:destroy|banish|return|shuffle|send|change|take control)|"  # "Target 1 ... on the field; banish it"
                     r"(?:your opponent|neither player) (?:cannot|can't|can only)", re.I)
# a trigger on the opponent's action that removes or negates (Bottomless Trap Hole: "When your opponent Summons ...:
# Destroy that monster")
ON_OPP = re.compile(r"(?:when|if) your opponent [^:.]{0,80}: [^.]{0,40}?(?:destroy|banish|negate|return)", re.I)
FROM_HAND = re.compile(r"discard this card|send this card from your hand|reveal this card in your hand|"
                       r"activate this card from your hand|banish this card from your hand", re.I)
EXTRA = 0x40 | 0x2000 | 0x800000 | 0x4000000  # Fusion, Synchro, Xyz, Link: never in hand


def interruption_codes(cdb: Path) -> tuple[set[int], set[int]]:
    """(field interruptions, hand interruptions). Field: a Trap, Quick-Play Spell or timed card that disrupts the
    opponent. Hand: a disrupting card whose text lets it act from the hand (hand traps, Impermanence)."""
    db = sqlite3.connect(cdb)
    field, hand = set(), set()
    for code, typ, desc in db.execute("select d.id, d.type, t.desc from datas d join texts t on d.id = t.id"):
        desc = desc or ""
        if not (DISRUPT.search(desc) or ON_OPP.search(desc)):
            continue
        if typ & TYPE_TRAP or (typ & TYPE_SPELL and typ & TYPE_QUICKPLAY) or TIMING.search(desc):
            field.add(int(code))
        if FROM_HAND.search(desc) and not typ & EXTRA and (TIMING.search(desc) or typ & TYPE_TRAP):
            hand.add(int(code))
    return field, hand


def end_cards(engine, rec: dict) -> tuple[list[int], list[int], int]:
    """Player 0's field and hand card codes at the end of turn 1 (the replay stops when turn 2 starts, as
    `playtest._replay_field`), read from the engine, and how many activations / summons player 0 chose on the way.
    Kind.SPSUMMON is an idle-command special summon (a procedure such as Link or Xyz Summon), so every counted choice
    opens a response window for the opponent; effect special summons are not choices and are not counted."""
    env = PyEnv(engine, record_config(rec), auto_single=rec.get("auto_single", True))
    obs, acts = env.reset(), 0
    for c in rec["choices"]:
        if obs is None or env.duel.turn > 1:
            break
        if env.player == 0:
            o = _opts(obs)
            acts += int(c) < len(o) and H._kind(o[int(c)][0]) in ACTS
        obs = env.step(int(c))

    def codes(loc: int) -> list[int]:
        return [int(x.code) for x in env.duel.query_location(0, loc) if x is not None]

    return codes(C.LOCATION_MZONE) + codes(C.LOCATION_SZONE), codes(C.LOCATION_HAND), acts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.all:
        names = sorted({p.stem for p in ROOT.glob("data/demos/*/verified/*.jsonl")}
                       | {p.stem for p in ROOT.glob("data/demos/planned*/improved/verified/*.jsonl")})  # fmt: skip
    else:
        names = [n for n in a.decks.split(",") if n]
    cdb = ROOT / "third_party/BabelCDB/cards.cdb"
    inter, hand_inter = interruption_codes(cdb)
    nm = dict(sqlite3.connect(cdb).execute("select id, name from texts"))
    engine = Engine()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with a.out.open("w") as fo:
        print(f"{'deck':44s} lines hands  best: acts f_int h_int field | all: acts f_int  field  failed")
        for name in names:
            rows, failed = [], 0
            for rec in PT.verified_records(name, limit=1_000_000):
                try:
                    # record_config resolves its own setup from the record; the env's view of face-down cards is
                    # the owner's, so set cards keep their codes.
                    field, hand, acts = end_cards(engine, rec)
                except Exception:  # noqa: BLE001 - a record that no longer replays is counted, not fatal
                    failed += 1
                    continue
                key = tuple(sorted(int(s[1]) for s in rec["config"].get("setup", []) if s[0] == 0 and s[2] == HAND_LOC))
                rows.append({"deck": name, "line_id": rec["line_id"], "hand": list(key),
                             "acts": acts, "field": len(field), "f_int": sum(c in inter for c in field),
                             "h_int": sum(c in hand_inter for c in hand),
                             "field_cards": [nm.get(c, str(c)) for c in field],
                             "int_cards": [nm.get(c, str(c)) for c in field if c in inter]
                             + [nm.get(c, str(c)) + " (hand)" for c in hand if c in hand_inter]})  # fmt: skip
            by_hand = defaultdict(list)
            for r in rows:
                by_hand[tuple(r["hand"])].append(r)
            best = [max(rs, key=lambda r: (r["f_int"], r["h_int"], r["field"])) for rs in by_hand.values()]
            for r in rows:
                r["best_for_hand"] = any(r is b for b in best)
                fo.write(json.dumps(r) + "\n")
            n_ok += len(rows)
            if rows:
                m = statistics.mean
                print(f"{name:44s} {len(rows):5d} {len(best):5d}  {m(r['acts'] for r in best):5.1f} {m(r['f_int'] for r in best):5.2f} "
                      f"{m(r['h_int'] for r in best):5.2f} {m(r['field'] for r in best):5.2f} | "
                      f"{m(r['acts'] for r in rows):5.1f} {m(r['f_int'] for r in rows):5.2f} {m(r['field'] for r in rows):5.2f}  {failed}",
                      flush=True)  # fmt: skip
            else:
                print(f"{name:44s} no line replays ({failed} failed)", flush=True)
    if n_ok == 0:
        print("ALERT rank_lines: no line replayed", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
