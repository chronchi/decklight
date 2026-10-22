"""Playability check for custom cards (extra EDOPro card DBs + scripts via `YGO_EXTRA_CARDS`).

With `YGO_EXTRA_CARDS` set, plays random-vs-random games (`PyEnv`, random option indices) of
`--deck` against a pool deck and reports:
- script errors from the core log handler (`engine/duel.py` per-duel log), each with the card code;
- for each custom code in the deck: drawn or seen, activated (an ACTIVATE history entry), summoned;
- games finished vs stuck.

Exits non-zero on any script error or on a custom card that never got activated or summoned
across all games (printed: it may just be rare).

    YGO_EXTRA_CARDS=data/cache/custom/zedja uv run python scripts/check_custom.py \
        --deck decks/custom/wiccanthrope.ydk --games 20
"""

import argparse
import random
import re
import sqlite3
import sys
from pathlib import Path

from decklight.decks import load_ydk
from decklight.engine.actions import Kind
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.engine.resources import default_cdbs, extra_card_dirs

ROOT = Path(__file__).resolve().parents[1]
LOG_ERROR = 0  # OCG_LOG_TYPE_ERROR (ocgapi_types.h)
_SUMMON_KINDS = frozenset({int(Kind.SUMMON), int(Kind.SPSUMMON)})
_SCRIPT_RE = re.compile(r"c(\d+)\.lua")
_CODE_RE = re.compile(r"\b(\d{7,10})\b")


def extra_codes() -> set[int]:
    """Codes defined in the `YGO_EXTRA_CARDS` cdbs (fail fast when the env var finds nothing)."""
    dirs = extra_card_dirs()
    if not dirs:
        sys.exit("ALERT check_custom: YGO_EXTRA_CARDS is unset or names no existing dirs")
    out: set[int] = set()
    for d in dirs:
        for cdb in sorted(d.glob("*.cdb")):
            con = sqlite3.connect(f"file:{cdb}?mode=ro", uri=True)
            try:
                out.update(int(r[0]) for r in con.execute("SELECT id FROM datas").fetchall())
            finally:
                con.close()
    if not out:
        sys.exit(f"ALERT check_custom: no card codes in {dirs}")
    return out


def card_names() -> dict[int, str]:
    out: dict[int, str] = {}
    for path in default_cdbs():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for code, name in con.execute("SELECT id, name FROM texts").fetchall():
                out[int(code)] = str(name)
        finally:
            con.close()
    return out


def error_code(msg: str, customs: set[int]) -> int:
    """Card code for a core log error: the `c<code>.lua` script name, else a custom code in the
    message, else 0 (unknown)."""
    m = _SCRIPT_RE.search(msg)
    if m:
        return int(m.group(1))
    for m in _CODE_RE.finditer(msg):
        if int(m.group(1)) in customs:
            return int(m.group(1))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", type=Path, required=True, help="custom .ydk under test")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--opp", default="blue-eyes", help="pool deck name (decks/pool/<name>.ydk) or a .ydk path")
    ap.add_argument("--seed", type=int, default=0, help="first game seed")
    ap.add_argument("--max-steps", type=int, default=3000, help="agent steps per game before it counts as stuck")
    args = ap.parse_args()

    customs = extra_codes()
    deck = load_ydk(args.deck)
    deck_custom = sorted((set(deck.main) | set(deck.extra)) & customs)
    if not deck_custom:
        sys.exit(f"ALERT check_custom: no custom codes in {args.deck} (YGO_EXTRA_CARDS={extra_card_dirs()})")
    opp_path = Path(args.opp) if args.opp.endswith(".ydk") else ROOT / "decks" / "pool" / f"{args.opp}.ydk"
    opp = load_ydk(opp_path)
    names = card_names()
    print(f"deck {args.deck}: {len(deck.main)} main + {len(deck.extra)} extra, "
          f"{len(deck_custom)} custom codes; opp {opp_path.stem}", flush=True)

    engine = Engine()
    missing = [c for c in (*deck.main, *deck.extra) if c not in engine.cards]
    if missing:
        sys.exit(f"ALERT check_custom: deck has {len(missing)} codes missing from the card DB, e.g. {missing[:3]} "
                 f"(is YGO_EXTRA_CARDS set?)")

    seen: set[int] = set()
    activated = {c: 0 for c in deck_custom}
    summoned = {c: 0 for c in deck_custom}
    errors: list[tuple[int, int, str]] = []  # (game, code, message)
    finished = stuck = 0
    for g in range(args.games):
        rng = random.Random(args.seed + g)
        decks = (deck, opp) if g % 2 == 0 else (opp, deck)
        env = PyEnv(engine, DuelConfig(decks[0], decks[1], seed=args.seed + g))
        obs = env.reset()
        steps = 0
        game_seen: set[int] = set()
        while obs is not None and steps < args.max_steps:
            game_seen.update(env.board.visible_codes())
            obs = env.step(rng.randrange(int(obs["action_mask"].sum())))
            steps += 1
        if obs is None:
            finished += 1
        else:
            stuck += 1
        for e in env.history:
            code = int(e.row[2])
            if code in customs:
                game_seen.add(code)
                if code in activated and int(e.row[0]) == int(Kind.ACTIVATE):
                    activated[code] += 1
                if code in summoned and int(e.row[0]) in _SUMMON_KINDS:
                    summoned[code] += 1
        seen.update(c for c in game_seen if c in customs)
        for t, msg in env.duel.log:
            if t == LOG_ERROR:
                errors.append((g, error_code(msg, customs), msg))
        env.duel.close()
        print(f"game {g}: {'done' if obs is None else 'STUCK'} winner={env.winner} "
              f"steps={steps} custom_seen={sorted(c for c in game_seen if c in customs)}", flush=True)

    print(f"\ngames finished {finished}/{args.games}, stuck {stuck}", flush=True)
    print(f"script errors: {len(errors)}", flush=True)
    for g, code, msg in errors:
        print(f"  game {g} code {code}: {msg[:200]}", flush=True)
    print("custom code  name  seen  activated  summoned", flush=True)
    for c in deck_custom:
        print(f"  {c}  {names.get(c, '?')}  {c in seen}  {activated[c]}  {summoned[c]}", flush=True)

    never = [c for c in deck_custom if not activated[c] and not summoned[c]]
    if never:
        print(f"NEVER USED: {[(c, names.get(c, '?')) for c in never]}", flush=True)
    if errors or never:
        sys.exit(f"ALERT check_custom: {len(errors)} script errors, {len(never)} custom cards never "
                 f"activated or summoned ({[names.get(c, c) for c in never]})")
    print("OK: no script errors, every custom card activated or summoned", flush=True)


if __name__ == "__main__":
    main()
