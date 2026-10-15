"""Render search_match game records as Markdown a human can review (engine replay only, no model; laptop-safe).

    uv run python scripts/render_showcase.py runs/<job>/<variant>/games-0.jsonl --idx 12 --out docs/showcase/g1.md \
        --label0 "Our AI (P + search)" --label1 "ygo-agent" --title "Game 1"

Per game: the decks and who goes first, the result, then turn by turn:
- both hands and both boards at the turn's first decision (spectator view: both hands are shown; the players did not
  see each other's hands; our AI's search only samples hidden cards it could know about);
- every real decision (passes and forced moves left out) in plain words, with the engine's prompt for card selections;
- where the search overrode the network's first choice ("search: chose X over the network's Y").
The replay must reproduce the record's winner, else the script exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hand_play as H

from ygo_zero.decks import load_ydk
from ygo_zero.demos.game import _opts, hidden_card, option_desc
from ygo_zero.demos.names import CardNames
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine

VERB = {
    Kind.SUMMON: "Normal Summons", Kind.SPSUMMON: "Special Summons", Kind.REPOS: "changes the position of",
    Kind.MSET: "sets monster", Kind.SSET: "sets", Kind.ACTIVATE: "activates", Kind.ATTACK: "attacks with",
    Kind.TO_BATTLE: "enters the Battle Phase", Kind.TO_MAIN2: "goes to Main Phase 2", Kind.TO_END: "ends the turn",
    Kind.YES: "answers YES", Kind.NO: "answers NO", Kind.OPTION: "chooses", Kind.SELECT: "selects",
    Kind.UNSELECT: "unselects", Kind.FINISH: "finishes the selection", Kind.CANCEL: "cancels", Kind.PLACE: "places in",
    Kind.POSITION: "chooses position", Kind.NUMBER: "declares the number", Kind.ATTRIBUTE: "declares attribute",
    Kind.RACE: "declares type", Kind.DECLARE: "declares the card", Kind.SORT: "orders", Kind.COUNTER: "removes counters from",
    Kind.RPS: "plays rock-paper-scissors", Kind.PASS: "passes",
}  # fmt: skip
SKIP = {Kind.PASS, Kind.PLACE}  # passes and zone picks: noise for a reader


def deck_path(name: str) -> Path:
    for d in ("pool", "gauntlet", "newcards"):
        p = ROOT / "decks" / d / f"{name}.ydk"
        if p.is_file():
            return p
    sys.exit(f"deck {name} not found in decks/pool|gauntlet|newcards")


def board(env, names: dict, p: int) -> list[str]:
    c = env.cards()
    n = len(c["code"])

    def nm(i: int) -> str:
        return names.get(int(c["code"][i]), f"#{int(c['code'][i])}")

    def where(loc: int) -> list[int]:
        return sorted((i for i in range(n) if int(c["player"][i]) == p and int(c["location"][i]) == loc),
                      key=lambda i: int(c["sequence"][i]))  # fmt: skip

    def pos(i: int) -> str:
        q = int(c["position"][i])
        return {C.POS_FACEUP_ATTACK: "ATK", C.POS_FACEDOWN_ATTACK: "face-down", C.POS_FACEUP_DEFENSE: "DEF",
                C.POS_FACEDOWN_DEFENSE: "face-down DEF"}.get(q, "set" if q & C.POS_FACEDOWN else "")  # fmt: skip

    out = []
    hand = where(C.LOCATION_HAND)
    out.append(f"hand ({len(hand)}): " + (", ".join(nm(i) for i in hand) or "-"))
    mz = where(C.LOCATION_MZONE)
    out.append("monsters: " + (", ".join(f"{nm(i)} ({pos(i)})" for i in mz) or "-"))
    sz = where(C.LOCATION_SZONE)
    out.append("spells/traps: " + (", ".join(nm(i) + (" (set)" if int(c["position"][i]) & C.POS_FACEDOWN else "")
                                             for i in sz) or "-"))  # fmt: skip
    gy, ban = where(C.LOCATION_GRAVE), where(C.LOCATION_REMOVED)
    out.append(f"GY ({len(gy)}): " + (", ".join(nm(i) for i in gy) or "-"))
    if ban:
        out.append(f"banished ({len(ban)}): " + ", ".join(nm(i) for i in ban))
    out.append(f"deck {len(where(C.LOCATION_DECK))}, extra deck {len(where(C.LOCATION_EXTRA))}")
    return out


def describe(env, obs, i: int, names: dict) -> str:
    kind, code = _opts(obs)[i]
    try:
        k = Kind(kind)
    except ValueError:
        return f"option {i}"
    if k == Kind.NUMBER:
        return f"{VERB[k]} {H._arg_of(obs, i)}"
    what = names.get(code, f"#{code}") if code else (hidden_card(obs, i) or "")
    if k in (Kind.YES, Kind.NO) and code:
        return ("uses the effect of " if k == Kind.YES else "declines the effect of ") + what
    desc = option_desc(obs, i)
    s = f"{VERB.get(k, k.name)} {what}".strip()
    if desc and k in (Kind.ACTIVATE, Kind.OPTION, Kind.YES, Kind.NO):
        s += f' ("{desc}")'
    elif desc and k == Kind.POSITION:
        s = f"{VERB[k]} {desc}"
    return s


def render(rec: dict, labels: tuple[str, str], title: str, names: dict, engine: Engine) -> str:
    d0, d1 = rec["decks"]
    cfg = DuelConfig(load_ydk(deck_path(d0)), load_ydk(deck_path(d1)), seed=rec["seed"])
    who = [f"{labels[0]} ({d0})", f"{labels[1]} ({d1})"]
    search = {int(e["k"]): e for e in rec.get("search", [])}
    env = CppEnv(engine, cfg)
    obs = env.reset()
    body: list[str] = []
    cur_turn = None
    for k, ch in enumerate(rec["choices"]):
        if obs is None:
            sys.exit(f"game {rec['game_idx']}: replay ended before choice {k}")
        g = obs["global"]
        actor = int(env.player)
        turn = int(g[2])
        if turn != cur_turn:
            cur_turn = turn
            tp = actor if int(g[5]) == 1 else 1 - actor
            lp = (int(g[0]), int(g[1])) if actor == 0 else (int(g[1]), int(g[0]))
            body.append(
                f"\n## Turn {turn}: {who[tp]}'s turn  (LP {labels[0]} {lp[0]} / {labels[1]} {lp[1]})\n"
            )
            body.append("At the start of the turn:\n")
            for p in (0, 1):
                body.append(f"- **{labels[p]}**")
                body.extend(f"  - {x}" for x in board(env, names, p))
            body.append("\nPlays:\n")
        n_opts = int(obs["action_mask"].sum())
        kind = _opts(obs)[int(ch)][0]
        e = search.get(k)
        overridden = (
            e is not None
            and int(e["chosen"]) != int(e["played"])
            and kind != Kind.PLACE
            and describe(env, obs, int(e["chosen"]), names) != describe(env, obs, int(e["played"]), names)
        )
        if n_opts > 1 and (kind not in SKIP or overridden):
            line = f"- {labels[actor]} {describe(env, obs, int(ch), names)}"
            msg = int(g[21])
            hint = H._hint(env, obs) if msg in H._MULTI_MSGS or msg == C.MSG_SELECT_CHAIN else ""
            if hint:
                line += f"  _[{hint}]_"
            if overridden:
                line += f"  **(search: chose this over the network's first choice: {describe(env, obs, int(e['played']), names)})**"
            body.append(line)
        last_lp = (int(g[0]), int(g[1])) if actor == 0 else (int(g[1]), int(g[0]))
        obs = env.step(int(ch))
    if rec.get("truncated"):
        result = "cut at the decision cap (no winner)"
    else:
        if obs is not None or int(env.winner) != rec["winner"]:
            print(
                f"ALERT game {rec['game_idx']}: replay winner {env.winner} != record {rec['winner']}",
                file=sys.stderr,
            )
            sys.exit(2)
        result = f"**{who[rec['winner']]} wins** on turn {rec['turns']}"
        body.append(
            f"\n## Game over: {who[rec['winner']]} wins (LP before the final blow: {labels[0]} {last_lp[0]} / "
            f"{labels[1]} {last_lp[1]})"
        )
    n_s = sum(1 for e in rec.get("search", []))
    n_o = sum(1 for e in rec.get("search", []) if int(e["chosen"]) != int(e["played"]))
    head = [
        f"# {title}: {who[0]} vs {who[1]}",
        "",
        f"- Goes first: {who[0]}",
        f"- Result: {result}",
        "- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).",
        (
            f"- Deal seed {rec['seed']}; {len(rec['choices'])} decisions in total, {n_s} searched, {n_o} changed by the "
            "search (changes of zone or between identical-looking options are not marked)"
        ),
        "- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).",
        "- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.",
    ]
    return "\n".join(head + body) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", type=Path, help="a search_match games-*.jsonl")
    ap.add_argument("--idx", type=int, required=True, help="game_idx of the record")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label0", default="Player 1", help="name for seat 0 (goes first)")
    ap.add_argument("--label1", default="Player 2")
    ap.add_argument("--title", default="Game")
    a = ap.parse_args()
    rec = next(
        (json.loads(x) for x in a.games.read_text().splitlines() if json.loads(x)["game_idx"] == a.idx), None
    )
    if rec is None:
        sys.exit(f"no game_idx {a.idx} in {a.games}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(rec, (a.label0, a.label1), a.title, CardNames().names, Engine()))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
