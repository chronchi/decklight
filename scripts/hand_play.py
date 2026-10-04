"""Hand-play a full duel (both seats) through the engine and save it as a "gold" game record.

A person (or Claude) drives every decision by kind + card name, with a short reason; zone (PLACE) prompts are
answered automatically; positions are chosen with POS:attack / POS:defense. State lives in data/demos/gold_work/<game>/state.json,
so each command is a fresh process that replays the duel.

    uv run python scripts/hand_play.py new  g1 snake-eye tenpai-dragon 7
    uv run python scripts/hand_play.py show g1 [--texts]
    uv run python scripts/hand_play.py do   g1 "ACT:Pre-Preparation SEL:Saji P END" --why "search the starter"
    uv run python scripts/hand_play.py undo g1 3
    uv run python scripts/hand_play.py save g1      # -> data/demos/gold/<d0>-vs-<d1>.jsonl

Tokens: an option number; P (pass/no/cancel at a prompt); KIND:name-substring where KIND is a prefix of the
option kind (ACT, SUM, SPS, MSET, SSET, SEL, UNSEL, YES, NO, ATK, OPT, FIN, CANCEL) or one of END (TO_END),
BP (TO_BATTLE), M2 (TO_MAIN2). `do` stops at the first token that matches no option and prints the state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import load_ydk
from ygo_zero.demos.game import _effect_text, _opts, chain_names, describe_state, hidden_card, option_desc
from ygo_zero.demos.names import CardNames
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine

WORK = ROOT / "data" / "demos" / "gold_work"
OUT = ROOT / "data" / "demos" / "gold"
_ALIASES = {"END": Kind.TO_END, "BP": Kind.TO_BATTLE, "M2": Kind.TO_MAIN2, "ATK": Kind.ATTACK}
_PASSIVE = (Kind.PASS, Kind.NO, Kind.CANCEL, Kind.FINISH)
_MSGS = {v: k for k, v in vars(C).items() if k.startswith("MSG_")}


def _load(game: str) -> dict:
    return json.loads((WORK / game / "state.json").read_text())


def _store(game: str, st: dict) -> None:
    (WORK / game).mkdir(parents=True, exist_ok=True)
    (WORK / game / "state.json").write_text(json.dumps(st))


def _config(st: dict) -> DuelConfig:
    d0 = load_ydk(ROOT / f"decks/pool/{st['decks'][0]}.ydk")
    d1 = load_ydk(ROOT / f"decks/pool/{st['decks'][1]}.ydk")
    return DuelConfig(d0, d1, seed=st["seed"])


def _replay(engine: Engine, st: dict):
    env = CppEnv(engine, _config(st))
    obs = env.reset()
    for i in st["choices"]:
        obs = env.step(i)
        if obs is None:
            break
    return env, obs


def _kind(k: int) -> str:
    try:
        return Kind(k).name
    except ValueError:
        return str(k)


def _match(obs, tok: str, names: dict[int, str]) -> int | None:
    opts = _opts(obs)
    if tok.isdigit():
        i = int(tok)
        return i if i < len(opts) else None
    if tok.upper() == "P":
        return next((i for i, (k, _) in enumerate(opts) if k in _PASSIVE), None)
    head, _, sub = tok.partition(":")
    head, sub = head.upper(), sub.replace("_", " ").lower()
    best = None  # (rank, index): exact name < name prefix < substring of name/description
    for i, (k, code) in enumerate(opts):
        kname = _kind(k)
        ok_kind = (k == _ALIASES[head]) if head in _ALIASES else kname.startswith(head)
        nm = names.get(code, "").lower()
        if not ok_kind:
            continue
        rank = 0 if not sub or nm == sub else 1 if nm.startswith(sub) else 2 if sub in f"{nm} {option_desc(obs, i)}".lower() else None
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, i)
    return best[1] if best else None


def _auto(env, obs, st: dict):
    """Answer zone prompts with their first option (positions are real decisions and wait for a token)."""
    while obs is not None and int(obs["global"][21]) == C.MSG_SELECT_PLACE:
        st["choices"].append(0)
        st["labels"].append("auto")
        st["actors"].append(env.player)
        st["turns"].append(int(obs["global"][2]))
        obs = env.step(0)
    return obs


def _show(env, obs, st: dict, names: CardNames, texts: bool) -> None:
    if obs is None:
        print(f"DUEL OVER: winner {env.winner}")
        return
    p = env.player
    g = obs["global"]
    print(f"[{len(st['choices'])}] P{p} {st['decks'][p]} to act — {_MSGS.get(int(g[21]))}")
    print(describe_state(obs, names.names).replace("Your banished: none\n", "").replace("Opponent banished: none\n", ""))
    ch = chain_names(obs, names.names)
    if ch:
        print("CHAIN:", " -> ".join(ch))
    seen = set(st.setdefault("seen", []))
    for i, (k, code) in enumerate(_opts(obs)):
        nm = names.names.get(code, f"#{code}") if code else hidden_card(obs, i)
        what = option_desc(obs, i)
        nm = f'{nm} "{what}"' if what else nm
        txt = ""
        if code and (texts or code not in seen):
            txt = " | " + _effect_text(code)
            seen.add(code)
        print(f"  {i}: {_kind(k)} {nm}{txt}")
    st["seen"] = sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["new", "show", "do", "undo", "save"])
    ap.add_argument("game")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--why", default="")
    ap.add_argument("--texts", action="store_true")
    ap.add_argument("--cross", action="store_true", help="let tokens continue after the acting seat changes")
    a = ap.parse_args()
    engine, names = Engine(), CardNames()
    if a.cmd == "new":
        d0, d1, seed = a.rest
        st = {"decks": [d0, d1], "seed": int(seed), "choices": [], "labels": [], "actors": [], "turns": [], "notes": []}
        env, obs = _replay(engine, st)
        obs = _auto(env, obs, st)
        _show(env, obs, st, names, a.texts)
        _store(a.game, st)
        return
    st = _load(a.game)
    if a.cmd == "undo":
        n = int(a.rest[0]) if a.rest else 1
        for key in ("choices", "labels", "actors", "turns"):
            st[key] = st[key][:-n] if n else st[key]
        st["notes"] = [x for x in st["notes"] if x[0] < len(st["choices"])]
        # also drop trailing auto answers so the state ends on a real decision
        while st["labels"] and st["labels"][-1] == "auto":
            for key in ("choices", "labels", "actors", "turns"):
                st[key].pop()
    env, obs = _replay(engine, st)
    if a.cmd == "do":
        start = len(st["choices"])
        actor = env.player if obs is not None else None
        for tok in " ".join(a.rest).split():
            if obs is None:
                break
            if env.player != actor and not a.cross:  # never let leftover tokens decide for the other seat
                print(f"!! player changed to P{env.player}; stopped before {tok!r}")
                break
            i = _match(obs, tok, names.names)
            if i is None:
                print(f"!! token {tok!r} matches no option; stopped here")
                break
            st["choices"].append(i)
            st["labels"].append("gold")
            st["actors"].append(env.player)
            st["turns"].append(int(obs["global"][2]))
            obs = env.step(i)
            obs = _auto(env, obs, st)
        if a.why and len(st["choices"]) > start:
            st["notes"].append([start, a.why])
    if a.cmd == "save":
        OUT.mkdir(parents=True, exist_ok=True)
        cfg = _config(st)
        rec = {
            "game_id": f"gold-{a.game}",
            "line_id": f"{st['decks'][0]}-vs-{st['decks'][1]}/gold-{a.game}",
            "level": "L4",
            "decks": st["decks"],
            "config": {
                "deck0_main": list(cfg.deck0.main), "deck0_extra": list(cfg.deck0.extra),
                "deck1_main": list(cfg.deck1.main), "deck1_extra": list(cfg.deck1.extra),
                "seed": cfg.seed, "hand": cfg.hand, "draw": cfg.draw, "lp": cfg.lp, "setup": [],
                "shuffle": cfg.shuffle,
            },
            "auto_single": True,
            "choices": st["choices"], "labels": st["labels"], "actors": st["actors"], "turns": st["turns"],
            "notes": st["notes"],
            "winner": env.winner if obs is None else None,
            "end_turn": st["turns"][-1] if st["turns"] else 0,
            "source": "hand-played (Claude), both seats",
        }  # fmt: skip
        path = OUT / f"{st['decks'][0]}-vs-{st['decks'][1]}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"saved {rec['game_id']} ({len(st['choices'])} choices, winner {rec['winner']}) -> {path}")
        return
    _show(env, obs, st, names, a.texts)
    _store(a.game, st)


if __name__ == "__main__":
    main()
