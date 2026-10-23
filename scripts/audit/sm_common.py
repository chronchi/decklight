"""Shared helpers for the search_match behaviour-audit scripts (render_sm, flip_audit, ...).

A search_match `--opp ygoagent` game record holds `decks` in SEAT order ([seat0 deck, seat1
deck]), `seed`, `s_seat` (the seat S pilots; S's deck is `decks[s_seat]`), `choices` + `actors`,
`winner`, `truncated`. The duel config is rebuilt from the gauntlet .ydk files in seat order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk
from decklight.demos.game import _opts, hidden_card, option_desc
from decklight.demos.names import CardNames
from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine

GAUNTLET = ROOT / "decks" / "gauntlet"

_MSG_NAMES = {v: k.replace("MSG_", "") for k, v in vars(C).items() if k.startswith("MSG_") and isinstance(v, int)}

_decks: dict[str, object] = {}


def load_deck(stem: str, gauntlet: Path = GAUNTLET):
    """A gauntlet deck by stem (cached)."""
    key = str(gauntlet / stem)
    if key not in _decks:
        _decks[key] = load_ydk(gauntlet / f"{stem}.ydk")
    return _decks[key]


def seat_decks(rec: dict) -> tuple[str, str]:
    """Deck names in duel-seat order (seat 0 first). Our bot pilots `decks[s_seat]`."""
    return (rec["decks"][0], rec["decks"][1])


def duel_config(rec: dict, gauntlet: Path = GAUNTLET) -> DuelConfig:
    """The DuelConfig that reproduces a search_match ygoagent record."""
    n0, n1 = seat_decks(rec)
    return DuelConfig(load_deck(n0, gauntlet), load_deck(n1, gauntlet), seed=rec["seed"])


def iter_records(arm_dir: Path) -> list[dict]:
    """All game records under an arm dir (games-*.jsonl)."""
    out = []
    for f in sorted(arm_dir.glob("games-*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def deal_key(rec: dict) -> tuple:
    """The deal identity shared across arms: (decks, seed, s_seat)."""
    return (tuple(rec["decks"]), rec["seed"], rec["s_seat"])


def our_score(rec: dict) -> float:
    """S's score: 1 win, 0 loss, 0.5 draw/cut (as in deck_card / paired_sm)."""
    if rec.get("truncated") or rec.get("winner") not in (0, 1):
        return 0.5
    return 1.0 if rec["winner"] == rec["s_seat"] else 0.0


def result_class(rec: dict) -> str:
    """S's result class: W (win), L (loss) or D (draw/truncated)."""
    if rec.get("truncated") or rec.get("winner") not in (0, 1):
        return "D"
    return "W" if rec["winner"] == rec["s_seat"] else "L"


def kind_name(k: int) -> str:
    try:
        return Kind(k).name
    except ValueError:
        return str(k)


def replay_trace(rec: dict, engine: Engine, names: CardNames, gauntlet: Path = GAUNTLET) -> dict:
    """Replay a record's choices; return per-decision trace + replay check.

    Each decision: {k, choice, turn, actor, msg, kind, kind_name, code, name, desc, nopts}.
    `ok` is True iff the replay ends with the recorded winner (and every recorded actor
    matches the env's player). Raises on engine errors; never on a mismatch (reports it).
    """
    cfg = duel_config(rec, gauntlet)
    env = CppEnv(engine, cfg, True)
    obs = env.reset()
    decisions = []
    actor_bad = 0
    nm = names.names
    for k, ch in enumerate(rec["choices"]):
        if obs is None:
            break
        actor = env.player
        if k < len(rec.get("actors", [])) and int(rec["actors"][k]) != actor:
            actor_bad += 1
        g = obs["global"]
        turn = int(g[2])
        msg = _MSG_NAMES.get(int(g[21]), str(int(g[21])))
        oo = _opts(obs)
        kind, code = oo[int(ch)] if 0 <= int(ch) < len(oo) else (None, 0)
        if kind is not None and int(kind) == Kind.NUMBER:
            name, desc = str(_number_arg(obs, int(ch))), ""
        elif code:
            name = nm.get(code, f"#{code}")
            desc = option_desc(obs, int(ch))
        else:
            name, desc = hidden_card(obs, int(ch)), option_desc(obs, int(ch))
        decisions.append(
            {
                "k": k, "choice": int(ch), "turn": turn, "actor": actor, "msg": msg,
                "kind": int(kind) if kind is not None else -1,
                "kind_name": kind_name(int(kind)) if kind is not None else "?",
                "code": int(code), "name": name, "desc": desc, "nopts": len(oo),
            }
        )
        obs = env.step(int(ch))
    full = len(decisions) == len(rec["choices"])
    if rec.get("truncated"):
        # cut by a decision cap: the duel is still going, so only require full consumption
        ok = full and actor_bad == 0
    else:
        ok = full and env.done and env.winner == rec.get("winner") and actor_bad == 0
    return {"decisions": decisions, "winner": env.winner, "done": env.done,
            "actor_bad": actor_bad, "ok": ok, "replayed": len(decisions)}


def _number_arg(obs, i: int) -> int:
    try:
        return int(list(obs["actions"])[i][5])
    except (IndexError, TypeError, KeyError):
        return 0


def header_of(rec: dict, trace_ok: bool = True) -> str:
    """One-line header: decks, seat, who went first, winner, turns."""
    n0, n1 = seat_decks(rec)
    s = rec["s_seat"]
    us = rec["decks"][s]
    first = f"P0 ({us + ', us' if s == 0 else n0})"
    w = rec.get("winner")
    wtxt = f"P{w} ({'us' if w == s else 'opp'}, {result_class(rec)})" if w in (0, 1) else result_class(rec)
    flag = "" if trace_ok else " [REPLAY MISMATCH]"
    return (
        f"game {rec.get('game_idx')}: us=P{s} {us} vs {rec['decks'][1 - s] if s in (0, 1) else '?'} "
        f"(seat0={n0} seat1={n1}) | seed {rec.get('seed')} | first {first} | "
        f"winner {wtxt} | turns {rec.get('turns')}{flag}"
    )


def render_record(rec: dict, engine: Engine, names: CardNames, gauntlet: Path = GAUNTLET) -> tuple[str, bool]:
    """Readable text of one game: header + turn-by-turn log, our decisions marked `>>`.

    Returns (text, ok) where ok is the replay check from replay_trace.
    """
    t = replay_trace(rec, engine, names, gauntlet)
    s = rec["s_seat"]
    lines = [header_of(rec, t["ok"])]
    cur = None
    for d in t["decisions"]:
        if d["turn"] != cur:
            cur = d["turn"]
            lines.append(f"== Turn {cur}")
        mark = ">>" if d["actor"] == s else "  "
        pick = f"{d['kind_name']} {d['name']}" + (f' "{d["desc"]}"' if d["desc"] else "")
        lines.append(f"{mark} [{d['k']}] T{d['turn']} P{d['actor']} {d['msg']} {pick} /{d['nopts']}")
    w = t["winner"]
    tail = f"DUEL OVER: winner {('P' + str(w)) if w in (0, 1) else w}" if t["done"] else "DUEL CUT (truncated, undecided)"
    lines.append(tail)
    return "\n".join(lines) + "\n", t["ok"]


def first_divergence(ta: dict, tb: dict) -> dict | None:
    """First decision where trace B differs from trace A (same deal).

    Returns {idx, turn, actor, kind_name, a_pick, b_pick} or None when the choice
    sequences are identical (then only the lengths may differ).
    """
    da, db = ta["decisions"], tb["decisions"]
    for d, e in zip(da, db):
        if d["choice"] != e["choice"] or d["kind"] != e["kind"] or d["nopts"] != e["nopts"]:
            return {
                "idx": d["k"], "turn": d["turn"], "actor": d["actor"], "kind_name": d["kind_name"],
                "a_pick": _pick_str(d), "b_pick": _pick_str(e),
            }
    if len(da) != len(db):
        d = (da if len(da) > len(db) else db)[min(len(da), len(db))]
        longer = "A" if len(da) > len(db) else "B"
        return {"idx": d["k"], "turn": d["turn"], "actor": d["actor"], "kind_name": d["kind_name"],
                "a_pick": _pick_str(d) if longer == "A" else "(ended)",
                "b_pick": _pick_str(d) if longer == "B" else "(ended)"}
    return None


def _pick_str(d: dict) -> str:
    return f"{d['kind_name']} {d['name']}" + (f' "{d["desc"]}"' if d["desc"] else "") + f" /{d['nopts']}"
