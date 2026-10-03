"""Full interactive games between two decks (ADR 0004: L2 going second, L3 interruption + replanning, L4 full games).

A normal duel (5-card hands, draws). At each player's open main-phase state on their own turn, the planner (an LLM
via `complete_fn`) gets a text description of the *current* state and writes the rest of the turn as a ComboLine; the
engine search verifies it from that state (`verify.plan_from_state`, passive opponent; one repair round with the
explain feedback). The verified choices are then executed in the real duel, where the opponent's decisions come from
the rule-based `Responder`. If the responder deviates from the passive answer the plan assumed (a hand trap, a
negate), the plan is broken and the turn player replans from the new state (once per turn). Decisions outside any plan
are also made by the responder.

Record (replayable like the demo records): config, choices, labels (stated / inferred from verified plans; auto /
responder / fallback otherwise; BC weights only stated+inferred by default), actors, turn per choice, plans per turn
(line + verdict), winner (None at the turn cap) and LP at the end.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable

from ygo_zero.decks import Deck
from ygo_zero.demos.explain import explain_failure
from ygo_zero.demos.extract import _fill_codes, _parse_json
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.plan import SCHEMA_TEXT, card_sheet, compact_line_json, repair_prompt
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import plan_from_state
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine

TURN_SYSTEM = (
    "You are an expert Yu-Gi-Oh! player. Given the exact current game state, write the strongest plan for the rest "
    "of your current turn as strict JSON. Use exact card names as written. Output names only. JSON only, no prose."
)
_OPEN_MSGS = (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)
_PHASES = {0x01: "Draw", 0x02: "Standby", 0x04: "Main 1", 0x08: "Battle", 0x10: "Battle", 0x20: "Battle",
           0x40: "Battle", 0x80: "Battle", 0x100: "Main 2", 0x200: "End"}  # fmt: skip
_POS = {1: "face-up ATK", 2: "face-down ATK", 3: "face-up DEF", 4: "face-down DEF", 5: "face-up", 6: "set"}


def is_open(obs) -> bool:
    g = obs["global"]
    return int(g[21]) in _OPEN_MSGS and int(g[20]) == 0


def describe_state(obs, names: dict[int, str]) -> str:
    """The acting player's view as text (controller columns are relative to the player to act)."""
    g = obs["global"]
    rows = obs["cards"][obs["card_mask"]]

    def nm(code: int) -> str:
        return names.get(int(code), f"#{int(code)}") if int(code) else "(hidden)"

    def pick(ctrl: int, loc: int) -> list:
        return [r for r in rows if int(r[3]) == ctrl and int(r[1]) == loc]

    def mons(ctrl: int) -> str:
        out = []
        for r in sorted(pick(ctrl, 3), key=lambda r: int(r[2])):
            stats = f"{int(r[14])}/{int(r[15])}" if int(r[0]) else "?"
            out.append(f"{nm(r[0])} [{_POS.get(int(r[5]), '?')}, {stats}, zone {int(r[2]) - 1}]")
        return "; ".join(out) or "none"

    def sts(ctrl: int) -> str:
        out = [
            f"{nm(r[0])} [{_POS.get(int(r[5]), '?')}]" for r in sorted(pick(ctrl, 4), key=lambda r: int(r[2]))
        ]
        return "; ".join(out) or "none"

    def names_at(ctrl: int, loc: int) -> str:
        return ", ".join(nm(r[0]) for r in pick(ctrl, loc)) or "none"

    phase = _PHASES.get(int(g[3]), str(int(g[3])))
    return "\n".join([
        (
            f"Turn {int(g[2])} ({'your' if int(g[5]) else 'opponent'} turn), phase {phase}. "
            f"LP: you {int(g[0])}, opponent {int(g[1])}. You went {'first' if int(g[4]) else 'second'}."
        ),
        f"Your hand: {names_at(0, 2)}",
        f"Your monsters: {mons(0)}",
        f"Your spells/traps: {sts(0)}",
        f"Your GY: {names_at(0, 5)}",
        f"Your banished: {names_at(0, 6)}",
        f"Your Extra Deck: {names_at(0, 7)}",
        f"Your Deck: {int(g[6])} cards",
        f"Opponent monsters: {mons(1)}",
        f"Opponent spells/traps: {sts(1)}",
        f"Opponent GY: {names_at(1, 5)}",
        f"Opponent banished: {names_at(1, 6)}",
        f"Opponent hand: {int(g[14])} cards; Deck: {int(g[13])} cards",
    ])  # fmt: skip


def turn_prompt(deck_name: str, sheet: str, state: str, can_attack: bool, examples: list[ComboLine]) -> str:
    ex = ""
    if examples:
        ex = "\nEXAMPLES of verified lines for this deck (names only):\n" + "\n\n".join(
            compact_line_json(e) for e in examples[:2]
        )
    attack_rule = (
        "- After Main Phase 1 you may attack: add `attack` steps (card = your attacker, targets = [the opponent "
        "monster] or [] for a direct attack), then optional Main Phase 2 steps."
        if can_attack
        else "- You cannot attack this turn (first turn of the duel)."
    )
    return f"""You are playing {deck_name}.

YOUR DECKLIST (card sheet):
{sheet}
{ex}
CURRENT STATE (exact):
{state}

Plan the rest of YOUR current turn from this exact state as JSON: {{"lines": [ComboLine]}}
{SCHEMA_TEXT}

Rules:
- `hand` = your current hand above. Only cards you have (hand, field, GY, banished, Extra Deck) or can reach through a
  previous step's search/summon. Normal Summon at most once (if not already used); respect once-per-turn limits.
- Deal with the opponent's board if it threatens you (removal, negation, out-valuing), then build the strongest position.
{attack_rule}
- `going` = "first" if you went first in the duel, else "second". `end_board` = your field at the end of the turn.
- If there is nothing useful to do, return a short line (e.g. set a card) rather than an illegal flashy one.
JSON only, no prose."""


class Responder:
    """Rule-based decisions outside plans: activate hand traps / quick effects at the opponent's chain windows with
    probability `p_trap`, answer optional triggers YES with `p_yes`, attack when possible, otherwise the first
    non-cancel option. Seeded, so a game is reproducible."""

    def __init__(self, seed: int, p_trap: float = 0.7, p_yes: float = 0.8):
        self.rng = random.Random(seed)
        self.p_trap = p_trap
        self.p_yes = p_yes

    def passive(self, obs) -> int:
        for i, (k, _) in enumerate(_opts(obs)):
            if k in (Kind.PASS, Kind.NO, Kind.FINISH, Kind.CANCEL, Kind.TO_END):
                return i
        return 0

    def __call__(self, obs) -> int:
        opts = _opts(obs)
        kinds = [k for k, _ in opts]
        msg, my_turn = int(obs["global"][21]), bool(int(obs["global"][5]))
        if msg == C.MSG_SELECT_CHAIN and not my_turn:
            acts = [i for i, (k, _) in enumerate(opts) if k == Kind.ACTIVATE]
            if acts and self.rng.random() < self.p_trap:
                return self.rng.choice(acts)
            return self.passive(obs)
        if Kind.YES in kinds and Kind.NO in kinds:
            return kinds.index(Kind.YES) if self.rng.random() < self.p_yes else kinds.index(Kind.NO)
        if msg == C.MSG_SELECT_BATTLECMD:
            atk = [i for i, k in enumerate(kinds) if k == Kind.ATTACK]
            if atk:
                return self.rng.choice(atk)
        if (
            msg == C.MSG_SELECT_IDLECMD
        ):  # an unplanned main phase: go to battle if possible, else end the turn
            for want in (Kind.TO_BATTLE, Kind.TO_END):
                if want in kinds:
                    return kinds.index(want)
        cand = [i for i, k in enumerate(kinds) if k != Kind.CANCEL]
        return cand[0] if cand else 0


def _opts(obs) -> list[tuple[int, int]]:
    m = int(obs["action_mask"].sum())
    return [(int(r[0]), int(r[2])) for r in obs["actions"][:m]]


def play_game(
    engine: Engine,
    decks: tuple[tuple[str, Deck], tuple[str, Deck]],
    seed: int,
    complete_fn: Callable[[str], str],
    names: CardNames,
    max_turns: int = 6,
    budget: int = 4000,
    repairs: int = 1,
    examples: dict[str, list[ComboLine]] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Play one game; deck 0 goes first. Returns the record (see module docstring)."""
    (name0, deck0), (name1, deck1) = decks
    config = DuelConfig(deck0, deck1, seed=seed)
    sheets = {0: card_sheet(engine, deck0, names), 1: card_sheet(engine, deck1, names)}
    dnames = {0: name0, 1: name1}
    pdecks = {0: deck0, 1: deck1}
    responder = Responder(seed)
    env = CppEnv(engine, config)
    obs = env.reset()
    choices: list[int] = []
    labels: list[str] = []
    actors: list[int] = []
    turns: list[int] = []
    plans: list[dict] = []
    llm_calls = 0
    planned: list[tuple[int, str]] = []  # remaining (choice, label) of the active plan
    plan_owner, plan_turn = -1, -1
    replanned: set[tuple[int, int]] = set()  # (turn, player) that already replanned
    tried: set[tuple[int, int]] = set()  # (turn, player) that already got a plan attempt
    t0 = time.time()

    def take(i: int, lab: str) -> None:
        nonlocal obs
        actors.append(env.player)
        turns.append(int(obs["global"][2]))
        choices.append(i)
        labels.append(lab)
        obs = env.step(i)

    def make_plan(p: int, turn: int, why: str) -> list[tuple[int, str]]:
        nonlocal llm_calls
        state = describe_state(obs, names.names)
        can_attack = turn > 1
        base = turn_prompt(dnames[p], sheets[p], state, can_attack, (examples or {}).get(dnames[p], []))
        prompt = base
        entry = {"turn": turn, "player": p, "deck": dnames[p], "why": why, "attempts": []}
        for attempt in range(repairs + 1):
            llm_calls += 1
            try:
                resp = complete_fn(prompt)
            except Exception as e:  # noqa: BLE001 - an LLM failure ends the attempt
                entry["attempts"].append({"reason": f"llm_error: {e}"[:200]})
                break
            dicts = _parse_json(resp)
            if not dicts:
                entry["attempts"].append({"reason": "unparsable"})
                prompt = (
                    base + "\n\nYour previous answer was not valid JSON with a 'lines' list. Answer again."
                )
                continue
            try:
                line = ComboLine.from_json(dicts[0])
            except Exception as e:  # noqa: BLE001
                entry["attempts"].append({"reason": f"malformed: {e}"[:200]})
                continue
            line.id = f"game{seed}/t{turn}p{p}a{attempt}"
            line.deck = dnames[p]
            line.going = "first" if bool(int(obs["global"][4])) else "second"
            line.source_file = "game"
            # "pass" steps name no card and do nothing; an all-pass plan ends the turn via the fallback
            line.steps = [st for st in line.steps if st.card or st.action not in ("other", "pass")]
            if not line.steps:
                entry["attempts"].append({"line": dicts[0], "reason": "pass", "nodes": 0})
                plans.append(entry)
                log(f"  turn {turn} p{p} ({dnames[p]}): plan = pass")
                return []
            for k, st in enumerate(line.steps):
                st.i = k
            _fill_codes(line, names)
            v = plan_from_state(engine, config, choices, line, pdecks[p], budget=budget)
            entry["attempts"].append(
                {"line": json.loads(compact_line_json(line)), "reason": v.reason, "nodes": v.nodes}
            )
            if v.ok:
                plans.append(entry)
                log(f"  turn {turn} p{p} ({dnames[p]}): plan ok ({len(line.steps)} steps, {v.nodes} nodes)")
                return list(zip(v.choices[len(choices) :], v.labels[len(choices) :], strict=True))
            try:
                ex = explain_failure(engine, line, pdecks[p], v, names.names)
            except Exception:  # noqa: BLE001
                ex = ""
            prompt = repair_prompt(base, compact_line_json(line), f"{v.reason}: {v.detail}", ex)
        plans.append(entry)
        log(
            f"  turn {turn} p{p} ({dnames[p]}): no verified plan ({[a.get('reason') for a in entry['attempts']]})"
        )
        return []

    while obs is not None:
        turn = int(obs["global"][2])
        if turn > max_turns:
            break
        p = env.player
        my_turn = bool(int(obs["global"][5]))
        if planned and plan_turn != turn:
            planned = []  # the plan's turn is over (verified plans run on through the passive opponent's turn)
        if planned:
            want, lab = planned[0]
            if p == plan_owner:
                take(want, lab)
                planned.pop(0)
                continue
            # an opponent decision inside the plan: the plan assumed the passive answer
            r = responder(obs)
            if r == want:
                take(r, "auto")
                planned.pop(0)
            else:
                take(r, "responder")
                planned = []  # interrupted: the turn player replans at its next open state
                log(f"  turn {turn}: p{p} interrupts p{plan_owner}'s plan")
            continue
        if my_turn and is_open(obs) and int(obs["global"][21]) == C.MSG_SELECT_IDLECMD:
            key = (turn, p)
            # first plan of this turn, or one replan (after an interruption or a failed plan, e.g. at Main Phase 2)
            if key not in tried or (key not in replanned and plan_owner == p and plan_turn == turn):
                why = "replan" if key in tried else "turn"
                if key in tried:
                    replanned.add(key)
                tried.add(key)
                planned = make_plan(p, turn, why)
                plan_owner, plan_turn = p, turn
                if planned:
                    continue
        take(responder(obs), "responder" if p != plan_owner or turn != plan_turn else "fallback")
    winner = env.winner if obs is None else None
    g = obs["global"] if obs is not None else None
    return {
        "game_id": f"game{seed}",
        "line_id": f"{name0}-vs-{name1}/game{seed}",
        "level": "L4",
        "decks": [name0, name1],
        "config": {
            "deck0_main": list(deck0.main), "deck0_extra": list(deck0.extra),
            "deck1_main": list(deck1.main), "deck1_extra": list(deck1.extra),
            "seed": seed, "hand": config.hand, "draw": config.draw, "setup": [],
        },
        "auto_single": True,
        "choices": choices,
        "labels": labels,
        "actors": actors,
        "turns": turns,
        "plans": plans,
        "winner": winner,
        "end_turn": turns[-1] if turns else 0,
        # obs global[0:2] = (LP of the player to act, LP of the other): stored by absolute player
        "lp": None if g is None else ([int(g[0]), int(g[1])] if env.player == 0 else [int(g[1]), int(g[0])]),
        "llm_calls": llm_calls,
        "seconds": round(time.time() - t0, 1),
    }  # fmt: skip
