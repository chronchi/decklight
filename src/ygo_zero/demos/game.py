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
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ygo_zero.decks import Deck
from ygo_zero.demos.dataset import record_config
from ygo_zero.demos.explain import explain_failure
from ygo_zero.demos.extract import _fill_codes, _parse_json
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.plan import SCHEMA_TEXT, card_sheet, compact_line_json, repair_prompt
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.strategy import notes_block
from ygo_zero.demos.verify import plan_from_state
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine, Placement
from ygo_zero.engine.resources import default_cdbs

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


def _vs_set_board(state: str) -> bool:
    """Whether the acting player went second and the opponent already holds a board."""
    if "went second" not in state.lower():
        return False
    for line in state.splitlines():
        s = line.strip().lower()
        if s.startswith(("opponent monsters:", "opponent spells/traps:")) and s.split(":", 1)[1].strip() not in (
            "", "none",
        ):
            return True
    return False


_GO_SECOND_HINT = (
    "Plan how to break or play through the opponent's board (board breakers, removal, "
    "hand traps you already used), then build or attack for lethal if possible."
)


_PUSH = (
    "\n\nYour previous plan only {what} although you hold {n} cards in hand. A strong player uses their resources: "
    "Normal/Special Summons, searches, Extra Deck plays, removal or hand traps against the opponent's board. Plan the "
    "strongest legal line from this exact state; only if truly nothing works, keep the short line."
)


def hand_size(state: str) -> int:
    for line in state.splitlines():
        if line.strip().lower().startswith("your hand:"):
            rest = line.split(":", 1)[1].strip()
            return 0 if rest.lower() in ("", "none") else len([x for x in rest.split(", ") if x.strip()])
    return 0


def weak_plan(steps) -> str | None:
    """'passed' / 'set a card' when a plan does no more than that, else None (2026-10-03: 141 of ~700 second-player
    turn-2 plans were a pass or one Set; the prompt's "set a card if nothing useful" was taken too readily)."""
    if not steps:
        return "passed"
    if len(steps) == 1 and steps[0].action in ("set_spelltrap", "set_monster"):
        return "set a card"
    return None


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
    second = f"- {_GO_SECOND_HINT}\n" if _vs_set_board(state) else ""
    return f"""You are playing {deck_name}.
{notes_block(deck_name)}
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
{second}{attack_rule}
- `going` = "first" if you went first in the duel, else "second". `end_board` = your field at the end of the turn.
- If there is nothing useful to do, return a short line (e.g. set a card) rather than an illegal flashy one.
JSON only, no prose."""


class Responder:
    """Rule-based decisions outside plans: activate hand traps / quick effects at the opponent's chain windows with
    probability `p_trap`, answer optional triggers YES with `p_yes`, attack when possible, otherwise the first
    non-cancel option. Seeded, so a game is reproducible."""

    def __init__(self, seed: int, p_trap: float = 0.25, p_yes: float = 0.8):
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
        if msg == C.MSG_SELECT_CHAIN:
            if not my_turn:
                acts = [i for i, (k, _) in enumerate(opts) if k == Kind.ACTIVATE]
                if acts and self.rng.random() < self.p_trap:
                    return self.rng.choice(acts)
            # own-turn chain windows outside a plan: pass (the old fall-through picked the first option, which fired
            # hand traps like Ash/Fuwalos on the player's own turn; the judge flagged it, 2026-10-03)
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
    m = obs["action_mask"]
    try:
        n = int(m.sum())
    except AttributeError:
        n = sum(1 for x in m if x)
    return [(int(r[0]), int(r[2])) for r in list(obs["actions"])[:n]]


def _passive_index(obs) -> int:
    for i, (k, _) in enumerate(_opts(obs)):
        if k in (Kind.PASS, Kind.NO, Kind.FINISH, Kind.CANCEL, Kind.TO_END):
            return i
    return 0


def _masked_rows(obs, arr_key: str, mask_key: str) -> list:
    return [r for r, m in zip(list(obs[arr_key]), list(obs[mask_key])) if m]


_INTERRUPT_MSGS = (C.MSG_SELECT_CHAIN, C.MSG_SELECT_EFFECTYN, C.MSG_SELECT_YESNO)

_DESC_CACHE: dict[int, str] | None = None


def _effect_text(code: int) -> str:
    """First 300 chars of the card's effect text (whitespace-collapsed)."""
    global _DESC_CACHE
    if _DESC_CACHE is None:
        out: dict[int, str] = {}
        for path in default_cdbs():
            try:
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            except sqlite3.Error:
                continue
            try:
                for c, desc in con.execute("SELECT id, desc FROM texts").fetchall():
                    out.setdefault(int(c), str(desc or ""))
            finally:
                con.close()
        _DESC_CACHE = out
    raw = _DESC_CACHE.get(int(code), "")
    return re.sub(r"\s+", " ", raw).strip()[:300]


def chain_names(obs, names: dict[int, str]) -> list[str]:
    out = []
    for r in _masked_rows(obs, "chain", "chain_mask"):
        code = int(r[1])
        out.append(names.get(code, f"#{code}") if code else "(hidden)")
    return out


def options_text(obs, names: dict[int, str], engine=None) -> str:
    lines = []
    for i, (k, code) in enumerate(_opts(obs)):
        try:
            kind = Kind(k).name
        except ValueError:
            kind = str(k)
        nm = names.get(int(code), f"#{int(code)}") if int(code) else "(no card)"
        desc = _effect_text(code) if engine is not None and int(code) else ""
        lines.append(f"{i}: {kind} {nm}" + (f" | {desc}" if desc else ""))
    return "\n".join(lines)


def interrupt_prompt(deck_name: str, state: str, chain: list[str], recent: list[str], options: str) -> str:
    return f"""You are playing {deck_name} (responding on the opponent's turn).
{notes_block(deck_name)}
CURRENT STATE:
{state}

CHAIN: {", ".join(chain) if chain else "empty"}
OPPONENT'S LAST ACTIONS: {", ".join(recent) if recent else "none yet"}

YOUR LEGAL OPTIONS (reply with the option number):
{options}

You are an expert player. Hand traps and quick effects are limited; use one only if this is a good moment (it stops a key search/summon/combo piece or prevents lethal). Otherwise pass.
Reply with JSON only: {{"choice": <option number or -1 for pass>, "why": "..."}}."""


def parse_interrupt_answer(text: str, n_options: int, passive: int) -> tuple[int, str]:
    """(option index, why); anything invalid means pass (the passive index)."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            return passive, "unparsable"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return passive, "unparsable"
    if not isinstance(obj, dict):
        return passive, "unparsable"
    why = str(obj.get("why", ""))[:500]
    try:
        choice = int(obj.get("choice", -1))
    except (TypeError, ValueError):
        return passive, why or "bad-choice"
    if choice == -1:
        return passive, why
    if 0 <= choice < n_options:
        return choice, why
    return passive, why or "bad-choice"


_TRIGGER_KINDS = (Kind.SUMMON, Kind.SPSUMMON, Kind.ACTIVATE)


class Interrupter:
    """Deliberate opponent interruptions: ask the LLM whether a chain/quick-effect window on the
    opponent's turn is worth answering, instead of firing hand traps at random.

    Asks at most `max_per_turn` windows per turn; anything else (exhausted asks, own-turn windows)
    stays with the rule-based `Responder`. Invalid LLM answers mean pass."""

    def __init__(self, complete_fn: Callable[[str], str], names, max_per_turn: int = 3):
        self.complete_fn = complete_fn
        self.names = names.names if hasattr(names, "names") else dict(names)
        self.max_per_turn = max_per_turn
        self.used: dict[int, int] = {}

    def wants(self, obs) -> bool:
        """Whether this window qualifies: acting player is NOT the turn player, the message is a
        chain (or an effect-yes/no quick-effect window), and at least one option activates."""
        if bool(int(obs["global"][5])):
            return False
        if int(obs["global"][21]) not in _INTERRUPT_MSGS:
            return False
        return any(k == Kind.ACTIVATE for k, _ in _opts(obs))

    def can_ask(self, turn: int) -> bool:
        return self.used.get(int(turn), 0) < self.max_per_turn

    def __call__(self, obs, deck_name: str, turn: int, player: int, recent: list[str],
                 engine=None) -> tuple[int, str]:
        """Ask the LLM and return (option index, why). Counts one ask for this turn; any failure
        (unparsable answer, LLM error, no qualifying window, exhausted asks) returns the passive index."""
        passive = _passive_index(obs)
        if not self.wants(obs) or not self.can_ask(int(turn)):
            return passive, "no-ask"
        self.used[int(turn)] = self.used.get(int(turn), 0) + 1
        prompt = interrupt_prompt(
            deck_name,
            describe_state(obs, self.names),
            chain_names(obs, self.names),
            recent,
            options_text(obs, self.names, engine),
        )
        try:
            resp = self.complete_fn(prompt)
        except Exception:  # noqa: BLE001 - an LLM failure passes
            return passive, "llm_error"
        return parse_interrupt_answer(resp, len(_opts(obs)), passive)


_OPENING_PATTERNS = (
    "verified/{a}*.jsonl",
    "*/verified/{a}*.jsonl",
    "planned*/verified/{a}*.jsonl",
    "planned*/improved/verified/{a}*.jsonl",
)


def rank_openings(recs: list[dict], deck_a: str) -> list[dict]:
    """Verified turn-1 line records of deck A, biggest end board first (deduped by line_id)."""
    out: dict[str, dict] = {}
    for rec in recs:
        lid = str(rec.get("line_id", ""))
        if not lid.startswith(deck_a + "/") or lid in out:
            continue
        if not rec.get("choices") or len(rec.get("labels", [])) != len(rec["choices"]):
            continue
        out[lid] = rec
    return sorted(
        out.values(),
        key=lambda r: (-len(r.get("end_board") or []), -len(r["choices"]), str(r["line_id"])),
    )


def find_openings(deck_a: str, base: Path) -> list[dict]:
    """Ranked opening candidates for deck A among the verified records under `base`."""
    recs = []
    for pat in _OPENING_PATTERNS:
        for f in sorted(base.glob(pat.format(a=deck_a))):
            if not f.is_file():
                continue
            try:
                lines = f.read_text().splitlines()
            except OSError:
                continue
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    recs.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    return rank_openings(recs, deck_a)


def opening_valid(engine: Engine, config: DuelConfig, choices: list[int]) -> bool:
    """Whether a turn-1 line prefix replays cleanly under `config` (no early duel end, every index legal)."""
    env = CppEnv(engine, config)
    obs = env.reset()
    for i in choices:
        if obs is None:
            return False
        if not 0 <= int(i) < len(_opts(obs)):
            return False
        obs = env.step(int(i))
    return True


def _actors(engine: Engine, config: DuelConfig, choices: list[int]) -> list[int] | None:
    """The acting player of each recorded choice under `config`, or None when the choices do not replay."""
    env = CppEnv(engine, config)
    obs, out = env.reset(), []
    for i in choices:
        if obs is None or not 0 <= int(i) < len(_opts(obs)):
            return None
        out.append(env.player)
        obs = env.step(int(i))
    return out


OPENING_HAND = 5


def build_opening(engine: Engine, rec: dict, deck_b: Deck, seed: int = 0
                  ) -> tuple[DuelConfig, list[int], list[str]] | None:  # fmt: skip
    """An opening for `play_game` from a verified turn-1 line record: the record's config with deck1 replaced by the
    opponent deck `deck_b`, which now gets a real opening hand (OPENING_HAND random cards placed in hand and removed
    from its deck, like player 0's setup; line records start the opponent with 0 cards, so turn-1 hand traps were
    impossible, NOTEBOOK 2026-10-03 21:20). The choice sequence is re-derived for the new config: player 0's recorded
    choices in order, player 1 passive at every window (its hand traps now open windows the record never had; in the
    game those windows go through the interrupter). None when the line no longer replays."""
    if len(rec.get("labels", [])) != len(rec.get("choices", [])):
        return None
    try:
        orig = record_config(rec)
    except (KeyError, TypeError, ValueError):
        return None
    choices = [int(i) for i in rec["choices"]]
    labels = [str(x) for x in rec["labels"]]
    try:
        actors = _actors(engine, orig, choices)
    except Exception:  # noqa: BLE001 - an unreadable engine state skips the record
        actors = None
    if actors is None:
        return None
    mine = [(c, lab) for c, lab, a in zip(choices, labels, actors, strict=True) if a == 0]
    rng = random.Random(seed)
    main = list(deck_b.main)
    hand = [main.pop(rng.randrange(len(main))) for _ in range(min(OPENING_HAND, len(main)))]
    config = replace(orig, deck1=Deck(tuple(main), tuple(deck_b.extra), (), deck_b.name),
                     setup=tuple(orig.setup) + tuple(Placement(1, c, C.LOCATION_HAND) for c in hand))  # fmt: skip
    try:
        env = CppEnv(engine, config)
        obs = env.reset()
        out_c, out_l, k = [], [], 0
        while k < len(mine):
            if obs is None or int(obs["global"][2]) > 1:
                return None  # the line no longer completes on turn 1
            if env.player == 0:
                c, lab = mine[k]
                if not 0 <= c < len(_opts(obs)):
                    return None
                k += 1
            else:
                c, lab = _passive_index(obs), "auto"
            out_c.append(c)
            out_l.append(lab)
            obs = env.step(c)
    except Exception:  # noqa: BLE001
        return None
    return config, out_c, out_l


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
    max_seconds: float | None = None,
    plan_fn: Callable | None = None,
    interrupts: int = 3,
    interrupt_fn: Callable[[str], str] | None = None,  # LLM for interruption asks (default: complete_fn)
    opening: tuple[DuelConfig, list[int], list[str]] | None = None,
) -> dict:
    """Play one game; deck 0 goes first. Returns the record (see module docstring). `max_seconds`: stop at the next
    turn boundary once exceeded (the record is kept, winner None), so a slow game never loses its data to a kill.
    `opening`: (config, choices, labels) of a verified turn-1 line of deck 0, replayed blindly (actor 0, labels
    copied) instead of planning turn 1; the game then continues normally from turn 1's end, i.e. the opponent
    plans turn 2 against the established board. The record carries "opening" (None here; the driver fills in the
    source record's line_id). A prefix that no longer replays raises ValueError (the driver skips the record)."""
    (name0, deck0), (name1, deck1) = decks
    if opening is not None:
        oconfig, ochoices, olabels = opening
        config = oconfig
        pdecks = {0: config.deck0, 1: config.deck1}  # the decks actually in play (deck1 replaced by the driver)
        sheets = {0: card_sheet(engine, config.deck0, names), 1: card_sheet(engine, config.deck1, names)}
    else:
        config = DuelConfig(deck0, deck1, seed=seed)
        pdecks = {0: deck0, 1: deck1}
        sheets = {0: card_sheet(engine, deck0, names), 1: card_sheet(engine, deck1, names)}
    dnames = {0: name0, 1: name1}
    verify = plan_fn or plan_from_state
    responder = Responder(seed)
    interrupter = (Interrupter(interrupt_fn or complete_fn, names, max_per_turn=interrupts) if interrupts > 0 else None)
    env = CppEnv(engine, config)
    obs = env.reset()
    choices: list[int] = []
    labels: list[str] = []
    actors: list[int] = []
    turns: list[int] = []
    plans: list[dict] = []
    interrupts_log: list[dict] = []
    acted: list[tuple[int, int]] = []  # (actor, card code) of every taken decision, for recent-action context
    llm_calls = 0
    planned: list[tuple[int, str]] = []  # remaining (choice, label) of the active plan
    plan_owner, plan_turn = -1, -1
    replanned: set[tuple[int, int]] = set()  # (turn, player) that already replanned
    tried: set[tuple[int, int]] = set()  # (turn, player) that already got a plan attempt
    t0 = time.time()
    moves = [0, 0, -1]  # [turn-player summons/activations so far, value at the last ask / turn start, turn]

    def take(i: int, lab: str) -> None:
        nonlocal obs
        actors.append(env.player)
        turns.append(int(obs["global"][2]))
        try:
            code = int(obs["actions"][i][2])
        except (IndexError, TypeError):
            code = 0
        acted.append((env.player, code))
        if bool(int(obs["global"][5])) and int(obs["actions"][i][0]) in _TRIGGER_KINDS:
            moves[0] += 1  # the turn player summoned or activated something worth answering
        choices.append(i)
        labels.append(lab)
        obs = env.step(i)

    def decide(o, turn: int, p: int) -> tuple[int, bool]:
        """The opponent's decision: deliberate LLM interruption when the window qualifies and asks
        remain, else the rule-based responder. Returns (option index, is_interrupt)."""
        nonlocal llm_calls
        if moves[2] != turn:  # a new turn: earlier turns' actions are not something to respond to now
            moves[1], moves[2] = moves[0], turn
        fresh = bool(o["chain_mask"].any()) or moves[0] > moves[1]  # something to respond to (not Draw/Standby)
        if interrupter is not None and fresh and interrupter.wants(o) and interrupter.can_ask(turn):
            moves[1] = moves[0]
            recent = []
            for actor, code in reversed(acted):
                if actor != p:
                    recent.append(names.names.get(code, f"#{code}") if code else "(hidden)")
                    if len(recent) >= 3:
                        break
            idx, why = interrupter(o, dnames[p], turn, p, recent, engine)
            llm_calls += 1
            try:
                code = int(o["actions"][idx][2])
            except (IndexError, TypeError):
                code = 0
            card = names.names.get(code, f"#{code}") if code else "-"
            interrupts_log.append({"turn": turn, "player": p, "card": card, "why": why})
            return idx, True
        return responder(o), False

    def make_plan(p: int, turn: int, why: str) -> list[tuple[int, str]]:
        nonlocal llm_calls
        state = describe_state(obs, names.names)
        can_attack = turn > 1
        base = turn_prompt(dnames[p], sheets[p], state, can_attack, (examples or {}).get(dnames[p], []))
        prompt = base
        entry = {"turn": turn, "player": p, "deck": dnames[p], "why": why, "attempts": []}
        n_hand = hand_size(state)
        pushed = False
        fallback: list[tuple[int, str]] | None = None  # a verified weak plan, kept while a pushed retry is tried
        weak_at = -1
        attempt = -1
        while attempt < repairs + (1 if pushed else 0):
            attempt += 1
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
            weak = weak_plan(line.steps)
            if weak and n_hand >= 4 and not pushed:
                pushed = True
                if not line.steps:
                    entry["attempts"].append({"line": dicts[0], "reason": "pass_pushed", "nodes": 0})
                    prompt = base + _PUSH.format(what=weak, n=n_hand)
                    continue
            if not line.steps:
                if fallback is not None:
                    break
                entry["attempts"].append({"line": dicts[0], "reason": "pass", "nodes": 0})
                plans.append(entry)
                log(f"  turn {turn} p{p} ({dnames[p]}): plan = pass")
                return []
            for k, st in enumerate(line.steps):
                st.i = k
            _fill_codes(line, names)
            v = verify(engine, config, choices, line, pdecks[p], budget=budget)
            entry["attempts"].append(
                {"line": json.loads(compact_line_json(line)), "reason": v.reason, "nodes": v.nodes}
            )
            if v.ok:
                got = list(zip(v.choices[len(choices) :], v.labels[len(choices) :], strict=True))
                if weak and pushed and fallback is None and n_hand >= 4:
                    fallback = got  # verified but weak: ask once for a stronger line, keep this one if that fails
                    weak_at = len(entry["attempts"]) - 1
                    entry["attempts"][weak_at]["reason"] = "ok_weak_pushed"
                    prompt = base + _PUSH.format(what=weak, n=n_hand)
                    continue
                plans.append(entry)
                log(f"  turn {turn} p{p} ({dnames[p]}): plan ok ({len(line.steps)} steps, {v.nodes} nodes)"
                    + (" after push" if pushed else ""))
                return got
            try:
                ex = explain_failure(engine, line, pdecks[p], v, names.names)
            except Exception:  # noqa: BLE001
                ex = ""
            prompt = repair_prompt(base, compact_line_json(line), f"{v.reason}: {v.detail}", ex)
        if fallback is not None:
            entry["attempts"][weak_at]["reason"] = "ok"  # the line actually played (judge_games reads reason == "ok")
            plans.append(entry)
            log(f"  turn {turn} p{p} ({dnames[p]}): push failed, keeping the weak plan")
            return fallback
        plans.append(entry)
        log(
            f"  turn {turn} p{p} ({dnames[p]}): no verified plan ({[a.get('reason') for a in entry['attempts']]})"
        )
        return []

    if opening is not None:
        # the verified turn-1 line becomes player 0's turn-1 PLAN (not a blind prefix): the second player's chain windows
        # inside it go through `decide` (deliberate LLM interruptions / responder), so hand traps on turn 1 happen like in
        # any game; an interruption breaks the line and player 0 replans from the interrupted state.
        planned = [(int(i), str(lab)) for i, lab in zip(ochoices, olabels, strict=True)]
        plan_owner, plan_turn = 0, 1
        tried.add((1, 0))  # no fresh LLM plan at the start of turn 1: the opening line is the plan

    while obs is not None:
        turn = int(obs["global"][2])
        if turn > max_turns:
            break
        if max_seconds is not None and time.time() - t0 > max_seconds and turns and turn != turns[-1]:
            log(f"  wall-time cap {max_seconds:.0f}s reached at turn {turn}")
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
            r, is_int = decide(obs, turn, p)
            if r == want:
                take(r, "interrupt" if is_int else "auto")
                planned.pop(0)
            else:
                take(r, "interrupt" if is_int else "responder")
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
        r, is_int = decide(obs, turn, p)
        take(r, "interrupt" if is_int else ("responder" if p != plan_owner or turn != plan_turn else "fallback"))
    winner = env.winner if obs is None else None
    g = obs["global"] if obs is not None else None
    return {
        "game_id": f"game{seed}",
        "line_id": f"{name0}-vs-{name1}/game{seed}",
        "level": "L4",
        "decks": [name0, name1],
        # the config actually played (an opening replays a line record's config: its seed, deck0 and setup)
        "config": {
            "deck0_main": list(config.deck0.main), "deck0_extra": list(config.deck0.extra),
            "deck1_main": list(config.deck1.main), "deck1_extra": list(config.deck1.extra),
            "seed": config.seed, "hand": config.hand, "draw": config.draw, "lp": config.lp,
            "setup": [[pl.team, pl.code, pl.location, pl.sequence] for pl in config.setup],
        },
        "auto_single": True,
        "choices": choices,
        "labels": labels,
        "actors": actors,
        "turns": turns,
        "plans": plans,
        "interrupts": interrupts_log,
        "opening": None,  # the driver fills in the source record's line_id when an opening was replayed
        "winner": winner,
        "end_turn": turns[-1] if turns else 0,
        # obs global[0:2] = (LP of the player to act, LP of the other): stored by absolute player
        "lp": None if g is None else ([int(g[0]), int(g[1])] if env.player == 0 else [int(g[1]), int(g[0])]),
        "llm_calls": llm_calls,
        "seconds": round(time.time() - t0, 1),
    }  # fmt: skip
