"""Ground + verify extracted combo lines in the engine (ADR 0002 stage 3).

A line is kept only if it executes from a puzzle start and reaches its claimed
end board. Search is depth-first over env decisions with backtracking through
replay (``CppEnv``); options the guide states come first, the rest follow in
index order. Anything ambiguous is rejected: this is the only gate between
internet text and training data.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ygo_zero.decks import Deck
from ygo_zero.demos.generic import resolve_generic
from ygo_zero.demos.schema import ComboLine
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine, Placement

# obs columns (docs/obs_spec.md, docs/action_spec.md)
_LOC_MZONE, _LOC_SZONE = 3, 4
_MOVERS = (Kind.TO_BATTLE, Kind.TO_MAIN2, Kind.TO_END)
_DEMOTED = frozenset({Kind.TO_BATTLE, Kind.TO_MAIN2, Kind.TO_END, Kind.CANCEL})
_COMMAND_MSGS = (C.MSG_SELECT_IDLECMD, C.MSG_SELECT_BATTLECMD)  # step boundaries (with an empty chain)
_PASSIVE = (Kind.PASS, Kind.NO, Kind.TO_END, Kind.FINISH, Kind.CANCEL)
VERIFIED_DIR = Path(__file__).resolve().parents[3] / "data" / "demos" / "verified"
# Unstated main-phase actions allowed between steps, per line (each must use a card the line mentions later).
MAX_EXTRA_ACTIONS = 1
# Steps whose results must end up on my field (a "summoned" card that is still in the GY was not summoned).
_ON_FIELD = frozenset({
    "normal_summon", "set_monster", "special_summon", "link_summon", "synchro_summon", "xyz_summon",
    "fusion_summon", "ritual_summon", "set_spelltrap",
})  # fmt: skip

POST_CAP = 300  # extra decisions after the last step before giving up


@dataclass
class Verdict:
    line_id: str
    ok: bool
    reason: str  # ok | missing_card | unresolved | out_of_scope | no_anchor@<k> | budget | end_board_mismatch
    #            | vacuous | too_skipped | trivial
    choices: list[int]  # every index passed to env.step (replays to the final state)
    labels: list[str]  # per choice: "stated" | "inferred" | "auto"
    seed: int
    nodes: int  # decisions explored
    end_field_codes: list[int]  # my MZONE + SZONE codes at the end
    detail: str
    skipped_steps: list[int] | None = None  # conditional steps skipped (anchor never legal, no results)
    deck_source: str = ""  # which deck verified against ("fixture" | "guide:<source>" | ...; set by caller)
    executed_steps: int = 0  # len(steps) - len(skipped)
    inferred_frac: float = 0.0  # inferred / agent (non-auto) choices
    hand_inferred: bool = False  # hand was empty; anchors inferred it (see infer_hand_codes)
    config: DuelConfig | None = None  # the exact puzzle the choices replay on (store this, not a rebuilt one)


# Steps whose anchor card must start in the hand (normal/set/activate from hand).
_IN_HAND_ACTIONS = frozenset({"normal_summon", "set_monster", "set_spelltrap", "activate"})
# Steps that put new cards on the field from the deck/extra (never in-hand anchors).
_SPECIAL_ACTIONS = frozenset(
    {
        "special_summon",
        "link_summon",
        "synchro_summon",
        "xyz_summon",
        "fusion_summon",
        "ritual_summon",
    }
)


def _is_search_or_special(step) -> bool:
    """A step after which later anchors may come from the deck, not the opening hand."""
    if step.action in _SPECIAL_ACTIONS:
        return True
    own = step.card_code
    return any(c is not None and c != own for c in step.results_codes)


def infer_hand_codes(line: ComboLine) -> list[int]:
    """Infer an opening hand for lines whose hand field is empty (MDM lazy-load loss).

    Take the anchor card of each leading step that is an in-hand action
    (normal_summon/set/activate), stopping after the first search/special effect
    (a step whose results name a card other than its own, or any special summon):
    later cards may have been searched, so they are not hand cards. Only cards
    already named in the steps are used; nothing is invented. Returns [] when the
    line has a hand or nothing can be inferred (e.g. it opens with a fusion).
    """
    if any(c is not None for c in line.hand_codes):
        return []
    out: list[int] = []
    for s in line.steps:
        if s.action in _IN_HAND_ACTIONS and s.card_code is not None and s.card_code not in out:
            out.append(s.card_code)
        if _is_search_or_special(s):
            break
    return out


def effective_hand_codes(line: ComboLine) -> tuple[list[int], bool]:
    """(hand codes to play with, whether they were inferred)."""
    have = [c for c in line.hand_codes if c is not None]
    if have:
        return have, False
    inf = infer_hand_codes(line)
    return inf, bool(inf)


def inferred_frac_of(labels: list[str]) -> float:
    agent = [x for x in labels if x != "auto"]
    if not agent:
        return 0.0
    return sum(1 for x in agent if x == "inferred") / len(agent)


def quality_guard(n: int, skipped: list[int] | None, labels: list[str]) -> str | None:
    """Reject vacuous / over-skipped / fragment lines (docs/line_executor.md).

    Priority: vacuous (all steps skipped, or 0 agent choices) > too_skipped
    (>1/3 of steps skipped) > trivial (<2 executed steps). Returns the reason
    or None when the line passes.
    """
    sk = list(skipped or [])
    agent = [x for x in labels if x != "auto"]
    if n == 0 or (n > 0 and len(sk) >= n) or len(agent) == 0:
        return "vacuous"
    if len(sk) * 3 > n:
        return "too_skipped"
    if n - len(sk) < 2:
        return "trivial"
    return None


def going_second(line: ComboLine) -> bool:
    return line.going == "second" or any(st.action == "attack" for st in line.steps)


# Fallback opponent board when no verified line of another deck exists: two face-up Blue-Eyes.
_FALLBACK_BOARD = (89631139, 89631139)


def opponent_board(engine: Engine, line: ComboLine, seed: int, verified_dir: Path | None = None):
    """(opponent Deck, placements) for a going-second line: the end board of a verified going-first line of
    *another* deck (a realistic board to break), chosen by seed; else a fallback board."""
    verified_dir = verified_dir or VERIFIED_DIR
    recs = []
    for f in sorted(verified_dir.glob("*.jsonl")) if verified_dir.exists() else []:
        for raw in f.read_text().splitlines():
            if raw.strip():
                r = json.loads(raw)
                first = any(pl[0] == 0 and pl[2] == C.LOCATION_HAND for pl in r["config"]["setup"])
                if first and not r["deck"].startswith(line.deck):
                    recs.append(r)
    if recs:
        r = recs[seed % len(recs)]
        deck = Deck(tuple(r["config"]["deck0_main"]), tuple(r["config"]["deck0_extra"]))
        board = [c for c in r["end_board_codes"] if c is not None]
    else:
        deck, board = Deck(tuple([89631139] * 3 + [8240199] * 37), ()), list(_FALLBACK_BOARD)
    places, mz, sz = [], 0, 0
    for code in board:
        cd = engine.cards.get(code)
        if cd is None:
            continue
        if cd.type & C.TYPE_MONSTER:
            if mz < 5:
                places.append(Placement(0, code, C.LOCATION_MZONE, mz, C.POS_FACEUP_ATTACK))
                mz += 1
        elif cd.type & C.TYPE_FIELD:
            places.append(Placement(0, code, C.LOCATION_SZONE, 5, C.POS_FACEUP))
        elif sz < 5:
            pos = C.POS_FACEDOWN if cd.type & C.TYPE_TRAP else C.POS_FACEUP
            places.append(Placement(0, code, C.LOCATION_SZONE, sz, pos))
            sz += 1
    return deck, tuple(places)


def make_config(line: ComboLine, deck: Deck, seed: int = 0, engine: Engine | None = None) -> DuelConfig:
    """Puzzle start. Going first: my hand placed, deck minus those cards, opponent = same deck, empty hand.
    Going second (docs/line_executor.md): the opponent (player 0) starts with a board from `opponent_board`
    and passes; I am player 1 with the line's hand and draw normally on my turn 2."""
    hand, _ = effective_hand_codes(line)
    main = list(deck.main)
    for c in hand:
        if c in main:
            main.remove(c)
    mine = Deck(tuple(main), tuple(deck.extra), ())
    if not going_second(line):
        setup = tuple(Placement(0, c, C.LOCATION_HAND) for c in hand)
        return DuelConfig(mine, deck, seed=seed, hand=0, setup=setup)
    if engine is None:
        raise ValueError("a going-second line needs the engine to build the opponent board")
    opp_deck, board = opponent_board(engine, line, seed)
    setup = board + tuple(Placement(1, c, C.LOCATION_HAND) for c in hand)
    return DuelConfig(opp_deck, mine, seed=seed, hand=0, setup=setup)


def _unresolved(line: ComboLine) -> list[str]:
    bad = [f"hand:{n}" for n, c in zip(line.hand, line.hand_codes) if n and c is None]
    bad += [f"requires:{n}" for n, c in zip(line.requires, line.requires_codes) if n and c is None]
    for s in line.steps:
        if s.card_code is None:
            bad.append(f"step{s.i}:{s.card or s.action}")
        for names, codes in (
            (s.materials, s.materials_codes),
            (s.targets, s.targets_codes),
            (s.results, s.results_codes),
        ):
            bad += [f"step{s.i}:{n}" for n, c in zip(names, codes) if n and c is None]
    return bad


def _missing(line: ComboLine, deck: Deck) -> str:
    main = Counter(deck.main)
    hand, _ = effective_hand_codes(line)
    for c in hand:
        if main[c] <= 0:
            return f"hand card {c} not in deck"
        main[c] -= 1
    pool = main + Counter(deck.extra)
    for c in line.requires_codes:
        if c is None:
            continue
        if pool[c] <= 0:
            return f"required card {c} not in deck/extra"
        pool[c] -= 1
    have = set(deck.main) | set(deck.extra) | {c for c in line.hand_codes if c is not None}
    for s in line.steps:
        if s.card_code is not None and s.card_code not in have:
            return f"step{s.i} {s.card} ({s.card_code}) not in deck/hand"
    return ""


def apply_generics(line: ComboLine, deck: Deck, cards) -> ComboLine:
    """A copy of `line` where generic descriptions ("a Branded Spell/Trap", "Bystial") that match cards of `deck`
    become constrained wildcards (docs/line_executor.md "Generic names"). Names that match nothing stay unresolved."""
    line = copy.deepcopy(line)
    deck_codes = set(deck.main) | set(deck.extra)
    used: list[str] = []

    def cands(name: str) -> list[int]:
        return resolve_generic(name, deck_codes, cards) if name else []

    for i, (n, c) in enumerate(zip(line.hand, line.hand_codes)):
        if c is None and (cc := cands(n)):
            line.hand_codes[i] = cc[0]  # any matching card is a legal reading; take the first
            used.append(f"hand:{n}->{cc[0]}")
    keep = [(n, c) for n, c in zip(line.requires, line.requires_codes) if c is not None or not cands(n)]
    if len(keep) != len(line.requires):
        used.append("requires: generic entries dropped")
        line.requires, line.requires_codes = [n for n, _ in keep], [c for _, c in keep]
    for st in line.steps:
        st.any_codes = set()
        if st.card_code is None and (cc := cands(st.card)):
            st.card_code, st.card_any = cc[0], set(cc)
            st.any_codes |= set(cc)
            used.append(f"step{st.i}:{st.card}->{len(cc)} cards")
        for names, codes in ((st.materials, st.materials_codes), (st.targets, st.targets_codes)):
            for j, (n, c) in enumerate(zip(names, codes)):
                if c is None and (cc := cands(n)):
                    codes[j] = cc[0]
                    st.any_codes |= set(cc)
        keep_r = [(n, c) for n, c in zip(st.results, st.results_codes) if c is not None or not cands(n)]
        st.results, st.results_codes = [n for n, _ in keep_r], [c for _, c in keep_r]
    if used:
        line.notes = (
            f"{line.notes} | generic: {'; '.join(used)}" if line.notes else f"generic: {'; '.join(used)}"
        )
    return line


def _anchor(step_i: int, steps: list, kind: int, code: int) -> bool:
    """Whether option (kind, code) starts step `step_i` (docs/line_executor.md anchor table)."""
    s = steps[step_i]
    card = s.card_code
    if card is None:
        return False
    any_card = getattr(s, "card_any", None)
    if any_card and code in any_card:
        code = card  # a generic step card: any matching card anchors it
    a = s.action
    if a == "attack":
        return kind == Kind.ATTACK and code == card
    if a == "normal_summon":
        return kind == Kind.SUMMON and code == card
    # Guides describe many effects by their outcome ("Roar sets itself from the GY"): a set step is also
    # anchored by an activation of the same card.
    if a == "set_monster":
        return kind in (Kind.MSET, Kind.ACTIVATE, Kind.YES) and code == card
    if a == "set_spelltrap":
        return kind in (Kind.SSET, Kind.ACTIVATE, Kind.YES) and code == card
    if a == "activate":
        return kind in (Kind.ACTIVATE, Kind.YES) and code == card
    if a == "special_summon":
        named = {card} | {c for c in s.materials_codes + s.targets_codes if c is not None}
        return (kind == Kind.SPSUMMON and code == card) or (
            kind in (Kind.ACTIVATE, Kind.YES) and code in named
        )
    if a in ("link_summon", "synchro_summon", "xyz_summon"):
        return (kind == Kind.SPSUMMON and code == card) or (kind == Kind.YES and code == card)
    if a in ("fusion_summon", "ritual_summon"):
        return (
            kind == Kind.ACTIVATE
            or (kind == Kind.SPSUMMON and code == card)
            or (kind == Kind.YES and code == card)
        )
    return (kind == Kind.ACTIVATE and code == card) or (kind == Kind.YES and code == card)


class _Budget(Exception):
    pass


def verify_line(engine: Engine, line: ComboLine, deck: Deck, seed: int = 0, budget: int = 2000) -> Verdict:
    """Execute `line` from a puzzle start; keep it only if the end board is reached."""
    _, hand_inf = effective_hand_codes(line)
    if hand_inf and "hand_inferred" not in (line.notes or ""):
        line.notes = f"{line.notes} | hand_inferred=True" if line.notes else "hand_inferred=True"
    if line.going == "first" and any(s.action == "attack" for s in line.steps):
        return Verdict(
            line.id,
            False,
            "out_of_scope",
            [],
            [],
            seed,
            0,
            [],
            "a going-first line cannot attack on turn 1",
            [],
            hand_inferred=hand_inf,
        )
    line = apply_generics(line, deck, engine.cards)
    bad = _unresolved(line)
    if bad:
        return Verdict(
            line.id,
            False,
            "unresolved",
            [],
            [],
            seed,
            0,
            [],
            f"unresolved names: {bad[:8]}",
            [],
            hand_inferred=hand_inf,
        )
    miss = _missing(line, deck)
    if miss:
        return Verdict(line.id, False, "missing_card", [], [], seed, 0, [], miss, [], hand_inferred=hand_inf)
    config = make_config(line, deck, seed, engine)
    v = _search(engine, config, line, seed, budget, hand_inferred=hand_inf)
    v.config = config
    return v


def _search(
    engine: Engine,
    config: DuelConfig,
    line: ComboLine,
    seed: int,
    budget: int,
    hand_inferred: bool = False,
) -> Verdict:
    steps, n = line.steps, len(line.steps)
    sets, res, skippable = [], [], []
    for s in steps:
        codes = {s.card_code} if s.card_code is not None else set()
        for cc in (s.materials_codes, s.targets_codes, s.results_codes):
            codes |= {c for c in cc if c is not None}
        codes |= getattr(s, "any_codes", set())
        sets.append(codes)
        res.append([c for c in s.results_codes if c is not None])
        # Attacks have no listed results but are never optional: their anchor only exists in the Battle Phase,
        # so at a main-phase boundary the skip rule would otherwise fire before the search tries TO_BATTLE.
        skippable.append(
            not res[-1] and not any(s.materials_codes + s.targets_codes) and s.action != "attack"
        )
    later = [set().union(*sets[i:]) for i in range(n)]  # cards mentioned from step i on
    endset = {c for c in line.end_board_codes if c is not None}
    end_unres = sum(1 for c in line.end_board_codes if c is None)

    me = 1 if going_second(line) else 0
    my_turn = 1 + me  # going second, my turn is turn 2
    env = CppEnv(engine, config)
    obs = env.reset()
    choices: list[int] = []
    labels: list[str] = []
    # My (hand, field, other) after each choice (aligned with `choices`), and the depth at which each step's
    # anchor was taken: a summon step's result also counts if it left the deck during the step and moved on
    # ("Special Summon Saji from the Deck, then Tribute it" never shows Saji on the field).
    fhist: list[tuple[tuple[int, ...], ...]] = []
    start: list[int | None] = [None] * n
    nodes = 0
    post_count = 0
    visited: set = set()
    anchored = [False] * n
    skipped: list[int] = []
    implicit: list[int] = []  # steps the engine executed without asking (forced triggers)
    cur = [(), (), ()]  # my (hand, field, other) code tuples at the current obs
    best = [-1, [], []]  # deepest point reached: [step_idx, field codes, choices to get there]
    done: dict = {}
    carry_log: list[tuple[int, list[int]]] = []  # (from_step, missing codes) carried on this path

    def snap() -> None:
        rows = obs["cards"][obs["card_mask"]]
        mine = rows[(rows[:, 3] ^ env.player) == me]  # controller is relative to the player to act
        code, loc = mine[:, 0], mine[:, 1]
        cur[0] = tuple(sorted(code[loc == 2].tolist()))
        cur[1] = tuple(sorted(code[(loc == _LOC_MZONE) | (loc == _LOC_SZONE)].tolist()))
        # out of the deck but not hand/field: GY (5), banished (6), xyz material (8). NOT the Extra Deck (7):
        # an Extra Deck card has not "arrived" anywhere.
        cur[2] = tuple(sorted(code[(loc == 5) | (loc == 6) | (loc == 8)].tolist()))

    def take(idx: int, lab: str) -> None:
        nonlocal nodes, obs
        choices.append(idx)
        labels.append(lab)
        nodes += 1
        if nodes > budget:
            raise _Budget()
        obs = env.step(idx)
        if obs is not None:
            snap()
        fhist.append(tuple(cur))

    def restore(depth: int) -> None:
        nonlocal obs, env
        del choices[depth:]
        del labels[depth:]
        del fhist[depth:]
        for i in range(n):
            if start[i] is not None and start[i] >= depth:
                start[i] = None
        env = CppEnv(engine, config)
        obs = env.reset()
        for i in choices:
            obs = env.step(i)
        assert obs is not None
        snap()

    def passive_option() -> int:
        """The opponent's answer: pass / no / end / finish / cancel when available, else the first option."""
        for i, k, _ in options():
            if k in _PASSIVE:
                return i
        return 0

    def options() -> list[tuple[int, int, int]]:
        m = int(obs["action_mask"].sum())
        a = obs["actions"][:m]
        return [(i, int(a[i, 0]), int(a[i, 2])) for i in range(m)]

    def statekey(si: int, at: bool, cfrom: int | None, ccodes: tuple[int, ...] | None) -> tuple:
        g = obs["global"]
        return (
            si,
            at,
            cfrom,
            ccodes,
            int(g[2]),
            int(g[3]),
            int(g[21]),
            int(g[20]),
            cur[0],
            cur[1],
            cur[2],
            tuple((k, c) for _, k, c in options()),
        )

    def pool(si: int) -> Counter:
        """Where step si's results must be: on my field for summon/set steps, else anywhere out of the deck."""
        if steps[si].action in _ON_FIELD:
            have = Counter(cur[1])
            if start[si] is not None:
                want = set(res[si])
                # Copies that newly left the deck for GY/banished/material during the step were summoned and
                # moved on. Copies that left the hand (discards, costs) or the field (a resolved spell) do not.
                h0, f0, o0 = fhist[start[si] - 1] if start[si] > 0 else fhist0
                for c in want:
                    left = max(0, h0.count(c) - cur[0].count(c)) + max(0, f0.count(c) - cur[1].count(c))
                    gone = cur[2].count(c) - o0.count(c) - left
                    if gone > 0:
                        have[c] = max(have[c], cur[1].count(c) + gone)
            return have
        return Counter(cur[0] + cur[1] + cur[2])

    def results_out(si: int) -> bool:
        have = pool(si)
        return all(have[c] >= k for c, k in Counter(res[si]).items())

    def some_arrived(si: int) -> bool:
        if not res[si]:
            return False
        have = pool(si)
        return any(have[c] > 0 for c in res[si])

    def missing_of(si: int) -> list[int]:
        have = pool(si)
        miss: list[int] = []
        for c, k in Counter(res[si]).items():
            miss.extend([c] * max(0, k - have[c]))
        return miss

    def make_carry(miss: list[int]) -> tuple[tuple[int, int], ...]:
        """Carried results with a baseline: off-field copies already out now don't count later (a card that
        stayed in the GY was never summoned), but ones that newly arrive there do (used as material)."""
        off = Counter(cur[0] + cur[2])
        return tuple((c, off[c]) for c in miss)

    def carried_out(ccodes) -> bool:
        if not ccodes:
            return True
        field, off = Counter(cur[1]), Counter(cur[0] + cur[2])
        need = Counter(c for c, _ in ccodes)
        base = {c: b for c, b in ccodes}
        return all(field[c] + max(0, off[c] - base[c]) >= k for c, k in need.items())

    def end_ok() -> bool:
        if not endset:
            return True
        have = Counter(cur[1])
        return all(have[c] >= k for c, k in Counter(endset).items())

    def stats() -> str:
        mine_labs = [x for x in labels if x != "auto"]
        inf = sum(x == "inferred" for x in mine_labs)
        pct = 100.0 * inf / len(mine_labs) if mine_labs else 0.0
        return f"inferred {pct:.0f}% ({inf}/{len(mine_labs)})"

    def succeed() -> bool:
        exe = n - len(skipped)
        frac = inferred_frac_of(labels)
        carried = f"carried={carry_log}; " if carry_log else ""
        hand = "hand_inferred; " if hand_inferred else ""
        done["v"] = Verdict(
            line.id,
            True,
            "ok",
            list(choices),
            list(labels),
            seed,
            nodes,
            sorted(cur[1]),
            f"{n}/{n} steps; skipped={skipped}; implicit={implicit}; {carried}{hand}{stats()}; nodes={nodes}; "
            f"unresolved_endboard={end_unres}",
            list(skipped),
            executed_steps=exe,
            inferred_frac=frac,
            hand_inferred=hand_inferred,
        )
        return True

    def dead(si: int) -> bool:
        if si > best[0]:
            best[0], best[1], best[2] = si, list(cur[1]), list(choices)
        return False

    def interchangeable() -> set[int]:
        """Options that only repeat an earlier one: picking another copy of the same card from the deck, hand or
        Extra Deck, or another main-monster / S/T zone while no Link monster (arrows) is on my field. Trying them
        after a failure only costs replays, since the visited-state key cannot tell them apart anyway."""
        m = int(obs["action_mask"].sum())
        a = obs["actions"][:m]
        links = any((cd := engine.cards.get(c)) is not None and cd.type & C.TYPE_LINK for c in cur[1])
        seen: set = set()
        drop: set[int] = set()
        for i in range(m):
            k, loc = int(a[i, 0]), int(a[i, 7])
            if k == Kind.SELECT and loc in (1, 2, 7):
                key = (k, *(int(x) for x in a[i, 2:10]))
            elif k == Kind.PLACE and not links:
                z = int(a[i, 5])
                key = (k, "mz" if z <= 4 else "sz" if 8 <= z <= 12 else z)
            else:
                continue
            if key in seen:
                drop.add(i)
            else:
                seen.add(key)
        return drop

    def order(si: int, at: bool, boundary: bool, anchors: list, stated: set) -> list[tuple[int, str]]:
        dup = interchangeable()
        opts = [o for o in options() if o[0] not in dup]
        anchors = [x for x in anchors if x[0] not in dup]
        anch = {i for i, _, _ in anchors}
        stated_rest = [
            (i, "stated") for i, k, c in opts if i not in anch and c in stated and k not in _DEMOTED
        ]
        plain = [
            (i, "inferred") for i, k, c in opts if i not in anch and c not in stated and k not in _DEMOTED
        ]
        movers = [(i, "inferred") for i, k, c in opts if k in _MOVERS]
        cancel = [
            (i, "inferred")
            for i, k, c in opts
            if i not in anch and c not in stated and k not in _MOVERS and k in _DEMOTED
        ]
        if not at and boundary and not anchors:
            return movers + stated_rest + plain + cancel
        return [(i, "stated") for i, _, _ in anchors] + stated_rest + plain + movers + cancel

    def post() -> bool:
        """After the last step: drive to the End Phase, then read the board."""
        nonlocal post_count
        if end_ok():
            return succeed()
        if post_count >= POST_CAP:
            return dead(n)
        key = statekey(n, False, None, None)
        if key in visited:
            return False
        visited.add(key)
        post_count += 1
        depth = len(choices)
        for idx, lab in order(n, False, True, [], endset):
            take(idx, lab)
            if dfs(n, False, None, None):
                return True
            restore(depth)
        return False

    def dfs(si: int, at: bool, cfrom: int | None, ccodes, extras: int = 0) -> bool:
        while obs is not None and env.player != me:  # the opponent never interacts: passive answers
            take(passive_option(), "auto")
        if obs is None or int(obs["global"][2]) > my_turn:
            # My turn is over (or the duel ended): a step whose trigger resolved as the
            # last thing still counts if its results are out of the deck.
            if at and si < n:
                if cfrom is not None and not carried_out(ccodes):
                    return dead(si)
                if results_out(si):
                    si += 1
                else:
                    return dead(si)
            elif cfrom is not None and not carried_out(ccodes):
                return dead(si)
            return succeed() if si >= n and end_ok() else dead(si)
        if si >= n:
            return post()
        dead(si)  # remember the deepest step reached (and the path), for --explain
        boundary = int(obs["global"][21]) in _COMMAND_MSGS and int(obs["global"][20]) == 0
        if boundary and at:  # step ends: its results must be out of the deck
            if cfrom is not None and not carried_out(ccodes):
                return False  # the previous step's carry is still missing: fail as before
            if not results_out(si):
                # De-bundle: when only some results arrived, the missing ones may be
                # satisfied by the end of the *next* step (carried forward once).
                if res[si] and some_arrived(si) and si + 1 < n:
                    miss = missing_of(si)
                    carry_log.append((si, list(miss)))
                    if dfs(si + 1, False, si, make_carry(miss), extras):
                        return True
                    carry_log.pop()
                return False
            return dfs(si + 1, False, None, None, extras)
        key = (statekey(si, at, cfrom, tuple(ccodes) if ccodes else None), extras)
        if key in visited:
            return False
        visited.add(key)
        opts = options()
        anchors = [(i, k, c) for i, k, c in opts if _anchor(si, steps, k, c)]
        cands = order(si, at, boundary, anchors, sets[si])
        depth = len(choices)
        # A step the engine already performed without asking (forced / already-resolved trigger of a card
        # involved in the previous step): its results are out, so it counts as executed.
        if (
            not at
            and boundary
            and not anchors
            and res[si]
            and si > 0
            and steps[si].card_code in sets[si - 1]
            and results_out(si)
        ):
            implicit.append(si)
            if dfs(si + 1, False, cfrom, ccodes, extras):
                return True
            implicit.pop()
        # Mid-chain transition: inside step si, an option that anchors step si+1 (e.g. a summon trigger
        # offered as yes/no) closes step si; whatever si's results still miss is carried to si+1's end.
        if at and not boundary and si + 1 < n and (cfrom is None or carried_out(ccodes)):
            for idx, k, c in opts:
                if not _anchor(si + 1, steps, k, c):
                    continue
                miss = missing_of(si)
                start[si + 1] = depth
                take(idx, "stated")
                anchored[si + 1] = True
                if miss:
                    carry_log.append((si, list(miss)))
                if dfs(si + 1, True, si if miss else None, make_carry(miss) if miss else None, extras):
                    return True
                if miss:
                    carry_log.pop()
                restore(depth)
        if not at and boundary and not anchors and skippable[si]:
            skipped.append(si)  # conditional step (e.g. removal vs an empty board): skip it
            if dfs(si + 1, False, cfrom, ccodes, extras):
                return True
            skipped.pop()
        anch_idx = {i for i, _, _ in anchors}
        between = not at and boundary  # a full main-phase decision before step si starts
        kc = {i: (k, c) for i, k, c in opts}
        for idx, lab in cands:
            extra = between and idx not in anch_idx and kc[idx][0] not in _MOVERS
            if extra and (extras >= MAX_EXTRA_ACTIONS or kc[idx][1] not in later[si]):
                continue  # not part of the line: don't explore the whole game tree
            if idx in anch_idx and not at:
                start[si] = depth
            take(idx, lab)
            if idx in anch_idx:
                anchored[si] = True
            if dfs(si, at or idx in anch_idx, cfrom, ccodes, extras + extra):
                return True
            restore(depth)
        return False

    snap()
    fhist0 = tuple(cur)
    try:
        found = dfs(0, False, None, None)
    except _Budget:
        if best[0] < 0:
            best[0], best[1] = 0, list(cur[1])
        return Verdict(
            line.id,
            False,
            "budget",
            list(best[2]),
            [],
            seed,
            nodes,
            best[1],
            f"node budget {budget} exhausted; {stats()}; skipped={skipped}",
            list(skipped),
            executed_steps=n - len(skipped),
            inferred_frac=inferred_frac_of(labels),
            hand_inferred=hand_inferred,
        )
    if found:
        v: Verdict = done["v"]
        g = quality_guard(n, v.skipped_steps, v.labels)
        if g is not None:
            exe = n - len(v.skipped_steps or [])
            return Verdict(
                line.id,
                False,
                g,
                list(v.choices),
                list(v.labels),
                seed,
                v.nodes,
                list(v.end_field_codes),
                f"{g}: {exe}/{n} steps executed, skipped={v.skipped_steps}; {stats()}; nodes={v.nodes}",
                list(v.skipped_steps or []),
                executed_steps=exe,
                inferred_frac=v.inferred_frac,
                hand_inferred=hand_inferred,
            )
        return v
    k = next((i for i in range(n) if not anchored[i]), None)
    if k is not None:
        return Verdict(
            line.id,
            False,
            f"no_anchor@{k}",
            list(best[2]),
            [],
            seed,
            nodes,
            best[1],
            f"anchor for step {k} ({steps[k].action} {steps[k].card}) never legal; "
            f"{stats()}; skipped={skipped}",
            list(skipped),
            executed_steps=n - len(skipped),
            inferred_frac=inferred_frac_of(labels),
            hand_inferred=hand_inferred,
        )
    return Verdict(
        line.id,
        False,
        "end_board_mismatch",
        list(best[2]),
        [],
        seed,
        nodes,
        best[1],
        f"all steps executed but end board missing; {stats()}; skipped={skipped}; "
        f"unresolved_endboard={end_unres}",
        list(skipped),
        executed_steps=n - len(skipped),
        inferred_frac=inferred_frac_of(labels),
        hand_inferred=hand_inferred,
    )
