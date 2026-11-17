"""Server replays (`.yrpX`) → the exact opponent turns of a recorded game (NOTEBOOK 2026-11-10 "replays").

A replay holds both decks, the seed and every response, so the duel is rebuilt exactly: true hands, true deck order,
our real history. `drive` walks `PyEnv` through the recorded responses and returns the option indices it took
(`PyEnv.replay(engine, cfg, choices[:k])` is then the exact state before choice k) plus, per choice, who chose, the
turn, the turn player and the option's key (its row without the board slot, stable when the board differs).
`ReplayLine` lets a seat replay its recorded keys on a duel that may have diverged (another seat played differently).
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from decklight.engine import constants as C
from decklight.engine import query as Q
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.engine.obs import MAX_OPTIONS


def key(row: np.ndarray) -> tuple[int, ...]:
    """Option row minus col 1 (the card's board slot)."""
    return tuple(int(v) for i, v in enumerate(row) if i != 1)


def _cards(r: bytes) -> tuple[int, ...] | None:
    """A card-list response (SELECT_CARD / TRIBUTE / SUM: int32 type 0-3, `parse_response_cards` in the core) as its
    sorted indices; None if `r` is not one. EDOPro clients send a bitfield (type 3), our engine an index list."""
    if len(r) < 4:
        return None
    t = int.from_bytes(r[:4], "little", signed=True)
    if t == 3:
        return tuple(i for i in range(8 * (len(r) - 4)) if r[4 + i // 8] >> (i % 8) & 1)
    if t not in (0, 1, 2) or len(r) < 8:
        return None
    n, w = int.from_bytes(r[4:8], "little"), (4, 2, 1)[t]
    if len(r) < 8 + n * w:
        return None
    return tuple(sorted(int.from_bytes(r[8 + i * w : 8 + (i + 1) * w], "little") for i in range(n)))


def same(a: bytes, b: bytes) -> bool:
    return a == b or (_cards(a) is not None and _cards(a) == _cards(b))


class _Budget(Exception):
    pass


def _path(
    dec, want: bytes, memo0: dict, depth: int = 8, budget: list[int] | None = None
) -> list[tuple[int, int]] | None:
    """[(index, n_options)] per pick whose takes make `want`, or None. Raises _Budget past 20,000 tries."""
    budget = [20_000] if budget is None else budget
    for i in range(len(dec.options())):
        budget[0] -= 1
        if budget[0] < 0:
            raise _Budget
        d = copy.deepcopy(dec, dict(memo0))
        r = d.choose(i)
        n = len(dec.options())
        if r is not None:
            if same(bytes(r), want):
                return [(i, n)]
        elif depth > 0:
            rest = _path(d, want, memo0, depth - 1, budget)
            if rest is not None:
                return [(i, n), *rest]
    return None


@dataclass
class Step:
    player: int
    turn: int
    turn_player: int
    key: tuple[int, ...]
    lp: tuple[int, int]
    field: tuple[int, int]  # cards on each player's field before the choice


@dataclass
class Driven:
    choices: list[int] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    end_lp: tuple[int, int] = (0, 0)
    winner: int | None = None
    obs: list[dict | None] = field(default_factory=list)  # per step, the observation when `drive(keep_obs=p)` and p chose


def signed(lp: int) -> int:
    """LP below 0 comes back as an unsigned 32-bit value."""
    return int(lp) - (1 << 32) if int(lp) >= 1 << 31 else int(lp)


def field_count(env: PyEnv, p: int) -> int:
    return sum(
        c is not None for loc in (C.LOCATION_MZONE, C.LOCATION_SZONE) for c in env.duel.query_location(p, loc)
    )


def lps(env: PyEnv) -> tuple[int, int]:
    lp = Q.lps(env.duel.query_field())
    return signed(lp[0]), signed(lp[1])


def drive(engine: Engine, cfg: DuelConfig, responses: list[bytes], keep_obs: int | None = None) -> Driven:
    """Raises ValueError when a recorded response matches no option path (or an auto answer differs)."""
    sent: list[bytes] = []
    out = Driven()
    memo0 = {id(engine.cards): engine.cards, id(engine): engine}

    class _Hooked:  # records every response, the automatic ones taken inside reset() included
        def __getattr__(self, name):
            return getattr(engine, name)

        def new_duel(self, config):
            duel = engine.new_duel(config)
            orig = duel.respond

            def respond(r):
                sent.append(bytes(r))
                return orig(r)

            duel.respond = respond
            return duel

    env = PyEnv(_Hooked(), cfg, auto_single=True)
    obs = env.reset()
    while obs is not None:
        k = len(sent)
        if k >= len(responses):
            break
        try:
            path = _path(env.dec, responses[k], memo0)
        except _Budget:
            raise ValueError(
                f"response {k} ({responses[k].hex()}): option search over budget, msg {env.dec.msg.type}"
            ) from None
        if path is None:
            raise ValueError(f"response {k} ({responses[k].hex()}) matches no option at turn {env.duel.turn}")
        for i, n in path:
            if n == 1:  # PyEnv took it itself
                continue
            if i >= MAX_OPTIONS:
                raise ValueError(f"response {k}: option {i} beyond MAX_OPTIONS")
            out.steps.append(Step(env.player, int(env.duel.turn), int(env.duel.turn_player), key(obs["actions"][i]),
                                  lps(env), (field_count(env, 0), field_count(env, 1))))  # fmt: skip
            out.obs.append({k: np.array(v) for k, v in obs.items()} if env.player == keep_obs else None)
            out.choices.append(i)
            obs = env.step(i)
        bad = next((j for j, (a, b) in enumerate(zip(sent, responses)) if not same(a, b)), None)
        if (
            bad is not None
        ):  # more sent than recorded: the duel ended outside the core (surrender), auto answers
            raise ValueError(
                f"responses differ from the replay at {bad} ({len(sent)} sent, {len(responses)} recorded)"
            )
    out.end_lp = lps(env)
    out.winner = env.duel.winner if obs is None else None
    return out


@dataclass
class ExactTurn:
    """One opponent turn: the duel before choice `start` is the turn's first decision."""

    turn: int
    start: int
    end: int  # choices[start:end] are the turn's choices (both seats)
    our_field: int
    lp_lost_real: int
    we_first: bool


def opponent_turns(d: Driven, me: int, min_field: int = 1) -> list[ExactTurn]:
    out = []
    by_turn: dict[int, list[int]] = {}
    for j, s in enumerate(d.steps):
        by_turn.setdefault(s.turn, []).append(j)
    turns = sorted(by_turn)
    first_player = d.steps[0].turn_player if d.steps else 0
    for ti, t in enumerate(turns):
        js = by_turn[t]
        s0 = d.steps[js[0]]
        if s0.turn_player == me or s0.field[me] < min_field:
            continue
        lp_end = d.steps[by_turn[turns[ti + 1]][0]].lp[me] if ti + 1 < len(turns) else d.end_lp[me]
        out.append(ExactTurn(t, js[0], js[-1] + 1, s0.field[me], s0.lp[me] - lp_end, first_player == me))
    return out


class ReplayLine:
    """A seat's recorded keys in order: `choose` takes the next one when an option has it, else returns None (the
    caller's policy decides; the key stays next, so a decision our divergence added is skipped over)."""

    def __init__(self, keys: list[tuple[int, ...]]) -> None:
        self.keys, self.pos, self.missed = keys, 0, 0

    def choose(self, obs: dict) -> int | None:
        if self.pos >= len(self.keys):
            return None
        rows = obs["actions"][: int(np.asarray(obs["action_mask"]).sum())]
        want = self.keys[self.pos]
        for i, row in enumerate(rows):
            if key(row) == want:
                self.pos += 1
                return i
        self.missed += 1
        return None


def our_seat(cfg: DuelConfig, our_cards: list[int]) -> int:
    """The core player whose deck overlaps `our_cards` most (Bo3: sided, so not equal)."""
    ours = Counter(our_cards)
    score = [sum((Counter(d.main + d.extra) & ours).values()) for d in (cfg.deck0, cfg.deck1)]
    if score[0] == score[1]:
        raise ValueError(f"cannot tell our deck: overlap {score}")
    return int(score[1] > score[0])
