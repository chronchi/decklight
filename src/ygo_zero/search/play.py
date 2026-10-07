"""Live game: base policy + WindowSearch (S) vs the same policy raw (R).

`search_game_gen` plays ONE full game through the `drive_many` protocol (see
search/turn1.py): it yields `[obs, ...]` lists and receives the batched policy
logits back, so many games share one forward. Both seats' non-searched moves use
the report card's model selection (scripts/playtest.py builds its BCPolicy with
the default temperature 0.0, i.e. argmax) behind a LoopGuard, exactly the way
BCPolicy.__call__ applies it. Only S's decisions are ever searched:

- (a) a response window (`is_window`: a chain window on the opponent's turn
  where S can ACTIVATE), or
- (b) S's own turn (`global[5] == 1`) with >= 2 legal options from turn 2 on
  (turn >= 1 with `own_turn1`), sampled with probability `own_frac`;

and in both cases only when the policy is not confident: the move is played
raw when `softmax(z).max() >= conf_skip`.

Searching is `WindowSearch._candidates` plus `rollouts_gen` on the branches
`(choices_so_far + [cand], j, cand, sample)`, playing the best candidate by the
search's selected score with the same tie rule as `game_gen`
(`max(..., key=(score, -j))`: ties keep the policy's order, so firing must earn
its card). The policy's own pick is the guard-masked argmax; an override is
counted when the search plays something else.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.eval.loopguard import LoopGuard
from ygo_zero.search.determinize import Determinizer, SeenTracker
from ygo_zero.search.turn1 import _masked
from ygo_zero.search.window import WindowSearch, is_window

# Takes the loop-guard-masked logits of one decision, returns the option index.
SelectFn = Callable[[np.ndarray], int]


def argmax_select(z: np.ndarray) -> int:
    """The report card's model selection: BCPolicy at temperature 0.0 plays argmax."""
    return int(np.argmax(z))


def softmax_conf(z: np.ndarray) -> float:
    """The policy's confidence: the max softmax probability over the legal options."""
    finite = np.isfinite(z)
    if int(finite.sum()) <= 1:
        return 1.0
    v = np.asarray(z[finite], dtype=np.float64)
    v -= v.max()
    e = np.exp(v)
    return float((e / e.sum()).max())


@dataclass
class GameResult:
    winner: int | None  # 0/1 seat, 2 = draw; None when truncated
    truncated: bool  # still running after max_decisions
    turns: int  # turn number of the last decision
    choices: list[int] = field(default_factory=list)
    actors: list[int] = field(default_factory=list)
    n_searched: int = 0  # S decisions where the search picked the move
    n_override: int = 0  # searched decisions where it left the policy's own pick
    search_s: float = 0.0  # wall seconds spent inside search (incl. rollout forwards)
    n_branches: int = 0  # rollout branches launched over the game
    n_capped: int = 0  # of those that hit max_rollout (scored as losses)
    searched_idx: list[int] = field(default_factory=list)  # decision indices searched
    det_proposals: int = 0  # honest worlds proposed (rejection-sampling tries)
    det_accepted: int = 0  # ... accepted (root obs identical to the truth)
    det_fallbacks: int = 0  # ... rejected after `tries` (fell back to the true config)
    det_s: float = 0.0  # wall seconds spent drawing honest worlds


def search_game_gen(
    engine: Engine,
    config: DuelConfig,
    search: WindowSearch,
    s_seat: int,
    select_fn: SelectFn,
    game_seed: int,
    own_turn1: bool = False,
    conf_skip: float = 0.9,
    own_frac: float = 1.0,
    max_decisions: int = 3000,
    auto_single: bool = True,
    honest: bool = False,
):
    """Play one live game; S (absolute seat `s_seat`) searches, R plays raw. Returns a GameResult.

    With `honest`, S's searched decisions roll out from determinized worlds (everything S has
    not seen re-sampled consistently with S's observation) instead of the true hidden state.
    """
    rng = random.Random(search.seed * 1_000_003 + game_seed)
    env = CppEnv(engine, config, auto_single)
    obs = env.reset()
    tracker = SeenTracker(s_seat) if honest else None
    if tracker is not None:
        tracker.update(env)
    guard = LoopGuard()  # one slot: both seats share it, as one VecEnv row does in playtest matches
    res = GameResult(winner=None, truncated=False, turns=1)
    capped0 = search.capped
    while obs is not None and len(res.choices) < max_decisions:
        me = int(env.player)
        turn = int(obs["global"][2])
        res.turns = turn
        n_legal = int(obs["action_mask"].sum())
        z = _masked((yield [obs])[0], obs)
        if bool(np.isnan(np.asarray(z, dtype=np.float64)).any()):
            raise RuntimeError(
                f"ALERT search_match: NaN logits at decision {len(res.choices)} (game_seed={game_seed})"
            )
        z_g, key = guard.mask(0, obs, z)
        a_pol = select_fn(z_g)
        a = a_pol
        if me == s_seat:
            window = is_window(obs, me)
            own = (
                int(obs["global"][5]) == 1
                and n_legal >= 2
                and turn >= (1 if own_turn1 else 2)
                and rng.random() < own_frac
            )
            if (window or own) and softmax_conf(z_g) < conf_skip:
                cands = search._candidates(z_g, obs, window)
                if len(cands) >= 2:
                    t0 = time.perf_counter()
                    k = len(res.choices)
                    branches = [
                        (res.choices + [x], j, x, s)
                        for j, x in enumerate(cands)
                        for s in range(search.samples)
                    ]
                    worlds = None
                    if tracker is not None:
                        det = Determinizer(
                            engine, config, auto_single, me, res.choices, tracker.seen,
                            obs, random.Random(f"{search.seed}:{game_seed}:{k}:det"),
                        )
                        worlds = []
                        for _ in range(search.samples):
                            w = det.sample()
                            worlds.append(w if w is not None else config)
                        res.det_proposals += det.proposals
                        res.det_accepted += det.accepted
                        res.det_fallbacks += det.fallbacks
                        res.det_s += det.seconds
                    scores_all = yield from search.rollouts_gen(
                        config, auto_single, branches, me, turn_start=turn,
                        game_seed=game_seed, decision_k=k, worlds=worlds,
                    )  # fmt: skip
                    res.search_s += time.perf_counter() - t0
                    sel_scores = (
                        scores_all[search._select_key()]
                        if isinstance(scores_all, dict)
                        else scores_all
                    )
                    best = max(range(len(cands)), key=lambda j: (sel_scores[j], -j))
                    a = cands[best]
                    res.n_searched += 1
                    res.n_branches += len(branches)
                    res.searched_idx.append(k)
                    if a != a_pol:
                        res.n_override += 1
        guard.note(0, key, a)
        res.choices.append(a)
        res.actors.append(me)
        obs = env.step(a)
        if tracker is not None:
            tracker.update(env)
    res.truncated = obs is not None
    res.winner = None if res.truncated else int(env.winner)
    res.n_capped = search.capped - capped0
    return res
