"""Mid-game decision search (ADR 0006 M3): response windows (fire an interruption or hold) and own-turn plays.

States come from recorded games (any record with `config` + `choices`: ygo-agent teacher games, gold games, DAgger or
self-play records). The game is replayed; at each selected decision of player `me` the candidates are tried on engine
clones (replay of the prefix + the candidate) and each branch is played out by the policy for BOTH players (argmax)
until `horizon` turns have ended (1 = the end of the current turn). The candidate with the best position for `me`
becomes the decision's target; ties keep the policy's order (at a chain window PASS is ranked first, so firing must
earn its card).

Position score for `me`, default: field cards (mzone + szone) mine − theirs, + 0.5 x (hand mine − theirs),
+ (LP mine − theirs) / 2000; a finished duel scores ±50. With `value` (a learned value, ADR 0007 v2: P(the acting
player wins) in [-1, 1], e.g. `ValueNet.values`) the branch end is scored by the value from `me`'s view instead and a
finished duel scores ±1, so the target captures long-term effects a board count misses. The rollouts see the true hidden state (perfect
information): the target is "what would have worked here", which the student learns in expectation.

Selected decisions: `window` = a chain window (MSG_SELECT_CHAIN) on the opponent's turn where `me` can ACTIVATE;
`own` = an own-turn decision with >= 2 options from turn 2 on, sampled with probability `own_frac`.
Written as generators (see turn1.py) so many games share one batched forward (`run_many`).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine
from ygo_zero.search.turn1 import _batch, _masked, drive_many

LogitsFn = Callable[[dict[str, np.ndarray]], np.ndarray]
ValueFn = Callable[[dict[str, np.ndarray]], np.ndarray]  # batched obs -> value in [-1, 1], actor's view
WIN = 50.0


def position_score(obs, actor: int, me: int) -> float:
    """Score for player `me` from an obs seen by player `actor` (absolute seats)."""
    g = obs["global"]
    mine, theirs = (slice(6, 13), slice(13, 20)) if actor == me else (slice(13, 20), slice(6, 13))
    lp_m, lp_t = (int(g[0]), int(g[1])) if actor == me else (int(g[1]), int(g[0]))
    cm, ct = g[mine], g[theirs]
    return float((cm[2] + cm[3]) - (ct[2] + ct[3]) + 0.5 * (cm[1] - ct[1]) + (lp_m - lp_t) / 2000)


def is_window(obs, actor: int) -> bool:
    """A chain window on the opponent's turn where the acting player can activate something."""
    g = obs["global"]
    if int(g[21]) != C.MSG_SELECT_CHAIN or int(g[5]) == 1:
        return False
    acts, mask = obs["actions"], obs["action_mask"].astype(bool)
    return bool(np.any(acts[mask, 0] == Kind.ACTIVATE))


@dataclass
class WindowResult:
    targets: dict[int, int] = field(default_factory=dict)  # decision index -> searched option
    kinds: dict[int, str] = field(default_factory=dict)  # decision index -> "window" | "own"
    log: list[dict] = field(default_factory=list)
    replay_ok: bool = True


class WindowSearch:
    def __init__(self, engine: Engine, logits: LogitsFn, k: int = 4, horizon: int = 1, own_frac: float = 0.0,
                 max_rollout: int = 300, batch: int = 512, seed: int = 0, value: ValueFn | None = None) -> None:  # fmt: skip
        self.engine, self.logits, self.k, self.horizon = engine, logits, k, horizon
        self.value, self.win = value, (1.0 if value is not None else WIN)
        self.own_frac, self.max_rollout, self.batch = own_frac, max_rollout, batch
        self.seed = seed

    def _forward(self, obs_list: list[dict]) -> np.ndarray:
        out = [self.logits(_batch(obs_list[i : i + self.batch])) for i in range(0, len(obs_list), self.batch)]
        return np.concatenate(out)

    def _candidates(self, z: np.ndarray, obs, window: bool) -> list[int]:
        legal = np.flatnonzero(np.isfinite(z))
        order = [int(i) for i in legal[np.argsort(-z[legal], kind="stable")]]
        if not window:
            return order[: self.k]
        kinds = obs["actions"][:, 0]
        passes = [i for i in order if int(kinds[i]) == Kind.PASS]
        acts = [i for i in order if int(kinds[i]) == Kind.ACTIVATE]
        return (passes[:1] + acts)[: max(self.k, 2)]

    def rollouts_gen(self, config: DuelConfig, auto_single: bool, prefixes: list[list[int]], me: int, end_turn: int):
        """Play each prefix out (policy argmax, both players) until the turn number exceeds end_turn or the duel
        ends; returns the position score for `me` of each branch."""
        envs, obs = [], []
        for p in prefixes:
            e = CppEnv(self.engine, config, auto_single)
            o = e.reset()
            for c in p:
                o = e.step(c)
            envs.append(e)
            obs.append(o)
        score = [0.0] * len(envs)
        n = [0] * len(envs)
        live = [True] * len(envs)
        pending: list[tuple[int, dict, int]] = []  # branch ends scored by the learned value in one call
        while any(live):
            act = []
            for i, e in enumerate(envs):
                if not live[i]:
                    continue
                if obs[i] is None:
                    live[i] = False
                    w = int(e.winner)
                    score[i] = self.win if w == me else -self.win if w in (0, 1) else 0.0
                    continue
                if int(obs[i]["global"][2]) > end_turn or n[i] >= self.max_rollout:
                    live[i] = False
                    if self.value is None:
                        score[i] = position_score(obs[i], int(e.player), me)
                    else:
                        pending.append((i, obs[i], int(e.player)))
                    continue
                act.append(i)
            if not act:
                break
            lg = yield [obs[i] for i in act]
            for i, row in zip(act, lg, strict=True):
                n[i] += 1
                obs[i] = envs[i].step(int(np.argmax(_masked(row, obs[i]))))
        if pending:
            v = self.value(_batch([o for _, o, _ in pending]))
            for (i, _, actor), x in zip(pending, v, strict=True):
                score[i] = float(x) if actor == me else -float(x)
        return score

    def game_gen(self, rec: dict, config: DuelConfig, game_seed: int = 0):
        """Replay a recorded game and search its selected decisions. Returns a WindowResult."""
        rng = random.Random(self.seed * 1_000_003 + game_seed)
        auto_single = rec.get("auto_single", True)
        env = CppEnv(self.engine, config, auto_single)
        obs = env.reset()
        res = WindowResult()
        choices = rec["choices"]
        for k, c in enumerate(choices):
            if obs is None:
                res.replay_ok = False
                break
            me, turn = int(env.player), int(obs["global"][2])
            n_legal = int(obs["action_mask"].sum())
            kind = None
            if is_window(obs, me):
                kind = "window"
            elif int(obs["global"][5]) == 1 and turn >= 2 and n_legal >= 2 and rng.random() < self.own_frac:
                kind = "own"
            if kind is not None:
                z = _masked((yield [obs])[0], obs)
                cands = self._candidates(z, obs, kind == "window")
                if len(cands) >= 2:
                    end = turn + self.horizon - 1
                    scores = yield from self.rollouts_gen(config, auto_single, [choices[:k] + [x] for x in cands], me, end)
                    best = max(range(len(cands)), key=lambda j: (scores[j], -j))
                    res.targets[k] = cands[best]
                    res.kinds[k] = kind
                    res.log.append({"k": k, "kind": kind, "turn": turn, "cands": cands,
                                    "scores": [round(s, 2) for s in scores], "chosen": cands[best],
                                    "played": int(c),
                                    "fired": int(obs["actions"][cands[best], 0]) == Kind.ACTIVATE})  # fmt: skip
            obs = env.step(int(c))
        return res

    def run_many(self, jobs, width: int = 32):
        """Drive (key, generator) jobs concurrently with one batched forward per step."""
        return drive_many(jobs, self._forward, width)
