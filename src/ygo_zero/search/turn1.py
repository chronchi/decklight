"""Turn-1 search teacher (ADR 0006 M2): one-step policy improvement by engine look-ahead.

Player 0 goes first, the opponent answers passively (as in the playtest combo test). At every player-0 decision the
policy's top-k legal options are each played out to the end of turn 1 by the policy itself (argmax, passive
opponent), on engine clones made by replay (config + choice prefix, ~15 ms). The option whose rollout ends with the
best board is taken; ties keep the policy's ranking. The result is a record of the searched turn (game-record format,
labels "search" for player 0, "opp" for player 1) that BC can distill (kind `search`).

Board score = cards player 0 controls on the field (mzone + szone, the combo test's measure) + 0.01 x cards in its
hand (tie-break: same board with more resources left). The policy is any `logits(obs_batch) -> [n, A]` callable
(BCPolicy.logits on a box; random logits in tests: the laptop never runs a model).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine

LogitsFn = Callable[[dict[str, np.ndarray]], np.ndarray]
FIELD_LOCS = (3, 4)  # obs card column 1: 3 mzone, 4 szone (docs/obs_spec.md)
HAND_LOC = 2
_PASSIVE = (Kind.PASS, Kind.NO, Kind.FINISH, Kind.CANCEL, Kind.TO_END)


def passive_choice(obs) -> int:
    """The opponent's answer: the first PASS / NO / FINISH / CANCEL / TO_END option, else option 0."""
    acts, mask = obs["actions"], obs["action_mask"]
    for i in np.flatnonzero(mask):
        if int(acts[i, 0]) in _PASSIVE:
            return int(i)
    return 0


def board_score(obs, ctrl: int) -> float:
    """Field cards (+ 0.01 x hand cards) of the side whose relative controller column (obs card column 3) is ctrl."""
    c, m = obs["cards"], obs["card_mask"]
    rows = c[np.flatnonzero(m)]
    mine = rows[rows[:, 3] == ctrl]
    return float(np.isin(mine[:, 1], FIELD_LOCS).sum() + 0.01 * (mine[:, 1] == HAND_LOC).sum())


def _batch(obs_list: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _masked(logits: np.ndarray, obs: dict) -> np.ndarray:
    z = np.array(logits[: obs["action_mask"].shape[0]], dtype=np.float64)
    z[~obs["action_mask"].astype(bool)] = -np.inf
    return z


@dataclass
class Turn1Result:
    choices: list[int]
    actors: list[int]
    labels: list[str]
    end_score: float  # board_score of player 0 at the end of turn 1
    decisions: int  # player-0 decisions
    searched: int  # player-0 decisions with >= 2 candidates
    overrides: int  # searched decisions where the search left the policy's top choice
    log: list[dict] = field(default_factory=list)  # per searched decision: k, candidates, scores, chosen


class Turn1Search:
    """The search is written as generators that yield lists of obs needing policy logits and receive the logits back
    (`*_gen`), so many hands can share one batched forward (`run_many`, GPU on a box); `run` / `greedy` /
    `rollouts` drive a single hand with `self.logits`."""

    def __init__(self, engine: Engine, logits: LogitsFn, k: int = 4, max_decisions: int = 300,
                 auto_single: bool = True, batch: int = 512) -> None:  # fmt: skip
        self.engine, self.logits, self.k = engine, logits, k
        self.max_decisions, self.auto_single, self.batch = max_decisions, auto_single, batch

    def _env(self, config: DuelConfig, prefix: list[int]):
        env = CppEnv(self.engine, config, self.auto_single)
        obs = env.reset()
        for c in prefix:
            obs = env.step(c)
        return env, obs

    def _forward(self, obs_list: list[dict]) -> np.ndarray:
        out = [self.logits(_batch(obs_list[i : i + self.batch])) for i in range(0, len(obs_list), self.batch)]
        return np.concatenate(out)

    def _drive(self, gen):
        try:
            req = next(gen)
            while True:
                req = gen.send(self._forward(req))
        except StopIteration as stop:
            return stop.value

    def rollouts_gen(self, config: DuelConfig, prefixes: list[list[int]], n0: int):
        """Play each prefix out to the end of turn 1 (policy argmax for player 0, passive player 1) in lockstep,
        one logits request per step. n0 = player-0 decisions already in the prefixes (shared decision cap)."""
        envs, obs = zip(*(self._env(config, p) for p in prefixes), strict=True)
        envs, obs = list(envs), list(obs)
        last = [0.0] * len(envs)
        n = [n0] * len(envs)
        live = [True] * len(envs)
        while any(live):
            mine = []
            for i, e in enumerate(envs):
                if not live[i]:
                    continue
                while obs[i] is not None and int(obs[i]["global"][2]) <= 1 and e.player == 1:
                    obs[i] = e.step(passive_choice(obs[i]))
                if obs[i] is None or int(obs[i]["global"][2]) > 1 or n[i] >= self.max_decisions:
                    live[i] = False
                    if obs[i] is not None:
                        last[i] = board_score(obs[i], 0 if e.player == 0 else 1)
                    continue
                last[i] = board_score(obs[i], 0)
                mine.append(i)
            if not mine:
                break
            lg = yield [obs[i] for i in mine]
            for i, row in zip(mine, lg, strict=True):
                n[i] += 1
                obs[i] = envs[i].step(int(np.argmax(_masked(row, obs[i]))))
        return last

    def run_gen(self, config: DuelConfig):
        """Search player 0's whole turn 1 on this config (player 0 goes first). Returns a Turn1Result."""
        env = CppEnv(self.engine, config, self.auto_single)
        obs = env.reset()
        res = Turn1Result([], [], [], 0.0, 0, 0, 0)
        while obs is not None and int(obs["global"][2]) <= 1 and res.decisions < self.max_decisions:
            if env.player == 1:
                a, label = passive_choice(obs), "opp"
            else:
                z = _masked((yield [obs])[0], obs)
                legal = np.flatnonzero(np.isfinite(z))
                cands = [int(i) for i in legal[np.argsort(-z[legal], kind="stable")][: self.k]]
                a, label = cands[0], "search"
                if len(cands) > 1:
                    scores = yield from self.rollouts_gen(config, [res.choices + [c] for c in cands], res.decisions + 1)
                    best = max(range(len(cands)), key=lambda j: (scores[j], -j))  # ties: policy order
                    a = cands[best]
                    res.searched += 1
                    res.overrides += best != 0
                    res.log.append({"k": len(res.choices), "cands": cands, "scores": [round(s, 2) for s in scores],
                                    "chosen": a})  # fmt: skip
                res.decisions += 1
            res.choices.append(a)
            res.actors.append(int(env.player))
            res.labels.append(label)
            obs = env.step(a)
        if obs is not None:
            res.end_score = board_score(obs, 0 if env.player == 0 else 1)
        return res

    def greedy_gen(self, config: DuelConfig):
        """The policy's own turn 1 (argmax, no search): its end-of-turn board score."""
        return (yield from self.rollouts_gen(config, [[]], 0))[0]

    def both_gen(self, config: DuelConfig):
        """(greedy score, searched Turn1Result) for one hand."""
        g = yield from self.greedy_gen(config)
        r = yield from self.run_gen(config)
        return g, r

    def rollouts(self, config: DuelConfig, prefixes: list[list[int]], n0: int) -> list[float]:
        return self._drive(self.rollouts_gen(config, prefixes, n0))

    def run(self, config: DuelConfig) -> Turn1Result:
        return self._drive(self.run_gen(config))

    def greedy(self, config: DuelConfig) -> float:
        return self._drive(self.greedy_gen(config))

    def run_many(self, jobs, width: int = 64):
        """Drive many hand generators at once through one batched forward per step (see `drive_many`)."""
        return drive_many(jobs, self._forward, width)


def drive_many(jobs, forward, width: int = 64):
    """Drive many search generators at once: `jobs` is an iterable of (key, generator); up to `width` run
    concurrently and every step's requests from all of them go through one batched forward. Yields
    (key, result) as hands finish (completion order)."""
    jobs = iter(jobs)
    live: dict[int, tuple] = {}  # slot -> (key, gen, pending request)
    slot = 0

    def start(key, gen):
        nonlocal slot
        try:
            live[slot] = (key, gen, next(gen))
            slot += 1
            return None
        except StopIteration as stop:
            return key, stop.value

    exhausted = False
    while True:
        while not exhausted and len(live) < width:
            nxt = next(jobs, None)
            if nxt is None:
                exhausted = True
                break
            done = start(*nxt)
            if done is not None:
                yield done
        if not live:
            return
        slots = list(live)
        reqs = [live[s][2] for s in slots]
        lg = forward([o for r in reqs for o in r])
        off = 0
        for s, r in zip(slots, reqs, strict=True):
            key, gen, _ = live[s]
            part = lg[off : off + len(r)]
            off += len(r)
            try:
                live[s] = (key, gen, gen.send(part))
            except StopIteration as stop:
                del live[s]
                yield key, stop.value
