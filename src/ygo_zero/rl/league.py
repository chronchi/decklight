"""Opponent league for RL v2 part A (ADR 0006 M4, PLAN Phase 5).

Episode types, drawn per env at episode start: self-play with probability `p_self`
(both seats learner, both trained), otherwise vs one league opponent (only the
learner seat is trained). League opponents: `base` (the frozen init), frozen
learner snapshots (`snap:<iter>`), fixed checkpoints (`--league-fixed name=path`,
never evicted), `greedy` (the scripted GreedyPolicy, routed on raw buf rows like
ygo-agent), and ygo-agent (`ygoagent`).

Among non-self episodes ygo-agent gets a fixed absolute share `p_ygoagent` (and
likewise `p_greedy` / `p_fixed` for greedy / the fixed pool); the rest is split
between base and snapshots by prioritized fictitious self-play (PFSP):
weight_o = (1 - w_o)^2 + 0.05, where w_o is the learner's EMA win rate vs
o on training games (init 0.5, EMA rate 0.05 per finished game). The fixed pool
shares one absolute slice `p_fixed`, split inside by PFSP among the fixed names.

Gauntlet-aware mode (strategy review E0): with `--ga-episode-share F > 0` the
VecEnv draws F of episodes gauntlet-vs-gauntlet and flags them in `buf["ga"]`;
the Collector then draws via `draw_ga`, so ygo-agent only ever plays
gauntlet-vs-gauntlet (the matchups the report card measures) and pool-deck
episodes never draw YGO (absolute shares renormalized over the non-YGO mass).
`fixed=None, p_fixed=0, p_greedy=0` keeps `sample()` bit-identical to before.

Threading: the Collector may run in a background thread (train_rl --overlap)
while the main thread adds snapshots, so every method takes one lock (re-entrant:
`draw_ga` nests inside `sample_non_ygo`).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

SELF = "self"
BASE = "base"
YGO = "ygoagent"
GREEDY = "greedy"

EMA_RATE = 0.05


def pfsp_weight(w: float) -> float:
    """PFSP sampling weight for a learner win rate w (harder opponents sampled more)."""
    return (1.0 - w) ** 2 + 0.05


class League:
    """PFSP opponent mix + frozen snapshot roster.

    Snapshot nets are opaque callables/objects held by reference (train_rl builds
    frozen PolicyNet copies); only evicted-snapshot stats are dropped here.
    """

    def __init__(self, p_self: float, p_ygoagent: float, league_size: int,
                 rng, ema_rate: float = EMA_RATE,
                 fixed: dict[str, object] | None = None, p_fixed: float = 0.0,
                 p_greedy: float = 0.0, ga_ygo_share: float = 0.0) -> None:
        self.p_self = float(p_self)
        self.p_ygoagent = float(p_ygoagent)
        self.p_fixed = float(p_fixed)
        self.p_greedy = float(p_greedy)
        self.ga_ygo_share = float(ga_ygo_share)
        for tag, v in (("p_self", self.p_self), ("p_ygoagent", self.p_ygoagent),
                       ("p_fixed", self.p_fixed), ("p_greedy", self.p_greedy),
                       ("ga_ygo_share", self.ga_ygo_share)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"league {tag}={v} not in [0, 1]")
        if self.p_self + self.p_ygoagent + self.p_fixed + self.p_greedy > 1.0 + 1e-12:
            raise ValueError("league absolute shares sum past 1 "
                             f"(self={self.p_self} ygo={self.p_ygoagent} "
                             f"fixed={self.p_fixed} greedy={self.p_greedy})")
        self.fixed: dict[str, object] = dict(fixed or {})
        if self.fixed and self.p_fixed <= 0.0:
            raise ValueError("league fixed opponents need --league-fixed-share > 0")
        if not self.fixed and self.p_fixed > 0.0:
            raise ValueError("league --league-fixed-share > 0 but no --league-fixed name=path")
        for name in self.fixed:
            if name in (SELF, BASE, YGO, GREEDY) or name.startswith("snap:"):
                raise ValueError(f"league fixed name {name!r} collides with a reserved opponent")
        self.league_size = int(league_size)
        self.rng = rng
        self.ema_rate = float(ema_rate)
        self._lock = threading.RLock()
        self.snaps: OrderedDict[str, object] = OrderedDict()
        self.ema: dict[str, float] = {BASE: 0.5, **{n: 0.5 for n in self.fixed}}
        self.added = 0  # snapshots added ever (fail-loud: >0 expected once training runs)

    # -- roster ------------------------------------------------------------
    def add_snapshot(self, name: str, net) -> None:
        """Add a frozen snapshot; keep the newest `league_size`, drop evicted stats."""
        with self._lock:
            self.snaps[name] = net
            self.ema.setdefault(name, 0.5)
            self.added += 1
            while len(self.snaps) > self.league_size:
                old, _ = self.snaps.popitem(last=False)
                self.ema.pop(old, None)

    def get(self, name: str):
        """Snapshot/fixed net by name (KeyError with context when evicted/unknown)."""
        with self._lock:
            try:
                return self.snaps[name]
            except KeyError:
                pass
            try:
                return self.fixed[name]
            except KeyError:
                raise KeyError(f"league opponent {name!r} not in roster "
                               f"(have {list(self.snaps) + list(self.fixed)})") from None

    def names(self) -> list[str]:
        with self._lock:
            return [BASE, *self.snaps, *self.fixed]

    # -- sampling ----------------------------------------------------------
    def _weights_locked(self) -> dict[str, float]:
        return {name: pfsp_weight(self.ema.get(name, 0.5)) for name in [BASE, *self.snaps]}

    def _pick_locked(self, pool: list[str]) -> str:
        """PFSP pick among `pool` (one fresh uniform; lock held by the caller)."""
        w = {name: pfsp_weight(self.ema.get(name, 0.5)) for name in pool}
        tot = sum(w.values())
        # tot > 0 always (each weight >= 0.05); fall back to the first name regardless.
        x = float(self.rng.random()) * tot if tot > 0 else 0.0
        for name, wi in w.items():
            x -= wi
            if x <= 0:
                return name
        return pool[0]

    def sampling_probs(self) -> dict[str, float]:
        """Absolute per-episode probability of each opponent type (plus `self`)."""
        with self._lock:
            probs = {SELF: self.p_self}
            if self.p_greedy > 0:
                probs[GREEDY] = self.p_greedy
            probs[YGO] = self.p_ygoagent
            if self.fixed:
                fw = {n: pfsp_weight(self.ema.get(n, 0.5)) for n in self.fixed}
                ftot = sum(fw.values())
                for name, wi in fw.items():
                    probs[name] = self.p_fixed * wi / ftot if ftot > 0 else 0.0
            rest = max(0.0, 1.0 - self.p_self - self.p_greedy - self.p_ygoagent - self.p_fixed)
            w = self._weights_locked()
            tot = sum(w.values())
            for name, wi in w.items():
                probs[name] = rest * wi / tot if tot > 0 else 0.0
            return probs

    def sample(self) -> str:
        """Draw one episode's opponent type.

        With no fixed/greedy opponents the RNG draws and boundaries are exactly
        the old ones (bit-identical for the same seed).
        """
        with self._lock:
            r = float(self.rng.random())
            if r < self.p_self:
                return SELF
            if r < self.p_self + self.p_greedy:
                return GREEDY
            if r < self.p_self + self.p_greedy + self.p_ygoagent:
                return YGO
            if self.fixed and r < self.p_self + self.p_greedy + self.p_ygoagent + self.p_fixed:
                return self._pick_locked(list(self.fixed))
            w = self._weights_locked()
            tot = sum(w.values())
            # tot > 0 always (each weight >= 0.05); fall back to base regardless.
            x = float(self.rng.random()) * tot if tot > 0 else 0.0
            for name, wi in w.items():
                x -= wi
                if x <= 0:
                    return name
            return BASE

    def sample_non_ygo(self) -> str:
        """One opponent draw conditioned on 'not YGO' (gauntlet-aware rollouts).

        Absolute shares renormalized over the non-YGO mass; only called when
        `--ga-episode-share > 0`, so the RNG stream differs from `sample` by design.
        """
        with self._lock:
            tot = max(1e-12, 1.0 - self.p_ygoagent)
            r = float(self.rng.random()) * tot
            if r < self.p_self:
                return SELF
            r -= self.p_self
            if r < self.p_greedy:
                return GREEDY
            r -= self.p_greedy
            if self.fixed and r < self.p_fixed:
                return self._pick_locked(list(self.fixed))
            return self._pick_locked([BASE, *self.snaps])

    def draw_ga(self, is_ga: bool) -> str:
        """Gauntlet-aware episode draw: `is_ga` is the VecEnv's flag for this env's
        current episode (both decks gauntlet). YGO is returned only for gauntlet
        episodes (with probability `ga_ygo_share`); everything else comes from
        `sample_non_ygo`, so pool-deck episodes never draw YGO."""
        with self._lock:
            if is_ga and float(self.rng.random()) < self.ga_ygo_share:
                return YGO
            return self.sample_non_ygo()

    # -- results -----------------------------------------------------------
    def record(self, opp: str, learner_result: float) -> None:
        """Feed one finished training game vs `opp` (1/0.5/0 from the learner's seat)."""
        if opp == SELF:
            return
        with self._lock:
            w = self.ema.get(opp, 0.5)
            self.ema[opp] = w + self.ema_rate * (float(learner_result) - w)

    def weight_of(self, opp: str) -> float:
        with self._lock:
            return pfsp_weight(self.ema.get(opp, 0.5))
