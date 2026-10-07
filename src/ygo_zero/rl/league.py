"""Opponent league for RL v2 part A (ADR 0006 M4, PLAN Phase 5).

Episode types, drawn per env at episode start: self-play with probability `p_self`
(both seats learner, both trained), otherwise vs one league opponent (only the
learner seat is trained). League opponents: `base` (the frozen init), frozen
learner snapshots (`snap:<iter>`), and ygo-agent (`ygoagent`).

Among non-self episodes ygo-agent gets a fixed absolute share `p_ygoagent`; the
rest is split between base and snapshots by prioritized fictitious self-play
(PFSP): weight_o = (1 - w_o)^2 + 0.05, where w_o is the learner's EMA win rate vs
o on training games (init 0.5, EMA rate 0.05 per finished game).

Threading: the Collector may run in a background thread (train_rl --overlap)
while the main thread adds snapshots, so every method takes one lock.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

SELF = "self"
BASE = "base"
YGO = "ygoagent"

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
                 rng, ema_rate: float = EMA_RATE) -> None:
        self.p_self = float(p_self)
        self.p_ygoagent = float(p_ygoagent)
        self.league_size = int(league_size)
        self.rng = rng
        self.ema_rate = float(ema_rate)
        self._lock = threading.Lock()
        self.snaps: OrderedDict[str, object] = OrderedDict()
        self.ema: dict[str, float] = {BASE: 0.5}
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
        """Snapshot net by name (KeyError with context when evicted/unknown)."""
        with self._lock:
            try:
                return self.snaps[name]
            except KeyError:
                raise KeyError(f"league snapshot {name!r} not in roster "
                               f"(have {list(self.snaps)})") from None

    def names(self) -> list[str]:
        with self._lock:
            return [BASE, *self.snaps]

    # -- sampling ----------------------------------------------------------
    def _weights_locked(self) -> dict[str, float]:
        return {name: pfsp_weight(self.ema.get(name, 0.5)) for name in [BASE, *self.snaps]}

    def sampling_probs(self) -> dict[str, float]:
        """Absolute per-episode probability of each opponent type (plus `self`)."""
        with self._lock:
            probs = {SELF: self.p_self, YGO: self.p_ygoagent}
            rest = max(0.0, 1.0 - self.p_self - self.p_ygoagent)
            w = self._weights_locked()
            tot = sum(w.values())
            for name, wi in w.items():
                probs[name] = rest * wi / tot if tot > 0 else 0.0
            return probs

    def sample(self) -> str:
        """Draw one episode's opponent type."""
        with self._lock:
            r = float(self.rng.random())
            if r < self.p_self:
                return SELF
            if r < self.p_self + self.p_ygoagent:
                return YGO
            w = self._weights_locked()
            tot = sum(w.values())
            # tot > 0 always (each weight >= 0.05); fall back to base regardless.
            x = float(self.rng.random()) * tot if tot > 0 else 0.0
            for name, wi in w.items():
                x -= wi
                if x <= 0:
                    return name
            return BASE

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
