"""Pass-rate curriculum over DAGS starts (PLAN section 0a step 3, Kolibri section 3.2.2.5).

A "position" is a DAGS start (a row of `data/cache/dags/starts.jsonl`): `PassRate`
counts the learner's results (win 1 / draw 0.5 / loss 0, from the learner's seat)
for episodes that began at each start index. Starts the agent sometimes solves
get sampled more; solved (K/K) and hopeless (0/K) starts get only `eps` once
seen `min_n` times, so the PPO batch spends its samples on trainable positions.

Weight per start: `w = p(1-p) + eps` with `p = (wins + 1) / (n + 2)` (a Beta(1,1)
prior: unseen starts get the prior's weight 0.25 + eps, so they are explored).
With `--curric-drop-solved`, samples from episodes whose start has `n >= min_n`
and `p-hat` in {0, 1} exactly get loss weight 0 (their advantage is pure noise;
implemented as exclusion in `train_rl`, see `solved_mask`).
"""

from __future__ import annotations

import json
from pathlib import Path


class PassRate:
    """Per-start win/draw/loss counts and sampling weights (JSON-serializable)."""

    def __init__(self, n_starts: int, eps: float = 0.02, min_n: int = 4,
                 hint_bonus: dict[int, float] | None = None) -> None:
        if n_starts <= 0:
            raise ValueError(f"n_starts={n_starts} must be > 0")
        if not eps >= 0.0:
            raise ValueError(f"eps={eps} must be >= 0")
        self.n_starts = int(n_starts)
        self.eps = float(eps)
        self.min_n = int(min_n)
        self.wins = [0.0] * self.n_starts
        self.n = [0] * self.n_starts
        # Optional per-start weight bonus (e.g. hint-bank starts via --hint-start-bonus).
        self.hint_bonus = {int(k): float(v) for k, v in (hint_bonus or {}).items()}

    def record(self, start: int, result: float) -> None:
        """One finished learner episode that began at `start` (1 win / 0.5 draw / 0 loss)."""
        if not 0 <= start < self.n_starts:
            raise ValueError(f"start {start} out of range [0, {self.n_starts})")
        if result not in (0.0, 0.5, 1.0):
            raise ValueError(f"result {result!r} must be 0.0, 0.5 or 1.0")
        self.wins[start] += float(result)
        self.n[start] += 1

    def p_hat(self, start: int) -> float:
        """Posterior-mean pass rate with the Beta(1,1) prior (0.5 when unseen)."""
        return (self.wins[start] + 1.0) / (self.n[start] + 2.0)

    def is_solved(self, start: int) -> bool:
        """Seen `min_n` times and won (or lost) every game: zero-variance group."""
        if self.n[start] < self.min_n:
            return False
        return self.wins[start] == self.n[start] or self.wins[start] == 0.0

    def weight(self, start: int) -> float:
        """Sampling weight for one start (variance + eps, eps-only when solved)."""
        if self.is_solved(start):
            w = self.eps
        else:
            p = self.p_hat(start)
            w = p * (1.0 - p) + self.eps
        return w + self.hint_bonus.get(start, 0.0)

    def weights(self) -> list[float]:
        """One weight per start (all > 0 when eps > 0 or a bonus applies)."""
        return [self.weight(s) for s in range(self.n_starts)]

    def start_weights(self) -> list[float]:
        """Per-start weights when slots are (start, seat) pairs (slot = 2*start + seat): the C++
        VecEnv draws a start, not a seat, so a start gets the larger of its two seats' weights."""
        if self.n_starts % 2:
            raise ValueError(f"start_weights needs an even slot count, got {self.n_starts}")
        w = self.weights()
        return [max(w[2 * k], w[2 * k + 1]) for k in range(self.n_starts // 2)]

    def solved_mask(self, starts: list[int]) -> list[bool]:
        """Which episode-start indices belong to a solved/hopeless group (drop these)."""
        return [s >= 0 and self.is_solved(s) for s in starts]

    def stats(self) -> dict:
        """One-line curriculum summary (logged per push): seen/trainable/solved shares."""
        seen = sum(1 for c in self.n if c > 0)
        trainable = sum(1 for s in range(self.n_starts) if self.n[s] > 0 and not self.is_solved(s))
        solved = sum(1 for s in range(self.n_starts) if self.is_solved(s))
        exact_solved = sum(1 for s in range(self.n_starts)
                           if self.n[s] >= self.min_n and self.wins[s] == self.n[s])
        hopeless = solved - exact_solved
        w = self.weights()
        tot = sum(w)
        # Effective sample size of the weights (sum w)^2 / sum w^2), as a share of n_starts.
        ess = tot * tot / sum(x * x for x in w) if tot > 0 else 0.0
        return {
            "starts": self.n_starts, "seen": seen,
            "trainable": trainable, "solved": exact_solved, "hopeless": hopeless,
            "ess_share": round(ess / self.n_starts, 3) if self.n_starts else 0.0,
        }

    def to_dict(self) -> dict:
        return {"n_starts": self.n_starts, "eps": self.eps, "min_n": self.min_n,
                "wins": self.wins, "n": self.n,
                "hint_bonus": {str(k): v for k, v in self.hint_bonus.items()}}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "PassRate":
        d = json.loads(Path(path).read_text())
        self = cls(d["n_starts"], eps=d.get("eps", 0.02), min_n=d.get("min_n", 4),
                   hint_bonus={int(k): v for k, v in d.get("hint_bonus", {}).items()})
        if len(d["wins"]) != self.n_starts or len(d["n"]) != self.n_starts:
            raise ValueError(f"curriculum file {path} has n_starts={d['n_starts']} "
                             f"but count rows of length {len(d['wins'])}/{len(d['n'])}")
        self.wins = [float(x) for x in d["wins"]]
        self.n = [int(x) for x in d["n"]]
        return self
