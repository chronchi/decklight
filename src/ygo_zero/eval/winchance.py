"""Calibrated win chance from the value net (web win-chance switch, off by default).

The value net's raw output ``v`` in [-1, 1] (actor's view) is poorly calibrated
early in the game, so the display maps ``p_raw = (v + 1) / 2`` through one
isotonic (PAV) map per turn bucket, fit on scored games
(``scripts/fit_winchance.py``). Outputs are clipped to [0.03, 0.97]: the client
never prints 0% or 100%.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

P_LO, P_HI = 0.03, 0.97
MAX_KNOTS = 50

# Turn buckets: (lo, hi) with hi None meaning open-ended.
# Turns 3 and 4 share one bucket: alone, turn 4 overfit (held-out ECE worse
# after calibration), so the buckets are <=2, 3-4 and 5+.
BUCKETS: tuple[tuple[int, int | None], ...] = ((1, 2), (3, 4), (5, None))
BUCKET_NAMES = ("<=2", "3-4", "5+")


def bucket_of(turn: int) -> int:
    """Turn bucket index for a 1-based game turn."""
    t = int(turn)
    if t <= 2:
        return 0
    if t <= 4:
        return 1
    return 2


def bucket_name(i: int) -> str:
    return BUCKET_NAMES[i]


def pav_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic regression (increasing) by pool-adjacent-violators.

    Returns sorted unique ``xs`` and the fitted non-decreasing values.
    Ties in ``x`` are pooled with weight = multiplicity (block averages).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError(f"pav: len(x)={len(x)} != len(y)={len(y)}")
    if len(x) == 0:
        raise ValueError("pav: no data")
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    # Pool identical xs first (same p_raw is one point with weight = multiplicity).
    ux_all, inv = np.unique(xs, return_inverse=True)
    if len(ux_all) < len(xs):
        sums = np.bincount(inv, weights=ys)
        weights = np.bincount(inv)
        xs = ux_all
        ys = sums / weights
        rep = weights
    else:
        rep = np.ones(len(xs))
    # Blocks of (start, end, sum, weight); merge while decreasing.
    sums: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for i in range(len(xs)):
        sums.append(float(ys[i]) * float(rep[i]))
        weights.append(float(rep[i]))
        starts.append(i)
        ends.append(i + 1)
        while len(sums) >= 2 and sums[-2] / weights[-2] > sums[-1] / weights[-1]:
            s, w, e = sums.pop(), weights.pop(), ends.pop()
            starts.pop()
            sums[-1] += s
            weights[-1] += w
            ends[-1] = e
    ux = np.array([xs[s] for s in starts])
    fit = np.array([s / w for s, w in zip(sums, weights)])
    if bool((np.diff(ux) <= 0).any()) or bool((np.diff(fit) < 0).any()):
        raise AssertionError("pav: fitted map is not monotone")
    return ux, fit


def thin_knots(xs: np.ndarray, ys: np.ndarray, max_knots: int = MAX_KNOTS) -> tuple[list, list]:
    """Evenly subsample knots to at most ``max_knots``, keeping the endpoints."""
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    if len(xs) <= max_knots:
        idx = np.arange(len(xs))
    else:
        idx = np.unique(np.linspace(0, len(xs) - 1, max_knots).round().astype(int))
        if idx[0] != 0:
            idx = np.concatenate([[0], idx])
        if idx[-1] != len(xs) - 1:
            idx = np.concatenate([idx, [len(xs) - 1]])
    tx, ty = xs[idx].tolist(), ys[idx].tolist()
    if any(b < a for a, b in itertools.pairwise(ty)):
        raise AssertionError("thin_knots: map is not monotone")
    return tx, ty


def apply_knots(p: float | np.ndarray, xs: list | np.ndarray, ys: list | np.ndarray) -> float | np.ndarray:
    """Piecewise-linear map with flat extrapolation, clipped to [P_LO, P_HI]."""
    out = np.interp(np.asarray(p, dtype=float), np.asarray(xs, dtype=float), np.asarray(ys, dtype=float),
                    left=float(np.asarray(ys, dtype=float)[0]),
                    right=float(np.asarray(ys, dtype=float)[-1]))
    out = np.clip(out, P_LO, P_HI)
    return float(out) if np.ndim(p) == 0 else out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece_equal_mass(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins (by sorted p)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        raise ValueError("ece: no data")
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    parts = np.array_split(np.arange(len(ps)), bins)
    ece = 0.0
    for part in parts:
        if len(part) == 0:
            continue
        ece += len(part) / len(ps) * abs(float(np.mean(ys[part])) - float(np.mean(ps[part])))
    return ece


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Per-bin (n, mean p, win rate) with equal-mass bins, for the report."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    rows = []
    for part in np.array_split(np.arange(len(ps)), bins):
        if len(part) == 0:
            continue
        rows.append({"n": len(part), "mean_p": round(float(np.mean(ps[part])), 4),
                     "win_rate": round(float(np.mean(ys[part])), 4),
                     "p_lo": round(float(ps[part][0]), 4), "p_hi": round(float(ps[part][-1]), 4)})
    return rows


class CalibMap:
    """The committed calibration: one knot map per turn bucket."""

    def __init__(self, knots: list[tuple[list, list]]) -> None:
        if len(knots) != len(BUCKETS):
            raise ValueError(f"CalibMap: want {len(BUCKETS)} buckets, got {len(knots)}")
        self.knots = [(list(x), list(y)) for x, y in knots]

    def p(self, v: float, turn: int) -> float:
        """Calibrated win chance for raw value ``v`` in [-1, 1] at ``turn``."""
        p_raw = (float(v) + 1.0) / 2.0
        xs, ys = self.knots[bucket_of(turn)]
        return apply_knots(p_raw, xs, ys)  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return {"buckets": BUCKET_NAMES,
                "knots": [{"x": x, "y": y} for x, y in self.knots]}

    @staticmethod
    def from_dict(d: dict) -> CalibMap:
        return CalibMap([(e["x"], e["y"]) for e in d["knots"]])


CALIB_PATH = Path(__file__).resolve().parents[1] / "web" / "winchance_calib.json"


def load_calib(path: str | Path = CALIB_PATH) -> CalibMap:
    """Load the committed calibration map (raises if missing/invalid)."""
    d = json.loads(Path(path).read_text())
    m = CalibMap.from_dict(d["map"])
    for xs, ys in m.knots:
        if not xs or len(xs) != len(ys):
            raise ValueError(f"load_calib: bad knots in {path}")
        if any(b < a for a, b in itertools.pairwise(ys)):
            raise ValueError(f"load_calib: non-monotone map in {path}")
    return m
