"""Group-sequential paired test for forge pack acceptance.

Paired differences d_i = specialist score - release score per deal are examined
at looks n_k = k * look (k = 1..K). Efficacy uses an O'Brien-Fleming-shaped
boundary b_k = c * sqrt(K / k) with c calibrated by Monte Carlo so the one-sided
H0 crossing probability equals alpha under the canonical Brownian approximation.
Futility (REJECT) is non-binding with its own fixed bound: m_k + futility_z * se_k < margin (default
futility_z = 1.96). The O'Brien-Fleming b_k is too wide at early looks to stop a useless pack before the end
(NOTEBOOK 2026-10-29 "forge-seqtest"); a non-binding futility stop never raises the type-I error.

Pure Python/numpy: no torch, no engine.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import numpy as np

_N_PATHS = 400_000
_BISECT_TOL = 1e-3


@functools.lru_cache(maxsize=None)  # noqa: UP033 -- brief specifies lru_cache
def obf_constant(K: int, alpha: float = 0.025) -> float:
    """O'Brien-Fleming constant c for K looks at one-sided level alpha.

    c is chosen so that under H0, P(z_k >= c * sqrt(K / k) for some k) = alpha,
    under the canonical approximation z_k = W(t_k) / sqrt(t_k), t_k = k / K.
    Computed by Monte Carlo with numpy default_rng(0) and bisection to 1e-3.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rng = np.random.default_rng(0)
    incr = rng.normal(0.0, math.sqrt(1.0 / K), size=(_N_PATHS, K))
    paths = np.cumsum(incr, axis=1)
    maxima = np.max(paths, axis=1)
    lo, hi = 0.0, 5.0
    while float(np.mean(maxima >= hi)) > alpha:
        hi *= 2.0
    while hi - lo > _BISECT_TOL:
        mid = 0.5 * (lo + hi)
        if float(np.mean(maxima >= mid)) >= alpha:
            lo = mid
        else:
            hi = mid
    return hi


def boundaries(K: int, alpha: float = 0.025) -> list[float]:
    """Per-look efficacy boundaries b_k = c * sqrt(K / k), k = 1..K."""
    c = obf_constant(K, alpha)
    return [c * math.sqrt(K / k) for k in range(1, K + 1)]


def _look_stats(diffs: Sequence[float], n: int, b: float) -> dict:
    arr = np.asarray(diffs[:n], dtype=float)
    m = float(np.mean(arr))
    if n < 2:
        s = 0.0
    else:
        s = float(np.std(arr, ddof=1))
        if math.isnan(s):
            s = 0.0
    se = s / math.sqrt(n) if n > 0 else 0.0
    if se == 0.0:
        z = math.inf if m > 0 else (-math.inf if m < 0 else 0.0)
    else:
        z = m / se
    return {
        "n": n,
        "mean_pp": m * 100.0,
        "se_pp": se * 100.0,
        "z": z,
        "boundary": b,
        "rci_pp": [(m - b * se) * 100.0, (m + b * se) * 100.0],
    }


def decide(
    diffs: Sequence[float],
    look: int,
    K: int,
    margin: float = 0.03,
    alpha: float = 0.025,
    futility_z: float = 1.96,
) -> dict:
    """Evaluate looks 1..k_max (k_max = min(K, len(diffs) // look)) sequentially.

    Stops at the first non-CONTINUE verdict. ACCEPT needs both z_k >= b_k and
    m_k >= margin; REJECT on futility (m_k + futility_z * se_k below margin) or,
    at k = K, on anything not accepted.
    """
    n_total = len(diffs)
    bounds = boundaries(K, alpha)
    k_max = min(K, n_total // look) if look > 0 else 0
    if k_max < 1:
        base = _look_stats(diffs, n_total, bounds[0]) if n_total else {
            "n": 0, "mean_pp": 0.0, "se_pp": 0.0, "z": 0.0,
            "boundary": bounds[0], "rci_pp": [0.0, 0.0],
        }
        return {
            "decision": "CONTINUE", "look": 0, **base, "looks": [],
        }
    rows: list[dict] = []
    for k in range(1, k_max + 1):
        n_k = k * look
        b_k = bounds[k - 1]
        st = _look_stats(diffs, n_k, b_k)
        m = st["mean_pp"] / 100.0
        se = st["se_pp"] / 100.0
        z = st["z"]
        if z >= b_k and m >= margin:
            verdict = "ACCEPT"
        elif m + futility_z * se < margin:  # noqa: SIM114 -- keep spec steps separate
            verdict = "REJECT"
        elif k == K:
            verdict = "REJECT"
        else:
            verdict = "CONTINUE"
        row = {"look": k, "decision": verdict, **st}
        rows.append(row)
        if verdict != "CONTINUE":
            return {
                "decision": verdict, "look": k, **st, "looks": rows,
            }
    last = rows[-1]
    return {
        "decision": "CONTINUE", "look": last["look"], "n": last["n"],
        "mean_pp": last["mean_pp"], "se_pp": last["se_pp"], "z": last["z"],
        "boundary": last["boundary"], "rci_pp": last["rci_pp"], "looks": rows,
    }


def deal_order(keys: Sequence, seed: int) -> list:
    """Deterministic permutation of sorted(keys) via numpy default_rng(seed)."""
    ordered = sorted(keys)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    return [ordered[i] for i in perm]
