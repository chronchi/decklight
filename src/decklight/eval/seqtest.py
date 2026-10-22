"""Pre-registered sequential paired test (Wald SPRT on discordant deals).

Pure functions shared by ``scripts/audit/seq_test.py``. No engine, no torch.

Pairing mirrors ``scripts/audit/paired_sm.py``: a deal is a
``(tuple(decks), seed, s_seat)`` key, each side scores 1 win / 0 loss /
0.5 truncated from the scored side's perspective, and the paired value is
the B-minus-A diff in ``{-1, -0.5, 0, +0.5, +1}``.

Discordance convention (documented for the pre-registration): exact ties
(diff 0) are dropped from the SPRT. A half-diff (+/-0.5, one side
truncated and the other decisive) counts as half a discordance: only the
sign of B-A enters the win/loss direction, and the LLR increment is
weighted by ``|diff|`` (0.5). The ``discord`` and ``b_wins`` counters
accumulate the same fractional weights, so ``llr == b_wins * lw +
(discord - b_wins) * ll`` exactly.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from typing import Any

DEFAULT_D = 0.158
ONE_SIDED_Z = 1.6448536269514722  # 95% one-sided (truncation fallback)
TWO_SIDED_Z = 1.959963984540054  # 95% two-sided (reporting)


def p1_from_delta(delta_pp: float, d: float = DEFAULT_D) -> float:
    """Discordant-win probability under H1 for a paired mean effect."""
    return 0.5 + (delta_pp / 100.0) / (2.0 * d)


def p0_from_margin(margin_pp: float, d: float = DEFAULT_D) -> float:
    """Null discordant-win probability for a non-inferiority margin."""
    return 0.5 - (margin_pp / 100.0) / (2.0 * d)


def llr_win(p0: float, p1: float) -> float:
    return math.log(p1 / p0)


def llr_loss(p0: float, p1: float) -> float:
    return math.log((1.0 - p1) / (1.0 - p0))


def bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Wald bounds: (upper A for H1, lower B for H0)."""
    return math.log((1.0 - beta) / alpha), math.log(beta / (1.0 - alpha))


def resolve_params(prereg: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and derive (p0, p1) from a pre-registration dict."""
    mode = prereg.get("mode", "superiority")
    if mode not in ("superiority", "noninferiority"):
        raise ValueError(f"unknown mode {mode!r}")
    d = float(prereg.get("d", DEFAULT_D))
    delta_pp = float(prereg.get("delta_pp", 2))
    margin_pp = float(prereg.get("margin_pp", 2))
    if "p1" in prereg:
        p1 = float(prereg["p1"])
    elif mode == "superiority":
        p1 = p1_from_delta(delta_pp, d)
    else:
        p1 = 0.5
    if "p0" in prereg:
        p0 = float(prereg["p0"])
    elif mode == "superiority":
        p0 = 0.5
    else:
        p0 = p0_from_margin(margin_pp, d)
    if not 0.0 < p0 < 1.0 or not 0.0 < p1 < 1.0:
        raise ValueError(f"p0/p1 out of range: p0={p0} p1={p1}")
    return {
        "mode": mode,
        "p0": p0,
        "p1": p1,
        "alpha": float(prereg.get("alpha", 0.05)),
        "beta": float(prereg.get("beta", 0.2)),
        "block": int(prereg.get("block", 250)),
        "max_deals": int(prereg.get("max_deals", 2500)),
        "order_seed": int(prereg.get("order_seed", 0)),
        "d": d,
        "delta_pp": delta_pp,
        "margin_pp": margin_pp,
        # The fallback-CI threshold is -margin only for non-inferiority;
        # superiority always tests against 0.
        "margin": margin_pp / 100.0 if mode == "noninferiority" else 0.0,
    }


def _key_hash(order_seed: int, key: Hashable) -> bytes:
    return hashlib.sha256(f"{order_seed}:{key!r}".encode()).digest()


def ordered_keys(keys: Sequence[Hashable], order_seed: int) -> list[Hashable]:
    """Order pair keys by the hash of (order_seed, key).

    Deterministic and independent of when each game finished.
    """
    return sorted(keys, key=lambda k: _key_hash(order_seed, k))


def block_keys(prereg: dict[str, Any], keys: Sequence[Hashable]) -> list[list[Hashable]]:
    """Chunk the hash-ordered ``keys`` into blocks of ``prereg['block']``."""
    block = int(prereg.get("block", 250))
    ordered = ordered_keys(keys, int(prereg.get("order_seed", 0)))
    return [ordered[i : i + block] for i in range(0, len(ordered), block)]


def paired_ci(diffs: Sequence[float], z: float = TWO_SIDED_Z) -> tuple[float, float, float, float]:
    """Mean, (lo, hi) CI and SE of paired diffs (fractions, not pp)."""
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = sum(diffs) / n
    if n < 2:
        return m, m, m, 0.0
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - z * se, m + z * se, se


def _truncation_pass(
    diffs: Sequence[float], threshold: float, z: float = ONE_SIDED_Z
) -> tuple[bool, float, float, float, float]:
    """One-sided fallback at truncation: lower ``z`` bound > threshold."""
    m, _, _, se = paired_ci(diffs)
    lower = m - z * se
    return lower > threshold, m, lower, m + z * se, se


def run_sequential(
    diffs_ordered: Sequence[float],
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
    block: int,
    max_deals: int,
    margin: float = 0.0,
) -> dict[str, Any]:
    """Walk the SPRT over schedule-ordered paired diffs.

    Looks happen only after each complete ``block`` of deals. ``max_deals``
    truncates the walk: the decision then falls back to the fixed one-sided
    CI rule (lower 95% bound of the paired diff above 0, or above -margin
    for non-inferiority). Fewer available deals than a decision needs
    returns ``continue``.
    """
    a_upper, b_lower = bounds(alpha, beta)
    lw, ll = llr_win(p0, p1), llr_loss(p0, p1)
    rows: list[dict[str, Any]] = []
    llr = 0.0
    discord = 0.0
    b_wins = 0.0
    diffs = list(diffs_ordered)[:max_deals]
    n_blocks = len(diffs) // block
    decision = "continue"
    for b in range(1, n_blocks + 1):
        chunk = diffs[(b - 1) * block : b * block]
        for x in chunk:
            w = abs(x)
            if w == 0.0:
                continue
            discord += w
            if x > 0:
                b_wins += w
                llr += w * lw
            else:
                llr += w * ll
        deals = b * block
        if llr >= a_upper:
            blk = "accept_H1"
        elif llr <= b_lower:
            blk = "accept_H0"
        else:
            blk = "continue"
        rows.append(
            {
                "deals": deals,
                "discord": discord,
                "b_wins": b_wins,
                "llr": llr,
                "upper": a_upper,
                "lower": b_lower,
                "decision": blk,
            }
        )
        if blk != "continue":
            decision = blk
            diffs = diffs[:deals]
            break
    deals_used = len(rows) * block
    if decision == "continue" and len(diffs) >= max_deals:
        tail = diffs[:max_deals]
        passed, m, lower, upper1, se = _truncation_pass(tail, -margin if margin else 0.0)
        decision = "truncated_pass" if passed else "truncated_fail"
        deals_used = max_deals
        m2, lo2, hi2, _ = paired_ci(tail)
    else:
        m2, lo2, hi2, _ = paired_ci(diffs)
        m, se = m2, (hi2 - lo2) / (2 * TWO_SIDED_Z) if len(diffs) >= 2 else 0.0
        lower, upper1 = m - ONE_SIDED_Z * se, m + ONE_SIDED_Z * se
    return {
        "rows": rows,
        "decision": decision,
        "deals_used": deals_used if decision != "continue" else (len(rows) * block),
        "deals_available": len(diffs_ordered),
        "discord": discord,
        "b_wins": b_wins,
        "llr": llr,
        "upper": a_upper,
        "lower": b_lower,
        "mean": m2,
        "lo95": lo2,
        "hi95": hi2,
        "mean_1sided_lo": lower,
        "mean_1sided_hi": upper1,
    }


def simulate(
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
    block: int,
    max_deals: int,
    d: float = DEFAULT_D,
    deltas_pp: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    n_runs: int = 20000,
    seed: int = 0,
    margin: float = 0.0,
) -> dict[float, dict[str, float]]:
    """Monte Carlo operating characteristics (numpy, vectorized over runs).

    Each deal is discordant w.p. ``d``; conditional on discordance B wins
    w.p. ``p_true = 0.5 + delta/(2d)``. This models full-discordance deals
    (no halves); halves only make the real walk slightly more conservative.
    Returns per true delta: P(accept H1, early or via truncation), the
    average stopping deals (ASN), and P(truncation, i.e. no early stop).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    a_upper, b_lower = bounds(alpha, beta)
    lw, ll = llr_win(p0, p1), llr_loss(p0, p1)
    n_blocks = max_deals // block
    out: dict[float, dict[str, float]] = {}
    chunk = 2500  # runs per chunk: keeps float64 cumsum (~chunk*max_deals) small
    for delta in deltas_pp:
        p_true = 0.5 + (delta / 100.0) / (2.0 * d)
        p_true = min(1.0, max(0.0, p_true))
        n_hi = n_trunc = 0
        deals_total = 0.0
        done = 0
        while done < n_runs:
            m_runs = min(chunk, n_runs - done)
            idx = np.arange(m_runs)
            disc = rng.random((m_runs, max_deals)) < d
            wins = rng.random((m_runs, max_deals)) < p_true
            incr = np.where(disc, np.where(wins, lw, ll), 0.0)
            cum = np.cumsum(incr, axis=1)
            grid = cum[:, block - 1 :: block][:, :n_blocks]
            hi_hit = grid >= a_upper
            lo_hit = grid <= b_lower
            hit = hi_hit | lo_hit
            ever = hit.any(axis=1)
            first = np.where(ever, hit.argmax(axis=1) + 1, n_blocks)
            accept_h1_early = np.where(ever, hi_hit[idx, first - 1], False)
            # Truncation fallback: one-sided CI on the full paired diffs.
            x = np.where(disc, np.where(wins, 1.0, -1.0), 0.0)
            mean = x.mean(axis=1)
            sd = x.std(axis=1, ddof=1)
            se = sd / math.sqrt(max_deals)
            trunc_pass = (mean - ONE_SIDED_Z * se) > (-margin if margin else 0.0)
            n_hi += int((accept_h1_early | (~ever & trunc_pass)).sum())
            n_trunc += int((~ever).sum())
            deals_total += float(np.where(ever, first * block, max_deals).sum())
            done += m_runs
        out[float(delta)] = {
            "p_accept_h1": n_hi / n_runs,
            "asn": deals_total / n_runs,
            "p_trunc": n_trunc / n_runs,
        }
    return out
