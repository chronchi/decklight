"""Honest-override rule for search (retrain plan v2/v3 stage S; NOTEBOOK 2026-10-11 05:10).

hppo played the candidate with the best mean rollout score. Only 22% of its overrides had a paired gap >= 2 SE, and
38.7% picked a candidate that was the same card as the policy's pick (zones, identical copies): the optimizer's curse
over noisy means. This module gives search two fixes, both purely statistical (no play knowledge):

1. **Merge semantically identical candidates** (`merge_candidates`): two options with the same meaning (an
   identical copy, an equivalent zone) are one candidate, so the search never "chooses" between them.
2. **Override only on a paired, significant gap** (`decide`): candidates share worlds per sample (common random
   worlds), so the per-sample difference challenger − pick is paired. The challenger replaces the policy's pick
   only if mean(d) >= max(z * SE(d), min_gap). Undecided cases get more samples for the two candidates, up to
   `s_max` (adaptive sampling); the rest keep the policy's move.

`decisive_label` is the same test against the runner-up instead of the pick (stage L: a training label is emitted
only where the winner is significantly better than every other candidate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.search.determinize import option_key


@dataclass(frozen=True)
class HonestRule:
    z: float = 2.0  # SE multiple a gap must reach
    min_gap: float = 0.05  # absolute floor (value units, [-1, 1]); blocks tiny-SE near-deterministic rollouts
    s_max: int = 8  # max samples per candidate after adaptive escalation (s_min = WindowSearch.samples)
    step: int = 4  # samples added per escalation round (to the two compared candidates only)
    merge: bool = True  # merge semantically identical candidates


def _own_link_on_field(obs) -> bool:
    cards, mask = obs["cards"], obs["card_mask"].astype(bool)
    rows = cards[mask]
    own_m = rows[(rows[:, 1] == 3) & (rows[:, 3] == 0)]
    return bool(np.any(own_m[:, 7] & C.TYPE_LINK))


def semantic_key(obs, i: int, links: bool | None = None) -> tuple:
    """Meaning of option i for merging: `option_key` (kind, code, desc, arg, msg, loc, ctrl), except PLACE options,
    where every non-center main-monster zone is one key and every S/T zone is one key while no own Link monster is on
    the field (the rule of `demos/verify.py interchangeable()`; the center zone stays distinct)."""
    row = obs["actions"][int(i)]
    if int(row[0]) == Kind.PLACE:
        if links is None:
            links = _own_link_on_field(obs)
        if not links:
            zn = int(row[5])
            zk = "mz" if zn <= 4 and zn != 2 else "sz" if 8 <= zn <= 12 else zn
            return (Kind.PLACE, int(row[8]), zk)
    return option_key(row)


def merge_candidates(order: list[int], obs) -> list[int]:
    """Keep the first option of each semantic key, in `order`'s order."""
    links = _own_link_on_field(obs)
    seen: set = set()
    out = []
    for i in order:
        k = semantic_key(obs, i, links)
        if k in seen:
            continue
        seen.add(k)
        out.append(int(i))
    return out


def paired_gap(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """Mean and SE of a[s] - b[s] over the common samples (sample s of both shares one world). SE = inf for n < 2."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, math.inf, 0
    d = np.asarray(a[:n], dtype=np.float64) - np.asarray(b[:n], dtype=np.float64)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n >= 2 else math.inf
    return m, se, n


def significant(m: float, se: float, rule: HonestRule) -> bool:
    return m >= max(rule.z * se, rule.min_gap)


@dataclass
class Decision:
    chosen_j: int  # candidate index played (pick_j unless a significant override)
    challenger_j: int | None  # best other candidate by mean
    gap: float  # mean(challenger - pick), paired
    se: float
    n: int  # paired samples behind the gap
    status: str  # "keep" (challenger not better), "override", "undecided" (gap > 0, not significant)


def decide(samples: list[list[float]], pick_j: int, rule: HonestRule) -> Decision:
    """Override decision for one searched state. `samples[j]` = per-sample scores of candidate j (from `me`'s view)."""
    means = [float(np.mean(s)) if s else -math.inf for s in samples]
    others = [j for j in range(len(samples)) if j != pick_j and samples[j]]
    if not others:
        return Decision(pick_j, None, 0.0, math.inf, 0, "keep")
    c = max(others, key=lambda j: (means[j], -j))
    m, se, n = paired_gap(samples[c], samples[pick_j])
    if m <= 0:
        return Decision(pick_j, c, m, se, n, "keep")
    if significant(m, se, rule):
        return Decision(c, c, m, se, n, "override")
    return Decision(pick_j, c, m, se, n, "undecided")


def decide_pair(samples: list[list[float]], pick_j: int, c: int, rule: HonestRule) -> Decision:
    """`decide` with the challenger fixed (adaptive rounds keep comparing the same two candidates, so their sample
    lists stay aligned by sample index)."""
    m, se, n = paired_gap(samples[c], samples[pick_j])
    if m <= 0:
        return Decision(pick_j, c, m, se, n, "keep")
    if significant(m, se, rule):
        return Decision(c, c, m, se, n, "override")
    return Decision(pick_j, c, m, se, n, "undecided")


def decisive_label(samples: list[list[float]], rule: HonestRule) -> Decision:
    """Stage L label: the best candidate by mean, if it beats the runner-up significantly (status "label");
    otherwise status "undecided" (no label). `challenger_j` is the runner-up."""
    means = [float(np.mean(s)) if s else -math.inf for s in samples]
    live = [j for j in range(len(samples)) if samples[j]]
    if len(live) < 2:
        return Decision(live[0] if live else 0, None, 0.0, math.inf, 0, "undecided")
    order = sorted(live, key=lambda j: (-means[j], j))
    best, second = order[0], order[1]
    m, se, n = paired_gap(samples[best], samples[second])
    status = "label" if significant(m, se, rule) else "undecided"
    return Decision(best, second, m, se, n, status)
