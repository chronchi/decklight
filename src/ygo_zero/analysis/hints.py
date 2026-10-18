""" "Ask the bot" hints for practice mode (web-ux-plan step 4).

A hint is the agent's top-3 options for the current human decision, with the
policy probability ("bot's confidence") of each and, when the win-chance
switch is on, each option's one-step-lookahead estimate. It is computed only
from the human's observation (the same fairness rule as the review:
``review.human_obs_dict`` for the value net, the pending decision's own obs
for the policy — at a human prompt that obs is the human seat's).
"""

from __future__ import annotations

import time

import numpy as np


def compute_hint(
    engine, cfg, env, human_seat: int, launcher, calib=None, *, top_n: int = 3, max_steps: int | None = None
) -> dict:
    """Top-N hint options for the env's current human decision.

    Returns ``{"options": [{"index", "p", "est"|None}], "ms": float}`` ordered
    by policy probability descending. ``est`` is present only when ``calib``
    is given (the win-chance switch is on) and the launcher has a value net.
    Raises ``ValueError`` when there is no human decision to hint on.
    """
    from ygo_zero.analysis import review as R

    if env.done or env.player != human_seat or len(env.opts) <= 1:
        raise ValueError("no human decision to hint on")
    t0 = time.monotonic()
    obs = env._observe()
    probs = R._policy_probs(launcher, obs)
    legal = [int(i) for i in np.flatnonzero(obs["action_mask"])]
    order = sorted(legal, key=lambda j: (-float(probs[j]), int(j)))
    top = [int(j) for j in order[: max(1, top_n)]]
    est_map: dict[int, float] = {}
    if calib is not None and getattr(launcher, "value", None) is not None:
        kw = {} if max_steps is None else {"max_steps": max_steps}
        est = R.estimate_candidates(engine, cfg, list(env.choices), top, human_seat, launcher, calib, **kw)
        for idx, (p, _turn) in zip(top, est, strict=True):
            est_map[int(idx)] = float(p)
    ms = (time.monotonic() - t0) * 1000.0
    return {
        "options": [
            {
                "index": int(i),
                "p": round(float(probs[int(i)]), 4),
                **({"est": round(float(est_map[int(i)]), 4)} if int(i) in est_map else {"est": None}),
            }
            for i in top
        ],
        "ms": round(ms, 1),
    }
