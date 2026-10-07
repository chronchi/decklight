"""Privileged critic for RL v2 part B (PLAN Phase 5, step 5 / "oracle guiding").

In imperfect-information games a critic that sees hidden information gives much
lower-variance advantages, while the policy stays honest. The model already has a
"remembered hidden card" input: with `known=True`, `PolicyNet` reads `card_known`
(uint8 [160]) and `known_code` (int32 [160]), aligned with `cards` rows. The
privileged critic is a separate `PolicyNet(known=True, value=True)` fed an obs
where every hidden opponent row is marked `card_known = 1` with its true code.

The engine exposes the true codes as a new obs array `priv_code` int32 [160]
(aligned with `cards` rows; the true code on rows hidden from the viewer, else 0).
The policy never reads it: `gather(..., priv=False)` (the default) does not return
it and `bc_data.OBS_KEYS` / `KNOWN_KEYS` are unchanged.
"""

from __future__ import annotations

import numpy as np


def critic_obs(policy_ob: dict[str, np.ndarray], priv_code: np.ndarray) -> dict[str, np.ndarray]:
    """Critic input dict (numpy) from a gathered policy obs plus its `priv_code` rows.

    `card_known' = where(priv_code > 0 & card_known == 0, 1, card_known)`,
    `known_code' = where(priv_code > 0 & card_known == 0, priv_code, known_code)`;
    every other key is as the policy sees it. Missing `card_known`/`known_code`
    (policy built with `known=False`) are treated as all-zero. `priv_code` itself
    is dropped: the critic reads the revealed identity through the known inputs.
    """
    out = {k: v for k, v in policy_ob.items() if k != "priv_code"}
    n, m = out["cards"].shape[:2]
    ck = out.get("card_known")
    kc = out.get("known_code")
    ck = np.zeros((n, m), np.uint8) if ck is None else np.asarray(ck)
    kc = np.zeros((n, m), np.int32) if kc is None else np.asarray(kc)
    pc = np.asarray(priv_code).reshape(n, m)
    reveal = (pc > 0) & (ck == 0)
    out["card_known"] = np.where(reveal, np.uint8(1), ck).astype(np.uint8)
    out["known_code"] = np.where(reveal, pc, kc).astype(np.int32)
    return out
