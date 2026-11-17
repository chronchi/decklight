"""Inference-only input edits. Edited inputs can be off-distribution: read the deltas as sensitivity, not as plays.

Each edit reports the change in p(activate) and whether the argmax flipped (and flipped to an activation):
  - `history_own_same_turn`: mask the actor's own history entries from this turn (history_mask);
  - `opp_history_now_none`: mask the opponent's history entries from this turn (history_mask);
  - `chain_tokens_none`: clear every chain token (chain_mask) and global[20] (as if the chain were empty);
  - `chain_1` / `chain_0`: set the chain-length global input (global[20]) to 1 / 0, nothing else
    (global[20] only: models that read chain state from the chain tokens barely move);
  - `card_ctx_none`: set the top activation option's card-row index (actions[:,1]) to -1 (no own card context).
"""

from __future__ import annotations

import numpy as np

from decklight.diag.behaviour import ScoredFrame
from decklight.engine.actions import Kind

KINDS = (
    "history_own_same_turn",
    "opp_history_now_none",
    "chain_tokens_none",
    "chain_1",
    "chain_0",
    "card_ctx_none",
)


def apply_intervention(obs: dict, kind: str, *, option: int | None = None) -> tuple[dict, dict]:
    """Return (new obs, description of what changed). Only the intended arrays are modified."""
    if kind not in KINDS:
        raise ValueError(f"unknown intervention {kind!r} (want one of {KINDS})")
    new = {k: np.array(v, copy=True) for k, v in obs.items()}
    info: dict = {"kind": kind, "changed": []}
    if kind == "history_own_same_turn":
        h = np.asarray(new["history"])
        m = np.asarray(new["history_mask"]).astype(bool)
        own_now = (h[:, 10] == 0) & (h[:, 11] == 0)
        if not own_now.any():
            info["noop"] = True
            return new, info
        m &= ~own_now
        new["history_mask"] = m
        info["changed"] = ["history_mask"]
        info["masked"] = int(own_now.sum())
    elif kind == "opp_history_now_none":
        h = np.asarray(new["history"])
        m = np.asarray(new["history_mask"]).astype(bool)
        opp_now = (h[:, 10] == 1) & (h[:, 11] == 0)
        if not opp_now.any():
            info["noop"] = True
            return new, info
        m &= ~opp_now
        new["history_mask"] = m
        info["changed"] = ["history_mask"]
        info["masked"] = int(opp_now.sum())
    elif kind == "chain_tokens_none":
        if int(np.asarray(obs["global"])[20]) < 1:
            info["noop"] = True
            return new, info
        new["chain_mask"] = np.zeros_like(np.asarray(new["chain_mask"]), dtype=bool)
        new["global"] = np.asarray(new["global"]).copy()
        new["global"][20] = 0
        info["changed"] = ["chain_mask", "global[20]"]
        info["chain_before"] = int(np.asarray(obs["global"])[20])
    elif kind in ("chain_1", "chain_0"):
        new["global"] = np.asarray(new["global"]).copy()
        new["global"][20] = 1 if kind == "chain_1" else 0
        info["changed"] = ["global[20]"]
        info["chain_before"] = int(np.asarray(obs["global"])[20])
    else:
        if option is None:
            raise ValueError("card_ctx_none needs the option index")
        a = np.asarray(new["actions"]).copy()
        before = int(a[option, 1])
        a[option, 1] = -1
        new["actions"] = a
        info["changed"] = [f"actions[{option},1]"]
        info["card_row_before"] = before
    return new, info


def _p_act(probs: np.ndarray, obs: dict) -> float:
    acts = np.asarray(obs["actions"])
    mask = np.asarray(obs["action_mask"]).astype(bool)
    return float(sum(probs[j] for j in np.flatnonzero(mask) if int(acts[j][0]) == int(Kind.ACTIVATE)))


def _new_stat() -> dict:
    return {"n": 0, "delta": [], "flips": 0, "flips_to_activation": 0, "noop": 0}


def _stat_row(kind: str, chain: str, d: dict) -> dict:
    deltas = np.asarray(d["delta"], float)
    return {
        "intervention": kind,
        "chain": chain,
        "n": d["n"],
        "delta_p_act_mean": round(float(deltas.mean()), 4) if len(deltas) else None,
        "delta_p_act_median": round(float(np.median(deltas)), 4) if len(deltas) else None,
        "argmax_flip_share": round(d["flips"] / d["n"], 4) if d["n"] else None,
        "flips_to_activation": d["flips_to_activation"],
        "noop": d["noop"],
    }


def run_interventions(frames: list[ScoredFrame], adapter, *, progress=None) -> list[dict]:
    """Delta p(activate) and argmax flips for every edit, pooled and split by chain bucket (0 / 1+)."""
    pooled: dict[str, dict] = {k: _new_stat() for k in KINDS}
    by_bucket: dict[tuple[str, str], dict] = {(k, b): _new_stat() for k in KINDS for b in ("0", "1+")}
    for i, f in enumerate(frames):
        if progress is not None and (i + 1) % 100 == 0:
            progress(f"interventions {i + 1}/{len(frames)} windows")
        j = f.top_activation()
        if j is None:
            continue
        bucket = "0" if int(np.asarray(f.obs["global"])[20]) == 0 else "1+"
        base_argmax, base_act = f.argmax, f.p_act
        for kind in KINDS:
            obs2, info = apply_intervention(f.obs, kind, option=j)
            stats = (pooled[kind], by_bucket[(kind, bucket)])
            if info.get("noop"):
                for s in stats:
                    s["noop"] += 1
                continue
            out2 = adapter.run(obs2)
            p2 = _p_act(out2.probs, obs2)
            arg2 = int(np.argmax(out2.probs))
            flipped = arg2 != base_argmax
            flipped_to_act = flipped and int(np.asarray(obs2["actions"])[arg2][0]) == int(Kind.ACTIVATE)
            for s in stats:
                s["n"] += 1
                s["delta"].append(p2 - base_act)
                s["flips"] += int(flipped)
                s["flips_to_activation"] += int(flipped_to_act)
    rows = []
    for kind in KINDS:
        rows.append(_stat_row(kind, "all", pooled[kind]))
        for bucket in ("0", "1+"):
            rows.append(_stat_row(kind, bucket, by_bucket[(kind, bucket)]))
    return rows
