"""Hinted self-distillation bank (PLAN section 0a step 3, OPSD section 3.2.1.4).

We have no hint-conditioned policy, so "hinted -> unhinted" honestly means: the
unhinted policy is fine-tuned on successes found with the search's help
(`scripts/hint_bank.py` banks the actor's decisions from hinted games won on
hard DAGS starts). This module loads such a bank for the RL auxiliary loss:
`C x CE(policy(obs), banked action)` on a minibatch of banked decisions per PPO
update (see `train_rl --hint-bank/--hint-coef`).

Storage is replays only (config + full choice list + per-decision actor flags),
~1 KB/game; observations are regenerated here by replaying through the engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BANK_VERSION = 1


def _row_config(row: dict):
    """A bank row's DuelConfig (same dict shape as `eval.turn1.config_to_dict`)."""
    from ygo_zero.eval.turn1 import config_from_dict

    return config_from_dict(row["config"])


def replay_rows(engine, rows: list[dict], hist: int, known: bool,
                keys: tuple[str, ...] | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Regenerate (obs, banked action) for every actor decision in `rows`.

    For each row, replays config + choices[:k] for every k where the actor moved;
    fails loud (ValueError) when a prefix does not replay or a banked action is
    illegal there. Returns (obs dict with the model's input keys, int32 actions).
    """
    from ygo_zero.eval.turn1 import replay_prefix
    from ygo_zero.rl.ppo import model_keys

    # model_keys already holds action_mask: dedupe (a duplicate key appended each mask twice, so the masks came out
    # 2N long and misaligned with every other input; found 2026-10-08 before the bank's first training use)
    want = tuple(dict.fromkeys(tuple(keys if keys is not None else model_keys(hist, known)) + ("action_mask",)))
    acc: dict[str, list] = {k: [] for k in want}
    acts: list[int] = []
    for ri, row in enumerate(rows):
        cfg = _row_config(row)
        choices = [int(c) for c in row["choices"]]
        actors = [int(a) for a in row["actors"]]
        actor = int(row["actor_seat"])
        if not (len(choices) == len(actors)):
            raise ValueError(f"bank row {ri}: {len(choices)} choices but {len(actors)} actors")
        # Decisions inside the DAGS prefix come from the source game, not from the search: never distil them.
        skip = len(row.get("prefix") or [])
        if skip > len(choices):
            raise ValueError(f"bank row {ri}: prefix ({skip}) longer than the game ({len(choices)} choices)")
        # One pass per row (same env progression as replay_prefix(choices[:k]) for every k, without the
        # quadratic re-replays: 190 rows took minutes that way).
        env, obs = replay_prefix(engine, cfg, [])
        for k, a in enumerate(choices):
            if obs is None:
                raise ValueError(f"bank row {ri}: game ended at decision {k} of {len(choices)}")
            if actors[k] == actor and k >= skip:
                n_opts = int(obs["action_mask"].sum())
                if not 0 <= a < n_opts or not bool(obs["action_mask"][a]):
                    raise ValueError(f"bank row {ri} decision {k}: banked action {a} illegal "
                                     f"at replay ({n_opts} options)")
                for key in want:
                    acc[key].append(np.array(obs[key], copy=True))
                acts.append(a)
            obs = env.step(a)
    if not acts:
        raise ValueError("bank holds 0 actor decisions")
    out = {k: np.stack(v) for k, v in acc.items()}
    bad = {k: len(v) for k, v in out.items() if len(v) != len(acts)}
    if bad:
        raise ValueError(f"replayed inputs misaligned with {len(acts)} actions: {bad}")
    return out, np.array(acts, dtype=np.int32)


def load_bank(path: str | Path, engine, hist: int, known: bool,
              keys: tuple[str, ...] | None = None) -> dict:
    """Load + replay a hint bank file. Fails loud on empty/missing/illegal banks.

    Returns {"obs": {key: (N, ...) int32/bool}, "act": (N,) int32, "rows": R,
    "start_ids": sorted list of banked start ids, "n_decisions": N}.
    """
    path = Path(path)
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except OSError as e:
        raise ValueError(f"cannot read hint bank {path}: {e}") from e
    if not rows:
        raise ValueError(f"hint bank {path} is empty (0 banked decisions)")
    for ri, row in enumerate(rows):
        for key in ("config", "choices", "actors", "actor_seat"):
            if key not in row:
                raise ValueError(f"hint bank {path} row {ri} misses {key!r}")
    obs, act = replay_rows(engine, rows, hist, known, keys)
    return {"obs": obs, "act": act, "rows": len(rows),
            "start_ids": sorted({int(r.get("start_id", -1)) for r in rows if int(r.get("start_id", -1)) >= 0}),
            "n_decisions": len(act)}
