"""BC training data: replay demo records into observation shards, load them, split by line.

A shard is one `.npz` per record file with the stacked observation arrays (docs/obs_spec.md) plus
`action` (chosen index), `weight` (stated 1.0 / inferred 0.5), `line` (base line id: the variant suffix `#h<k>` and deck
tags stripped, so all hand variants of one line land in the same split), `deck` and `src`. Shards are temporary
training artifacts (AGENTS.md storage rules): rebuild them from the records, never commit them.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np

OBS_KEYS = (
    "cards",
    "card_mask",
    "global",
    "chain",
    "chain_mask",
    "actions",
    "action_mask",
    "history",
    "history_mask",
)

# Option-row columns (docs/action_spec.md): 0 kind, 1 card row, 2 code, 3 effect_code,
# 4 effect_idx, 5 arg, 6 msg, 7 location, 8 controller, 9 step.
PLACE_KIND = 19
_SINGLETON_KINDS = frozenset({11, 17, 18})  # PASS, FINISH, CANCEL: never equivalent to anything else


def equivalent_mask(
    actions: np.ndarray, action: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Bool [64] mask of options equivalent to the chosen option `action`.

    Option j is equivalent to the chosen option a when their action rows are equal on every
    column EXCEPT the ones that only identify a position among identical things: the card-row
    index (col 1) and, for PLACE-kind options (kind 19), the zone/sequence argument (col 5).
    Concretely, j != a is equivalent when: same kind (col 0), same code (col 2), same effect
    code/index (cols 3-4), same msg (col 6), same location (col 7), same controller (col 8),
    same step (col 9), and -- for non-PLACE kinds -- same arg (col 5). The chosen option itself
    is always included. Options of kind PASS (11), FINISH (17) or CANCEL (18) are never marked
    equivalent to anything else (a one-hot of `action` is returned when the choice itself is
    one of those). Illegal slots (per `mask`, when given) are never marked equivalent.
    """
    n = actions.shape[0]
    equiv = np.zeros(n, dtype=bool)
    equiv[action] = True
    a = actions[action]
    kind_a = int(a[0])
    if kind_a in _SINGLETON_KINDS:
        return equiv
    same = (
        (actions[:, 0] == kind_a)
        & (actions[:, 2] == a[2])
        & (actions[:, 3] == a[3])
        & (actions[:, 4] == a[4])
        & (actions[:, 6] == a[6])
        & (actions[:, 7] == a[7])
        & (actions[:, 8] == a[8])
        & (actions[:, 9] == a[9])
    )
    if kind_a != PLACE_KIND:
        same &= actions[:, 5] == a[5]
    for k in _SINGLETON_KINDS:
        same &= actions[:, 0] != k
    if mask is not None:
        same &= np.asarray(mask, dtype=bool)
    same[action] = True  # always include the choice itself, even if masked out
    return same


_ENGINE = None


def _init() -> None:
    global _ENGINE
    from ygo_zero.engine.duel import Engine

    _ENGINE = Engine()


def base_line(line_id: str) -> str:
    return line_id.split("#")[0].split("+")[0]


def _shard_one(job: tuple[str, str, str]) -> tuple[str, int]:
    from ygo_zero.demos.dataset import iter_samples

    src_file, out_file, src_tag = job
    cols: dict[str, list] = {k: [] for k in OBS_KEYS}
    action, weight, line, deck, kind, equiv = [], [], [], [], [], []
    for obs, a, w, meta in iter_samples(_ENGINE, Path(src_file)):
        for k in OBS_KEYS:
            cols[k].append(obs[k])
        equiv.append(equivalent_mask(obs["actions"], a, obs.get("action_mask")))
        action.append(a)
        weight.append(w)
        line.append(base_line(meta["line_id"]))
        deck.append(meta["deck"])
        kind.append(meta.get("kind", "unknown"))
    if not action:
        return out_file, 0
    np.savez_compressed(
        out_file,
        **{k: np.stack(v) for k, v in cols.items()},
        action=np.array(action, dtype=np.int64),
        equiv=np.stack(equiv),
        weight=np.array(weight, dtype=np.float32),
        line=np.array(line),
        deck=np.array(deck),
        kind=np.array(kind),
        src=np.array([src_tag] * len(action)),
    )
    return out_file, len(action)


def build_shards(sources: dict[str, list[Path]], out_dir: Path, jobs: int) -> int:
    """sources: tag -> record files. Writes one shard per file into out_dir; returns the sample count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for tag, files in sources.items():
        for f in files:
            # the full path makes names unique: planned-s1/verified/x and planned-s2/verified/x collided before
            # (later files silently overwrote earlier ones; fixed 2026-10-03)
            rel = Path(f).resolve()
            try:
                rel = rel.relative_to(Path(__file__).resolve().parents[3] / "data" / "demos")
            except ValueError:
                pass
            flat = "~".join(rel.with_suffix("").parts[-6:])
            work.append((str(f), str(out_dir / f"{tag}__{flat}.npz"), tag))
    total = 0
    with mp.get_context("fork").Pool(max(1, jobs), initializer=_init) as pool:
        for out, n in pool.imap_unordered(_shard_one, work):
            total += n
            print(f"{Path(out).name}: {n}", flush=True)
    return total


def load_shards(shard_dir: Path) -> dict[str, np.ndarray]:
    parts = [dict(np.load(p)) for p in sorted(shard_dir.glob("*.npz"))]
    for p in parts:  # backward compat: shards written before kind tags existed
        if "kind" not in p:
            p["kind"] = np.array(["unknown"] * len(p["action"]))
    for p in parts:  # backward compat: shards written before equiv masks existed
        if "equiv" not in p and "actions" in p:
            n = len(p["action"])
            eq = np.zeros((n, p["actions"].shape[1]), dtype=bool)
            eq[np.arange(n), p["action"]] = True
            p["equiv"] = eq
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def split_masks(
    data: dict[str, np.ndarray], val_frac: float = 0.1, holdout_decks: tuple[str, ...] = (),
    kind_val_frac: dict[str, float] | None = None,
) -> dict:
    """Boolean masks train / val / holdout. val = a stable hash of the base line id (all variants together);
    holdout = whole decks never trained on (generalisation to unseen decks). kind_val_frac overrides val_frac for
    samples of the given kinds (e.g. {"gold": 0.2}: more of the scarce hand-played games kept out to evaluate on).
    The hash threshold only grows with the fraction, so a line in val at 0.1 stays in val at 0.2."""
    hold = (
        np.isin(data["deck"], list(holdout_decks)) if holdout_decks else np.zeros(len(data["action"]), bool)
    )
    h = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 1000 for s in data["line"]])
    thr = np.full(len(h), int(val_frac * 1000))
    for kd, f in (kind_val_frac or {}).items():
        thr[data["kind"] == kd] = int(f * 1000)
    val = (h < thr) & ~hold
    return {"train": ~val & ~hold, "val": val, "holdout": hold}
