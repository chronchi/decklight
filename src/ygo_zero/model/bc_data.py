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
    action, weight, line, deck = [], [], [], []
    for obs, a, w, meta in iter_samples(_ENGINE, Path(src_file)):
        for k in OBS_KEYS:
            cols[k].append(obs[k])
        action.append(a)
        weight.append(w)
        line.append(base_line(meta["line_id"]))
        deck.append(meta["line_id"].split("/")[0])
    if not action:
        return out_file, 0
    np.savez_compressed(
        out_file,
        **{k: np.stack(v) for k, v in cols.items()},
        action=np.array(action, dtype=np.int64),
        weight=np.array(weight, dtype=np.float32),
        line=np.array(line),
        deck=np.array(deck),
        src=np.array([src_tag] * len(action)),
    )
    return out_file, len(action)


def build_shards(sources: dict[str, list[Path]], out_dir: Path, jobs: int) -> int:
    """sources: tag -> record files. Writes one shard per file into out_dir; returns the sample count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for tag, files in sources.items():
        for f in files:
            work.append((str(f), str(out_dir / f"{tag}__{f.parent.name}__{f.stem}.npz"), tag))
    total = 0
    with mp.get_context("fork").Pool(max(1, jobs), initializer=_init) as pool:
        for out, n in pool.imap_unordered(_shard_one, work):
            total += n
            print(f"{Path(out).name}: {n}", flush=True)
    return total


def load_shards(shard_dir: Path) -> dict[str, np.ndarray]:
    parts = [dict(np.load(p)) for p in sorted(shard_dir.glob("*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def split_masks(
    data: dict[str, np.ndarray], val_frac: float = 0.1, holdout_decks: tuple[str, ...] = ()
) -> dict:
    """Boolean masks train / val / holdout. val = a stable hash of the base line id (all variants together);
    holdout = whole decks never trained on (generalisation to unseen decks)."""
    hold = (
        np.isin(data["deck"], list(holdout_decks)) if holdout_decks else np.zeros(len(data["action"]), bool)
    )
    h = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 1000 for s in data["line"]])
    val = (h < int(val_frac * 1000)) & ~hold
    return {"train": ~val & ~hold, "val": val, "holdout": hold}
