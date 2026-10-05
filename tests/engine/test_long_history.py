"""Memory v2 step A: the shard log rebuilds exactly the env's history windows.

On replayed gold records, for every sample, `history_window` over the record's final log sliced at the
sample's `hlog_end` must equal the env obs: salient H=512 == `hist_long`, unfiltered H=32 == `history`.
"""

from pathlib import Path

import numpy as np

from ygo_zero.demos.dataset import iter_samples
from ygo_zero.model.bc_data import history_window

GOLD = Path(__file__).resolve().parents[2] / "data" / "demos" / "gold"


def _first_records(engine, path: Path, n_records: int = 3):
    """(obs, meta) of the first `n_records` replayable records of one file (whole records only)."""
    out, keys = [], []
    for obs, a, w, meta in iter_samples(engine, path):
        key = (meta["line_id"], meta["seed"])
        if key not in keys:
            if len(keys) == n_records:
                break
            keys.append(key)
        out.append((obs, meta))
    assert keys, f"no record replayed in {path.name}"
    return out


def test_history_window_matches_env_obs(engine):
    files = sorted(GOLD.glob("*.jsonl"))
    assert files, "no gold records"
    checked, salient_n = 0, 0
    for path in files[:2]:
        for obs, meta in _first_records(engine, path):
            log = np.asarray(meta["hlog"], dtype=np.int32)
            assert log.shape[1] == 14
            assert 0 <= meta["hlog_end"] <= len(log)
            kw = {
                "hlog": log, "start": np.array([0]), "end": np.array([meta["hlog_end"]]),
                "me": np.array([meta["me"]]), "turn": np.array([meta["turn"]]),
            }
            h512, m512 = history_window(**kw, H=512, salient=True)
            assert np.array_equal(h512[0], obs["hist_long"]), f"{path.name} k={meta['k']}"
            assert np.array_equal(m512[0], obs["hist_long_mask"])
            h32, m32 = history_window(**kw, H=32, salient=False)
            assert np.array_equal(h32[0], obs["history"]), f"{path.name} k={meta['k']}"
            assert np.array_equal(m32[0], obs["history_mask"])
            checked += 1
            salient_n += int(m512[0].sum())
    assert checked > 0, "no samples checked"
    print(f"checked {checked} samples, mean salient entries {salient_n / checked:.1f}")


def test_history_log_grows_and_bounds_windows(engine):
    """The log only grows; hist_long holds the salient subset of history (exact mask lengths)."""
    path = min(GOLD.glob("*.jsonl"))
    prev = -1
    for obs, meta in _first_records(engine, path, n_records=1):
        log = np.asarray(meta["hlog"], dtype=np.int32)
        assert meta["hlog_end"] >= prev
        prev = meta["hlog_end"]
        kinds_all = log[: meta["hlog_end"], 0]
        kinds_sal = kinds_all[~np.isin(kinds_all, [11, 19])]
        assert int(obs["hist_long_mask"].sum()) == min(len(kinds_sal), 512)
        assert int(obs["history_mask"].sum()) == min(len(kinds_all), 32)
