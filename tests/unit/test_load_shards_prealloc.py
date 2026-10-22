import numpy as np

from decklight.model import bc_data


def _shard(path, n, nlog, tag):
    rng = np.random.default_rng(n)
    np.savez_compressed(
        path, cards=rng.integers(0, 9, (n, 160, 21), dtype=np.int32), action=np.arange(n, dtype=np.int64),
        equiv=rng.random((n, 64)) < 0.1, kind=np.array([tag * (i % 3 + 1) for i in range(n)]),
        line=np.array([f"l{tag}{i}" for i in range(n)]), hlog=rng.integers(0, 5, (nlog, 14), dtype=np.int32),
        hlog_start=np.zeros(n, np.int64), hlog_end=np.minimum(np.arange(n), nlog).astype(np.int64),
        me=np.zeros(n, np.int32), turn=np.ones(n, np.int32),
    )


def test_prealloc_matches_concatenate(tmp_path):
    _shard(tmp_path / "a.npz", 7, 5, "gold")
    _shard(tmp_path / "b.npz", 4, 9, "x")
    fast = bc_data.load_shards(tmp_path)
    parts = [dict(np.load(p)) for p in sorted(tmp_path.glob("*.npz"))]
    off = len(parts[0]["hlog"])
    assert fast["hlog_start"].tolist() == [0] * 7 + [off] * 4
    assert fast["hlog_end"].tolist() == list(range(5)) + [5, 5] + [off + i for i in range(4)]
    for k in parts[0]:
        if k not in ("hlog_start", "hlog_end"):
            assert np.array_equal(fast[k], np.concatenate([p[k] for p in parts])), k
