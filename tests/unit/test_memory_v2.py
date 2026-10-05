"""Memory v2 step A, no engine needed: window transform, shard offsets, hist model."""

import numpy as np
import torch

from ygo_zero.model.bc_data import history_window, load_shards


def _row(kind, code=0, actor_frame_ctrl=0, public=1, turn=1, phase=2, actor=0):
    """One log row: stored option row (10) + actor, turn, phase, public."""
    return [kind, -1, code, 0, 0, 0, 1, 3, actor_frame_ctrl, 0, actor, turn, phase, public]


def test_history_window_transform():
    # kinds: SELECT 15, PASS 11, PLACE 19, ACTIVATE 6; viewer me=0 at turn 5
    log = np.array(
        [
            _row(15, code=100, actor=0, turn=1),  # oldest, salient
            _row(11, actor=0, turn=2),  # PASS: skipped when salient
            _row(6, code=200, actor_frame_ctrl=1, actor=1, public=0, turn=4),  # opp hidden: cols 2-4 zeroed
            _row(19, actor=1, turn=5),  # PLACE: skipped when salient
            _row(6, code=300, actor_frame_ctrl=0, actor=0, turn=5),  # newest
        ],
        np.int32,
    )
    start, end, me, turn = np.array([0]), np.array([5]), np.array([0]), np.array([5])

    h, m = history_window(log, start, end, me, turn, 512, salient=True)
    assert m[0].sum() == 3
    kinds = h[0, :3, 0].tolist()
    assert kinds == [6, 6, 15]  # most recent first, PASS/PLACE gone
    assert h[0, 0, 2] == 300 and h[0, 0, 10] == 0 and h[0, 0, 11] == 0 and h[0, 0, 12] == 2
    hid = h[0, 1]  # opp non-public ACTIVATE: code/effect cols zeroed, ctrl flipped to viewer frame
    assert hid[2] == hid[3] == hid[4] == 0 and hid[10] == 1 and hid[11] == 1 and hid[8] == 0
    assert h[0, 2, 2] == 100 and h[0, 2, 11] == 4  # turns ago
    assert not m[0, 3:].any() and (h[0, 3:] == 0).all()

    h32, m32 = history_window(log, start, end, me, turn, 32, salient=False)
    assert m32[0].sum() == 5 and h32[0, :5, 0].tolist() == [6, 19, 6, 11, 15]

    h2, m2 = history_window(log, start, end, me, turn, 2, salient=True)  # truncation keeps the newest
    assert m2[0].sum() == 2 and h2[0, :, 0].tolist() == [6, 6]

    e2 = history_window(log, np.array([1]), np.array([3]), me, turn, 512, salient=True)  # slice [1,3): PASS + hidden
    assert e2[1][0].sum() == 1 and e2[0][0, 0, 10] == 1


def test_load_shards_offsets_hlog(tmp_path):
    def shard(name, n, rows, start, end):
        np.savez(
            tmp_path / f"{name}.npz",
            action=np.zeros(n, dtype=np.int64),
            hlog=np.arange(rows * 14, dtype=np.int32).reshape(rows, 14),
            hlog_start=np.array(start, dtype=np.int64),
            hlog_end=np.array(end, dtype=np.int64),
            me=np.zeros(n, np.int32),
            turn=np.ones(n, np.int32),
        )

    shard("a", 3, 5, [0, 0, 2], [0, 2, 5])
    shard("b", 2, 4, [0, 1], [1, 4])
    data = load_shards(tmp_path)
    assert len(data["hlog"]) == 9
    assert data["hlog_start"].tolist() == [0, 0, 2, 5, 6]
    assert data["hlog_end"].tolist() == [0, 2, 5, 6, 9]
    assert (data["hlog"][:5] == np.arange(5 * 14, dtype=np.int32).reshape(5, 14)).all()


def _emb(d=8, n=10):
    rng = np.random.default_rng(0)
    return {
        "codes": np.arange(1, n + 1),
        "card": rng.normal(size=(n, d)).astype(np.float32),
        "effects": rng.normal(size=(n, 2, d)).astype(np.float32),
        "n_effects": np.ones(n, dtype=np.int64),
    }


def _batch(b=4, h=128):
    return {
        "cards": torch.zeros(b, 160, 21),
        "card_mask": torch.ones(b, 160, dtype=torch.bool),
        "global": torch.zeros(b, 24),
        "chain": torch.zeros(b, 16, 5),
        "chain_mask": torch.zeros(b, 16, dtype=torch.bool),
        "actions": torch.zeros(b, 64, 10),
        "action_mask": torch.zeros(b, 64, dtype=torch.bool).scatter_(1, torch.tensor([[0, 1]] * b), True),
        "history": torch.zeros(b, h, 13),
        "history_mask": torch.ones(b, h, dtype=torch.bool),
    }


def test_policy_hist0_loads_v1_strict_and_hist128_runs():
    from ygo_zero.model.policy import PolicyNet

    emb = _emb()
    v1 = PolicyNet(emb, d=16, layers=1, heads=2)
    sd = v1.state_dict()
    assert "hist_pos.weight" not in sd and sd["hist_age.weight"].shape[0] == 8
    fresh = PolicyNet(emb, d=16, layers=1, heads=2, hist=0)
    fresh.load_state_dict(sd, strict=True)  # old checkpoints keep working bit-for-bit
    v1.eval()
    fresh.eval()
    with torch.no_grad():
        a = v1(_batch(b=2, h=32))
        c = fresh(_batch(b=2, h=32))
    assert torch.equal(a, c)

    big = PolicyNet(emb, d=16, layers=1, heads=2, hist=128)
    assert big.state_dict()["hist_pos.weight"].shape[0] == 128
    assert big.state_dict()["hist_age.weight"].shape[0] == 32
    big.eval()
    with torch.no_grad():
        out = big(_batch(b=4, h=128))
    assert out.shape == (4, 64) and torch.isfinite(out[:, :2]).all()
