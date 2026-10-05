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


# --- memory v2 step B: card-knowledge tracker, shards, known model (no engine needed) ---

def _move(code, fc, fl, fs, fp, tc, tl, ts, tp=0xA):
    import struct

    from ygo_zero.engine import constants as C

    return bytes([C.MSG_MOVE]) + struct.pack("<I", code) + bytes([fc, fl]) + struct.pack(
        "<II", fs, fp) + bytes([tc, tl]) + struct.pack("<II", ts, tp) + struct.pack("<I", 0)


def _draw(player, codes, positions=None):
    import struct

    from ygo_zero.engine import constants as C

    positions = positions if positions is not None else [0x2] * len(codes)
    out = bytes([C.MSG_DRAW, player]) + struct.pack("<I", len(codes))
    for c, p in zip(codes, positions, strict=True):
        out += struct.pack("<II", c, p)
    return out


def _confirm(viewer, entries):
    import struct

    from ygo_zero.engine import constants as C

    out = bytes([C.MSG_CONFIRM_CARDS, viewer]) + struct.pack("<I", len(entries))
    for code, con, loc, seq in entries:
        out += struct.pack("<I", code) + bytes([con, loc]) + struct.pack("<I", seq)
    return out


def _shuffle(player, codes):
    import struct

    from ygo_zero.engine import constants as C

    out = bytes([C.MSG_SHUFFLE_HAND, player]) + struct.pack("<I", len(codes))
    for c in codes:
        out += struct.pack("<I", c)
    return out


def test_hand_memory_messages():
    from ygo_zero.engine import constants as C
    from ygo_zero.engine.knowledge import HandMemory

    m = HandMemory()
    assert m.note(_draw(0, [101, 102]))  # opening draw: unknown
    assert m.known == [[False, False], []]
    # search: Deck -> hand (still unknown), then confirmed to player 1
    assert m.note(_move(103, 0, C.LOCATION_DECK, 30, 0x8, 0, C.LOCATION_HAND, 2, 0xA))
    assert m.known[0] == [False, False, False] and m.codes[0][2] == 103
    assert m.note(_confirm(1, [(103, 0, C.LOCATION_HAND, 2)]))
    assert m.known[0] == [False, False, True]
    # confirm-to-self does not count
    assert m.note(_confirm(0, [(101, 0, C.LOCATION_HAND, 0)]))
    assert m.known[0] == [False, False, True]
    # bounce: face-up field -> hand is known at once; face-down is not
    assert m.note(_move(104, 0, C.LOCATION_MZONE, 1, 0x1, 0, C.LOCATION_HAND, 3, 0xA))
    assert m.note(_move(105, 0, C.LOCATION_SZONE, 0, 0x8, 0, C.LOCATION_HAND, 4, 0xA))
    assert m.known[0] == [False, False, True, True, False]
    # leaving drops the flag and shifts later seqs down
    assert m.note(_move(103, 0, C.LOCATION_HAND, 2, 0xA, 0, C.LOCATION_MZONE, 4, 0x1))
    assert m.known[0] == [False, False, True, False]
    # GY -> hand is public; banished face-down -> hand is not
    assert m.note(_move(106, 0, C.LOCATION_GRAVE, 5, 0x5, 0, C.LOCATION_HAND, 4, 0xA))
    assert m.note(_move(107, 0, C.LOCATION_REMOVED, 0, 0xA, 0, C.LOCATION_HAND, 5, 0xA))
    assert m.known[0] == [False, False, True, False, True, False]
    # shuffle re-aligns by code: the known 106 moves with its code
    assert m.note(_shuffle(0, [107, 101, 102, 104, 105, 106]))
    assert m.known[0] == [False, False, False, True, False, True]
    assert m.codes[0] == [107, 101, 102, 104, 105, 106]
    # malformed messages are ignored, not fatal
    assert m.note(bytes([C.MSG_MOVE, 1, 2, 3]))
    assert m.note(b"")
    assert m.known[0] == [False, False, False, True, False, True]
    # resync guard rebuilds a drifted list as unknown and reports False
    assert not m.guard((5, 3))
    assert m.known == [[False] * 5, [False] * 3]
    assert m.guard((5, 3))


def test_hand_memory_setup_hands():
    from ygo_zero.engine.knowledge import HandMemory

    m = HandMemory(([111, 112], [113]))
    assert m.known == [[False, False], [False]] and m.codes == [[111, 112], [113]]


def test_load_shards_known_mixed_raises_and_absent_ok(tmp_path):
    import numpy as np

    from ygo_zero.model.bc_data import load_shards

    def shard(name, known):
        d = {"action": np.zeros(2, dtype=np.int64)}
        if known:
            d["card_known"] = np.zeros((2, 160), np.uint8)
            d["known_code"] = np.zeros((2, 160), np.int32)
        np.savez(tmp_path / f"{name}.npz", **d)

    shard("a", True)
    shard("b", False)
    try:
        load_shards(tmp_path)
    except ValueError as e:
        assert "card_known" in str(e)
    else:
        raise AssertionError("mixed shards with/without card_known must raise")
    (tmp_path / "a.npz").unlink()
    data = load_shards(tmp_path)  # uniform without the keys: tolerated
    assert "card_known" not in data


def test_policy_known_false_loads_step_a_strict_and_known_runs():
    from ygo_zero.model.policy import PolicyNet

    emb = _emb()
    step_a = PolicyNet(emb, d=16, layers=1, heads=2, hist=128)
    sd = step_a.state_dict()
    assert "known_emb.weight" not in sd
    fresh = PolicyNet(emb, d=16, layers=1, heads=2, hist=128, known=False)
    fresh.load_state_dict(sd, strict=True)  # step-A checkpoints keep working bit-for-bit
    step_a.eval()
    fresh.eval()
    with torch.no_grad():
        assert torch.equal(step_a(_batch(b=2, h=128)), fresh(_batch(b=2, h=128)))

    assert "known_emb.weight" in PolicyNet(emb, d=16, layers=1, heads=2, known=True).state_dict()
    big = PolicyNet(emb, d=16, layers=1, heads=2, hist=128, known=True)
    big.eval()
    batch = _batch(b=2, h=128)
    batch["card_known"] = torch.zeros(2, 160, dtype=torch.long)
    batch["known_code"] = torch.zeros(2, 160, dtype=torch.long)
    batch["card_known"][0, 3] = 1
    batch["known_code"][0, 3] = 5  # a remembered code must change the card token, not stay "hidden"
    with torch.no_grad():
        out = big(batch)
    assert out.shape == (2, 64) and torch.isfinite(out[:, :2]).all()
    batch["card_known"][0, 3] = 0
    batch["known_code"][0, 3] = 0
    with torch.no_grad():
        out2 = big(batch)
    assert not torch.equal(out, out2)

