import numpy as np

from decklight.eval.loopguard import LoopGuard


def _obs(turn: int, sel: int) -> dict:
    g = np.zeros(24, np.int32)
    g[2] = turn
    cards = np.zeros((4, 3), np.int32)
    cards[0, 0] = sel  # the "selected" flag: SELECT then UNSELECT returns the exact same state
    return {"cards": cards, "card_mask": np.ones(4, bool), "global": g, "chain": np.zeros((2, 5), np.int32),
            "chain_mask": np.zeros(2, bool), "actions": np.zeros((4, 10), np.int32),
            "action_mask": np.array([True, True, True, False])}  # fmt: skip


def _pick(guard: LoopGuard, slot: int, obs: dict, z: np.ndarray) -> int:
    z2, key = guard.mask(slot, obs, z)
    a = int(np.argmax(z2))
    guard.note(slot, key, a)
    return a


def test_cycle_is_broken_and_other_options_are_tried():
    g = LoopGuard()
    z = np.array(
        [3.0, 2.0, 1.0, -np.inf]
    )  # argmax always 0 = "SELECT x", which a deterministic policy repeats
    picks = [_pick(g, 0, _obs(1, 0), z) for _ in range(3)]
    assert picks == [0, 1, 2]
    # every legal option already taken: nothing masked, still a legal pick
    assert _pick(g, 0, _obs(1, 0), z) == 0
    assert g.masked == 2


def test_slots_and_turns_are_independent():
    g = LoopGuard()
    z = np.array([3.0, 2.0, 1.0, -np.inf])
    assert _pick(g, 0, _obs(1, 0), z) == 0
    assert _pick(g, 1, _obs(1, 0), z) == 0  # another env in the same state is not affected
    assert _pick(g, 0, _obs(3, 0), z) == 0  # new turn clears the slot
    assert _pick(g, 0, _obs(3, 1), z) == 0  # a different state is not affected
    g.reset()
    assert _pick(g, 0, _obs(3, 0), z) == 0
    assert g.masked == 0


def test_illegal_options_never_unmasked():
    g = LoopGuard()
    z = np.array([3.0, -np.inf, -np.inf, -np.inf])
    assert [_pick(g, 0, _obs(1, 0), z) for _ in range(3)] == [0, 0, 0]


def test_chain_pass_is_never_masked():
    # Draw Phase: the engine offers the same SELECT_CHAIN again after a pass (NOTEBOOK 2026-10-14 21:30 UTC);
    # masking PASS there forced the activation
    obs = _obs(1, 0)
    obs["global"][21] = 0x10
    obs["actions"][0, 0] = 6  # ACTIVATE
    obs["actions"][1, 0] = 11  # PASS
    obs["action_mask"] = np.array([True, True, False, False])
    g = LoopGuard()
    z = np.array([1.0, 3.0, -np.inf, -np.inf])
    assert [_pick(g, 0, obs, z) for _ in range(3)] == [1, 1, 1]
    # an activation cancelled back to the same window is still masked
    g = LoopGuard()
    z = np.array([3.0, 1.0, -np.inf, -np.inf])
    assert [_pick(g, 0, obs, z) for _ in range(2)] == [0, 1]
