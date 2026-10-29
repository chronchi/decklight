"""Unit tests for the forge group-sequential paired test (numpy only)."""

from itertools import pairwise

import numpy as np

from decklight.forge.seqtest import boundaries, deal_order, decide, obf_constant


def test_obf_constants_and_monotone():
    c6 = obf_constant(6)
    assert 1.98 < c6 < 2.12
    c1 = obf_constant(1)
    assert 1.93 < c1 < 1.99
    b = boundaries(6)
    assert len(b) == 6
    assert all(x > y for x, y in pairwise(b))


def _draw(rng, n, probs):
    vals = np.array([-1.0, 0.0, 1.0])
    return rng.choice(vals, size=n, p=probs)


def test_type_i_error():
    rng = np.random.default_rng(123)
    n_trials, look, K = 4000, 256, 6
    draws = _draw(rng, (n_trials, look * K), [0.13, 0.74, 0.13])
    accepts = sum(
        decide(list(draws[i]), look, K, margin=0.0)["decision"] == "ACCEPT"
        for i in range(n_trials)
    )
    assert accepts / n_trials <= 0.035


def test_power_and_early_stop():
    rng = np.random.default_rng(456)
    n_trials, look, K = 4000, 256, 6
    draws = _draw(rng, (n_trials, look * K), [0.05, 0.75, 0.20])
    looks = []
    accepts = 0
    for i in range(n_trials):
        r = decide(list(draws[i]), look, K, margin=0.03)
        accepts += r["decision"] == "ACCEPT"
        looks.append(r["look"])
    assert accepts / n_trials >= 0.95
    assert float(np.median(looks)) <= 2


def test_futility():
    rng = np.random.default_rng(789)
    n_trials, look, K = 4000, 256, 6
    draws = _draw(rng, (n_trials, look * K), [0.13, 0.74, 0.13])
    looks = []
    rejects = 0
    for i in range(n_trials):
        r = decide(list(draws[i]), look, K, margin=0.03)
        rejects += r["decision"] == "REJECT"
        looks.append(r["look"])
    assert rejects / n_trials >= 0.90
    assert float(np.median(looks)) <= 4


def test_deal_order_and_short():
    keys = [(("B", "A"), 3, 0), (("A", "B"), 1, 1), (("A", "A"), 2, 0)]
    o1 = deal_order(keys, 0)
    assert o1 == deal_order(list(reversed(keys)), 0)
    assert sorted(map(repr, o1)) == sorted(map(repr, keys))
    assert deal_order(keys, 0) == o1
    assert deal_order(keys, 1) != o1 or len(keys) < 2
    r = decide([0.5, -0.5], 256, 6)
    assert r["decision"] == "CONTINUE" and r["look"] == 0 and r["n"] == 2


def test_modest_gain_survives_futility():
    """A real +7 pp pack (above the 3 pp margin) is rarely killed by the non-binding futility stop."""
    rng = np.random.default_rng(321)
    n_trials, look, K = 2000, 256, 6
    draws = _draw(rng, (n_trials, look * K), [0.10, 0.73, 0.17])
    accepts = sum(decide(list(draws[i]), look, K, margin=0.03)["decision"] == "ACCEPT" for i in range(n_trials))
    assert accepts / n_trials >= 0.80
