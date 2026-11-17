"""Reusable diagnosis package: window filters, probes (permutation/bootstrap), interventions, adapter attention,
report rendering. CPU only, tiny synthetic nets/obs (no model file, no engine)."""

from __future__ import annotations

import numpy as np

from decklight.diag.adapter import AdapterOutput, HookedAdapter
from decklight.diag.attention import attention_table
from decklight.diag.behaviour import ScoredFrame, behaviour_table
from decklight.diag.common import bootstrap_ci, nested_probe, permutation_test, softmax_options
from decklight.diag.interventions import apply_intervention
from decklight.diag.report import build_markdown
from decklight.diag.windows import Window, filter_windows, group_by_start
from decklight.engine.actions import Kind


def _window(wid, group, *, chain=0, phase=0x04, turn=0, kinds=None, cards=()):
    return Window(
        wid=wid, group=group, source="test", line_id=f"line/{group}", k=0,
        start={"config": {}, "prefix": [], "script": [], "our_seat": 1},
        prefix=[], our_seat=1, chain=chain, phase=phase, turn_owner=turn,
        legal_kinds=list(kinds or [int(Kind.ACTIVATE), int(Kind.PASS)]),
        activation_cards=list(cards),
    )  # fmt: skip


def test_filters_and_grouping():
    ws = [
        _window("a", "g1", chain=0, turn=0, phase=0x04),
        _window("b", "g1", chain=1, turn=0, phase=0x04),
        _window("c", "g2", chain=2, turn=1, phase=0x02, kinds=[int(Kind.PASS)], cards=()),
        _window("d", "g2", chain=0, turn=0, phase=0x200),
    ]
    assert {w.wid for w in filter_windows(ws, turn_owner=0)} == {"a", "b", "d"}
    assert {w.wid for w in filter_windows(ws, chain=0)} == {"a", "d"}
    assert {w.wid for w in filter_windows(ws, min_chain=1, max_chain=1)} == {"b"}
    assert {w.wid for w in filter_windows(ws, phase=[0x02, 0x200])} == {"c", "d"}
    assert {w.wid for w in filter_windows(ws, has_activation=False)} == {"c"}
    groups = group_by_start(ws)
    assert set(groups) == {"g1", "g2"} and len(groups["g1"]) == 2


def test_permutation_random_and_planted():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 4)
    n = len(groups)
    xnoise = rng.normal(size=(n, 4))
    random_y = rng.integers(0, 2, n)
    p_random = permutation_test(xnoise, random_y, groups, "auc", 0.01, n_perm=60, seed=1)
    assert p_random["p"] is not None and p_random["p"] > 0.05
    planted = rng.integers(0, 2, n)
    X = np.concatenate([planted[:, None] * 2.0, xnoise], 1)
    p_planted = permutation_test(X, planted, groups, "auc", 0.01, n_perm=60, seed=1)
    assert p_planted["p"] < 0.05


def test_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(50), 3)
    y = rng.integers(0, 2, len(groups))
    X = np.concatenate([y[:, None] * 1.5 + rng.normal(scale=0.5, size=(len(y), 1)), rng.normal(size=(len(y), 3))], 1)
    res = nested_probe(X, y, groups, "auc")
    lo, hi = bootstrap_ci(y, res["pred"], groups, "auc", n_boot=200, seed=3)
    assert lo is not None and hi is not None and lo <= res["score"] <= hi


def _obs():
    n_cards, n_opt, n_chain, n_hist = 5, 4, 2, 4
    cards = np.zeros((n_cards, 21), np.int64)
    cards[:, 0] = [100, 200, 300, 400, 500]
    cards[:, 1] = [2, 2, 3, 3, 5]
    cards[:, 3] = [0, 0, 0, 1, 1]
    cm = np.ones(n_cards, bool)
    glob = np.zeros(24, np.int64)
    glob[20] = 1
    chain = np.zeros((n_chain, 5), np.int64)
    chm = np.array([True, True])
    actions = np.zeros((n_opt, 10), np.int64)
    actions[:, 0] = [int(Kind.ACTIVATE), int(Kind.ACTIVATE), int(Kind.PASS), int(Kind.PASS)]
    actions[:, 1] = [0, 1, -1, -1]
    actions[:, 2] = [100, 200, 0, 0]
    am = np.array([True, True, True, False])
    hist = np.zeros((n_hist, 13), np.int64)
    hist[:, 10] = [0, 0, 1, 1]
    hist[:, 11] = [0, 1, 0, 1]
    hm = np.ones(n_hist, bool)
    return {"cards": cards, "card_mask": cm, "global": glob, "chain": chain, "chain_mask": chm,
            "actions": actions, "action_mask": am, "history": hist, "history_mask": hm}  # fmt: skip


def _tiny_v3():
    from decklight.model.policy_v3 import PolicyNetV3

    rng = np.random.default_rng(0)
    n_cards, slots, dim = 6, 2, 8
    emb = {
        "codes": np.array([0, 100, 200, 300, 400, 500], np.int64),
        "card": rng.normal(size=(n_cards, dim)).astype(np.float32),
        "effects": rng.normal(size=(n_cards, slots, dim)).astype(np.float32),
        "n_effects": np.full(n_cards, slots, np.int64),
        "str_keys": np.array([(100 << 20) | 1], np.int64),
        "str_rows": np.array([0], np.int64),
        "str_vecs": rng.normal(size=(2, dim)).astype(np.float32),
        "lua_keys": np.array([(100 << 20) | 1], np.int64),
        "lua_feats": np.zeros((1, 7), np.float32),
    }
    return PolicyNetV3(emb, d=16, layers=1, heads=2, dropout=0.0)


def test_adapter_attention_rows_sum_to_one():
    adapter = HookedAdapter(_tiny_v3())
    out = adapter.run(_obs())
    assert out.probs.shape == (4,)
    assert out.option_feats.shape[0] == 4 and out.option_feats.ndim == 2
    assert out.attn is not None and out.token_groups is not None
    assert out.attn.shape[0] == 4 and out.attn.shape[1] == len(out.token_groups)
    np.testing.assert_allclose(out.attn.sum(axis=1), 1.0, atol=1e-4)
    assert out.probs[3] == 0.0  # illegal option
    adapter.close()


def test_interventions_change_only_intended_arrays():
    obs = _obs()
    before = {k: np.array(v, copy=True) for k, v in obs.items()}

    new, info = apply_intervention(obs, "history_own_same_turn")
    assert info["masked"] == 1
    assert not new["history_mask"][0]
    assert new["history_mask"][1] and new["history_mask"][2] and new["history_mask"][3]
    for k in obs:
        if k != "history_mask":
            np.testing.assert_array_equal(new[k], before[k])

    new, info = apply_intervention(obs, "chain_1")
    assert new["global"][20] == 1 and before["global"][20] == 1
    new, info = apply_intervention(obs, "chain_0")
    assert new["global"][20] == 0
    for k in obs:
        if k != "global":
            np.testing.assert_array_equal(new[k], before[k])

    new, info = apply_intervention(obs, "card_ctx_none", option=0)
    assert new["actions"][0, 1] == -1 and before["actions"][0, 1] == 0
    np.testing.assert_array_equal(new["actions"][1:, :], before["actions"][1:, :])
    for k in obs:
        if k != "actions":
            np.testing.assert_array_equal(new[k], before[k])
    for k, v in obs.items():  # original untouched
        np.testing.assert_array_equal(v, before[k])


def test_chain_tokens_none_clears_chain_and_global():
    obs = _obs()
    before = {k: np.array(v, copy=True) for k, v in obs.items()}
    new, _ = apply_intervention(obs, "chain_tokens_none")
    assert not new["chain_mask"].any()
    assert new["global"][20] == 0
    assert before["chain_mask"].all() and before["global"][20] == 1
    for k in obs:
        if k not in ("chain_mask", "global"):
            np.testing.assert_array_equal(new[k], before[k])
    for k, v in obs.items():
        np.testing.assert_array_equal(v, before[k])

    obs0 = _obs()
    obs0["global"] = obs0["global"].copy()
    obs0["global"][20] = 0
    before0 = {k: np.array(v, copy=True) for k, v in obs0.items()}
    new0, info0 = apply_intervention(obs0, "chain_tokens_none")
    assert info0.get("noop") is True
    for k in obs0:
        np.testing.assert_array_equal(new0[k], before0[k])


def test_opp_history_now_none_masks_only_opp_same_turn():
    obs = _obs()
    before = {k: np.array(v, copy=True) for k, v in obs.items()}
    new, info = apply_intervention(obs, "opp_history_now_none")
    assert info["masked"] == 1
    assert new["history_mask"].tolist() == [True, True, False, True]
    for k in obs:
        if k != "history_mask":
            np.testing.assert_array_equal(new[k], before[k])
    for k, v in obs.items():
        np.testing.assert_array_equal(v, before[k])

    obs2 = _obs()
    h = obs2["history"].copy()
    h[:, 10] = 0
    obs2["history"] = h
    before2 = {k: np.array(v, copy=True) for k, v in obs2.items()}
    new2, info2 = apply_intervention(obs2, "opp_history_now_none")
    assert info2.get("noop") is True
    for k in obs2:
        np.testing.assert_array_equal(new2[k], before2[k])


def _scored(probs, *, argmax=0, p_act=None, chain=0):
    obs = _obs()
    out = AdapterOutput(probs=np.asarray(probs, float), option_feats=np.zeros((4, 3)),
                        global_feat=np.zeros(3), attn=np.full((4, 6), 1 / 6), token_groups=["g"] * 6)  # fmt: skip
    return ScoredFrame(_window("w", "g", chain=chain), obs, out, float(probs[0]) if p_act is None else p_act,
                       float(probs[2]), argmax, int(obs["actions"][argmax, 0]))  # fmt: skip


def test_behaviour_table_and_report_render():
    frames = [_scored([0.6, 0.2, 0.2, 0.0]), _scored([0.1, 0.1, 0.8, 0.0], argmax=2, chain=1)]
    table = behaviour_table(frames)
    assert table["windows"] == 2
    assert {r["chain"] for r in table["by_chain_turn_phase"]} == {"0", "1"}
    md = build_markdown(
        {"ckpt": "A", "windows": "test"},
        {"behaviour": table, "attention": {"available": False},
         "interventions": {"chain_0": {"n": 1, "delta_p_act_mean": -0.1}}},
    )  # fmt: skip
    assert "# Policy diagnosis" in md and "### Behaviour A" in md and "## Interventions" in md
    assert "not available" in md


def _attn_frame(chain, p_act, g_mass):
    obs = _obs()
    groups = ["g", "h"]
    attn = np.empty((4, len(groups)))
    attn[:, 0] = g_mass
    attn[:, 1] = 1.0 - g_mass
    out = AdapterOutput(
        probs=np.array([0.6, 0.2, 0.2, 0.0]), option_feats=np.zeros((4, 3)),
        global_feat=np.zeros(3), attn=attn, token_groups=groups,
    )  # fmt: skip
    return ScoredFrame(_window("w", f"grp{chain}", chain=chain), obs, out, float(p_act), 0.2, 0, int(Kind.ACTIVATE))


def test_attention_spearman_is_per_chain_bucket():
    frames = []
    for i in range(12):
        frames.append(_attn_frame(0, i, i / 11))
        frames.append(_attn_frame(1, i, 1.0 - i / 11))
    table = attention_table(frames)
    assert table["n_by_chain"] == {"0": 12, "1": 12}
    assert table["spearman_by_chain"]["0"]["g"] == 1.0
    assert table["spearman_by_chain"]["1"]["g"] == -1.0
    md = build_markdown({"ckpt": "A"}, {"attention": table})
    assert "| 0 | g | 12 |" in md and "| 1 | g | 12 |" in md
    assert "| 1.0 |" in md and "| -1.0 |" in md


def test_attention_spearman_none_for_small_or_constant_buckets():
    constant = [_attn_frame(0, i, 0.5) for i in range(12)]
    small = [_attn_frame(2, i, i / 5) for i in range(5)]
    table = attention_table(constant + small)
    assert table["spearman_by_chain"]["0"]["g"] is None
    assert table["spearman_by_chain"]["2+"]["g"] is None


def test_softmax_options_masks_illegal():
    p = softmax_options(np.array([1.0, 1.0, 1.0, 1.0]), np.array([True, True, False, False]))
    assert abs(p.sum() - 1.0) < 1e-9 and p[2] == 0.0 and p[3] == 0.0
