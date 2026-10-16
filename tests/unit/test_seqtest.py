"""Unit tests for the pre-registered sequential paired test (speed plan M1)."""

import math

from ygo_zero.eval.seqtest import (
    block_keys,
    bounds,
    llr_loss,
    llr_win,
    ordered_keys,
    p1_from_delta,
    resolve_params,
    run_sequential,
    simulate,
)


def test_bounds_formula():
    upper, lower = bounds(0.05, 0.2)
    assert upper == math.log(0.8 / 0.05)
    assert lower == math.log(0.2 / 0.95)
    assert abs(upper - 2.7725887) < 1e-6
    assert abs(lower + 1.5581446) < 1e-6


def test_default_prereg_derives_p1():
    params = resolve_params({"mode": "superiority", "delta_pp": 2, "d": 0.158})
    assert abs(params["p1"] - 0.5632911) < 1e-4
    assert params["p0"] == 0.5
    ni = resolve_params({"mode": "noninferiority", "margin_pp": 2, "d": 0.158})
    assert ni["p1"] == 0.5
    assert abs(ni["p0"] - (0.5 - 0.02 / (2 * 0.158))) < 1e-12
    assert p1_from_delta(2, 0.158) == params["p1"]


def test_all_win_stream_accepts_h1_at_first_look_past_bound():
    # 250 straight B wins: LLR = 250 * ln(0.563/0.5) = 29.7 >> 2.77.
    res = run_sequential([1.0] * 250, 0.5, 0.5632911, 0.05, 0.2, 250, 2500)
    assert res["decision"] == "accept_H1"
    assert len(res["rows"]) == 1
    assert res["rows"][0]["deals"] == 250
    assert res["discord"] == 250
    assert res["b_wins"] == 250


def test_ties_ignored():
    # 249 ties then one block of wins still accepts; ties add no LLR.
    res = run_sequential([0.0] * 249 + [1.0] * 251, 0.5, 0.5632911, 0.05, 0.2, 250, 2500)
    assert res["rows"][0]["discord"] == 1.0
    assert res["decision"] == "accept_H1"
    assert res["deals_used"] == 500


def test_half_discordance_counts_half():
    lw = llr_win(0.5, 0.5632911)
    res = run_sequential([0.5] * 500, 0.5, 0.5632911, 0.05, 0.2, 250, 2500)
    assert res["rows"][0]["discord"] == 125.0
    assert res["rows"][0]["b_wins"] == 125.0
    assert abs(res["rows"][0]["llr"] - 125.0 * lw) < 1e-9


def test_block_gating_no_early_peek():
    # A single B win in the first 249 deals crosses nothing; the walk only
    # looks at multiples of `block`. With block=250 and 251 wins available
    # but max_deals cut before the second look, no early decision leaks.
    diffs = [1.0] + [0.0] * 249
    res = run_sequential(diffs, 0.5, 0.9, 0.05, 0.2, 250, 2500)
    assert len(res["rows"]) == 1  # exactly one look at 250 deals
    assert res["rows"][0]["deals"] == 250
    # Partial trailing block (10 extra deals) never triggers a look.
    res2 = run_sequential(diffs + [1.0] * 10, 0.5, 0.5632911, 0.05, 0.2, 250, 2500)
    assert len(res2["rows"]) == 1
    assert res2["deals_used"] == 250


def test_all_loss_stream_accepts_h0():
    res = run_sequential([-1.0] * 250, 0.5, 0.5632911, 0.05, 0.2, 250, 2500)
    assert res["decision"] == "accept_H0"


def test_noninferiority_sign():
    # Under non-inferiority H0 (p0 < 0.5), a stream of B losses is evidence
    # *for* H1 = non-inferior... no: losses mean B worse, evidence for H0.
    # Wins (B better than a handicapped null) accept H1.
    ni = resolve_params({"mode": "noninferiority", "margin_pp": 2, "d": 0.158})
    assert ni["p0"] < 0.5 < ni["p1"] or ni["p0"] < ni["p1"] == 0.5
    win = run_sequential([1.0] * 250, ni["p0"], ni["p1"], 0.05, 0.2, 250, 2500)
    assert win["decision"] == "accept_H1"
    loss = run_sequential([-1.0] * 250, ni["p0"], ni["p1"], 0.05, 0.2, 250, 2500)
    assert loss["decision"] == "accept_H0"
    # A null stream at p=0.5 favours H1 (non-inferior): losses carry less
    # weight than wins because p0 < 0.5.
    assert llr_win(ni["p0"], ni["p1"]) > 0 > llr_loss(ni["p0"], ni["p1"])
    assert abs(llr_win(ni["p0"], ni["p1"])) > abs(llr_loss(ni["p0"], ni["p1"]))


def test_block_keys_deterministic_and_complete():
    keys = [(("A", "B"), s, seat) for s in range(600) for seat in (0, 1)]
    prereg = {"block": 250, "order_seed": 7}
    blocks = block_keys(prereg, keys)
    assert [len(b) for b in blocks] == [250, 250, 250, 250, 200]
    flat = [k for b in blocks for k in b]
    assert sorted(map(repr, flat)) == sorted(map(repr, keys))
    assert block_keys(prereg, keys) == blocks  # deterministic
    assert block_keys({"block": 250, "order_seed": 8}, keys) != blocks
    assert ordered_keys(keys, 7) == flat


def test_truncation_fallback_pass_and_fail():
    # Flat zeros at max_deals: mean 0, lower bound 0 -> not > 0 -> fail.
    res = run_sequential([0.0] * 500, 0.5, 0.5632911, 0.05, 0.2, 250, 500)
    assert res["decision"] == "truncated_fail"
    # Sub-boundary but consistent evidence: LLR stays inside (-1.56, 2.77)
    # at every look, yet the fixed CI at truncation is clearly positive.
    half = [1.0] * 20 + [-1.0] * 10 + [0.0] * 220
    res = run_sequential(half + half, 0.5, 0.5632911, 0.05, 0.2, 250, 500)
    assert len(res["rows"]) == 2
    assert all(r["decision"] == "continue" for r in res["rows"])
    assert res["decision"] == "truncated_pass"


def test_simulate_type_i_error_controlled():
    # Loose regression guard: the procedure's true alpha is ~0.052 (100k-run
    # estimate; early SPRT 0.022 + truncation-fallback inflation), so 0.06
    # leaves 5 SE of Monte Carlo headroom at 20k runs. The pre-registered
    # 0.055 bar is checked by the --simulate acceptance measurement.
    table = simulate(0.5, p1_from_delta(2, 0.158), 0.05, 0.2, 250, 2500,
                     d=0.158, deltas_pp=[0.0], n_runs=20000, seed=12345)
    assert table[0.0]["p_accept_h1"] <= 0.06


def test_simulate_power_at_plus2pp():
    table = simulate(0.5, p1_from_delta(2, 0.158), 0.05, 0.2, 250, 2500,
                     d=0.158, deltas_pp=[2.0], n_runs=20000, seed=12345)
    assert table[2.0]["p_accept_h1"] >= 0.75
