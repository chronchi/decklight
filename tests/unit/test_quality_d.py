"""Task D: equivalent-choice labels (equiv masks + set-target loss)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ygo_zero.model.bc_data import equivalent_mask, load_shards

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(kind, card=-1, code=0, ec=0, ei=0, arg=0, msg=0, loc=0, ctrl=-1, step=0):
    return [kind, card, code, ec, ei, arg, msg, loc, ctrl, step]


def _actions(rows):
    a = np.zeros((64, 10), dtype=np.int64)
    for i, r in enumerate(rows):
        a[i] = r
    return a


def test_same_code_different_row_equivalent():
    # two SELECT options for copies of the same card from the same place
    acts = _actions([
        _row(15, card=3, code=89631139, arg=7, msg=20, loc=2, ctrl=0),
        _row(15, card=5, code=89631139, arg=7, msg=20, loc=2, ctrl=0),
    ])
    mask = np.zeros(64, bool)
    mask[:2] = True
    eq = equivalent_mask(acts, 0, mask)
    assert eq[0] and eq[1]
    assert eq.sum() == 2


def test_different_code_not_equivalent():
    acts = _actions([
        _row(15, card=3, code=89631139, arg=7, msg=20, loc=2, ctrl=0),
        _row(15, card=5, code=12345678, arg=7, msg=20, loc=2, ctrl=0),
    ])
    mask = np.zeros(64, bool)
    mask[:2] = True
    eq = equivalent_mask(acts, 0, mask)
    assert eq[0] and not eq[1]


def test_place_zones_equivalent_but_arg_matters_otherwise():
    place = _actions([
        _row(19, card=-1, code=0, msg=18, arg=0),
        _row(19, card=-1, code=0, msg=18, arg=1),
    ])
    mask = np.zeros(64, bool)
    mask[:2] = True
    eq = equivalent_mask(place, 0, mask)
    assert eq[0] and eq[1]
    # non-PLACE: same rows except arg differ -> not equivalent
    sel = _actions([
        _row(15, card=3, code=89631139, arg=7, msg=20, loc=2, ctrl=0),
        _row(15, card=3, code=89631139, arg=9, msg=20, loc=2, ctrl=0),
    ])
    eq = equivalent_mask(sel, 0, mask)
    assert eq[0] and not eq[1]


def test_pass_finish_cancel_never_equivalent():
    acts = _actions([
        _row(11, msg=30),  # PASS chosen
        _row(11, msg=30),  # identical PASS row
    ])
    mask = np.zeros(64, bool)
    mask[:2] = True
    eq = equivalent_mask(acts, 0, mask)
    assert eq[0] and not eq[1] and eq.sum() == 1
    for kind in (17, 18):  # FINISH, CANCEL
        acts = _actions([_row(kind), _row(kind)])
        eq = equivalent_mask(acts, 0, mask)
        assert eq.sum() == 1 and eq[0]
    # a PASS candidate is never equivalent to a non-PASS choice either
    acts = _actions([_row(12, msg=10), _row(11, msg=10)])
    eq = equivalent_mask(acts, 0, mask)
    assert eq[0] and not eq[1]


def test_chosen_always_included_and_illegal_excluded():
    acts = _actions([
        _row(15, card=3, code=1, arg=7, msg=20, loc=2, ctrl=0),
        _row(15, card=5, code=1, arg=7, msg=20, loc=2, ctrl=0),  # equivalent but illegal
    ])
    mask = np.zeros(64, bool)
    mask[0] = True
    eq = equivalent_mask(acts, 0, mask)
    assert eq[0] and not eq[1]


def test_set_loss_equals_plain_ce_when_one_hot():
    train_bc = _load("train_bc")
    torch.manual_seed(0)
    logits = torch.randn(4, 64)
    logits[:, 10:] = float("-inf")  # only 10 legal options
    act = torch.tensor([0, 3, 5, 9])
    eq = torch.zeros(4, 64, dtype=torch.bool)
    eq[torch.arange(4), act] = True
    got = train_bc.set_loss(logits, eq)
    want = torch.nn.functional.cross_entropy(logits, act, reduction="none")
    torch.testing.assert_close(got, want)


def test_set_loss_with_two_equiv_options():
    train_bc = _load("train_bc")
    logits = torch.tensor([[1.0, 2.0, 0.5] + [float("-inf")] * 61])
    eq = torch.zeros(1, 64, dtype=torch.bool)
    eq[0, 0] = eq[0, 1] = True
    got = train_bc.set_loss(logits, eq).item()
    import math

    want = -math.log(math.exp(1.0) + math.exp(2.0)) + math.log(
        math.exp(1.0) + math.exp(2.0) + math.exp(0.5)
    )
    assert got == pytest.approx(want, abs=1e-6)


def test_load_shards_without_equiv_gives_one_hot(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    np.savez_compressed(
        d / "a.npz", actions=np.zeros((2, 64, 10), dtype=np.int64),
        action=np.array([0, 5]), weight=np.array([1.0, 0.5]),
        line=np.array(["l0", "l1"]), deck=np.array(["d0", "d0"]), src=np.array(["g", "g"]),
    )
    data = load_shards(d)
    assert data["equiv"].shape == (2, 64)
    assert data["equiv"][0].sum() == 1 and data["equiv"][0, 0]
    assert data["equiv"][1].sum() == 1 and data["equiv"][1, 5]
