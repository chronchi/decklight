"""CardText card-dropout (train-only unknown substitution)."""

import numpy as np
import torch

from ygo_zero.model.policy import CardText


def _card_text(d=8, seed=0):
    rng = np.random.default_rng(seed)
    codes = np.array([100, 200, 300])
    card = rng.normal(size=(3, 4)).astype(np.float32)
    effects = rng.normal(size=(3, 2, 4)).astype(np.float32)
    n_effects = np.array([1, 2, 0])
    return CardText(codes, card, effects, n_effects, d)


def test_card_dropout_train_replaces_all():
    ct = _card_text()
    ct.p_unknown = 1.0
    ct.train()
    codes = torch.tensor([100, 200, 300, 999, 0])
    out = ct(codes)
    expected = ct.proj(ct.unknown)
    torch.testing.assert_close(out, expected.expand(len(codes), -1))


def test_card_dropout_eval_unaffected():
    ct = _card_text()
    ct.p_unknown = 1.0
    ct.eval()
    with torch.no_grad():
        out = ct(torch.tensor([100, 200, 300, 999, 0]))
        unknown_proj = ct.proj(ct.unknown)
    # known codes keep their own vectors in eval mode
    assert not torch.allclose(out[0], unknown_proj, atol=1e-6)
    assert not torch.allclose(out[1], unknown_proj, atol=1e-6)
    # missing (999) and hidden (0) codes still map to unknown
    torch.testing.assert_close(out[3], unknown_proj)
    torch.testing.assert_close(out[4], unknown_proj)
