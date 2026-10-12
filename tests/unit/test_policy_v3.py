import numpy as np
import torch

from ygo_zero.engine.actions import Kind
from ygo_zero.model.policy import PolicyNet
from ygo_zero.model.policy_v3 import PolicyNetV3, lua_bits, make_policy, phase_index


def _emb(d=8, n=10):
    rng = np.random.default_rng(0)
    keys = np.array(sorted([(3 << 20) | 0, (3 << 20) | 1, (0 << 20) | 2, (5 << 20) | 0]), dtype=np.int64)
    return {
        "codes": np.arange(1, n + 1),
        "card": rng.normal(size=(n, d)).astype(np.float32),
        "effects": rng.normal(size=(n, 3, d)).astype(np.float16),
        "n_effects": np.array([0, 1, 2, 3, 3, 3, 1, 1, 2, 2], dtype=np.int8),
        "str_keys": keys,
        "str_rows": np.array([0, 1, 2, 2], dtype=np.int32),
        "str_vecs": rng.normal(size=(3, d)).astype(np.float16),
        "lua_keys": np.array([(3 << 20) | 0, (3 << 20) | 1], dtype=np.int64),
        "lua_feats": np.array([[0x100, 0x4000, 2, 0, 1, 1, 15, 1], [0x40, 1 << 33, 4, 0, 0, 0, 2, 1]], np.int64),
    }


def _batch(b=3, h=32):
    o = {
        "cards": torch.zeros(b, 160, 21, dtype=torch.int32),
        "card_mask": torch.zeros(b, 160, dtype=torch.bool),
        "global": torch.zeros(b, 24, dtype=torch.int32),
        "chain": torch.zeros(b, 16, 5, dtype=torch.int32),
        "chain_mask": torch.zeros(b, 16, dtype=torch.bool),
        "actions": torch.zeros(b, 64, 10, dtype=torch.int32),
        "action_mask": torch.zeros(b, 64, dtype=torch.bool),
        "history": torch.zeros(b, h, 13, dtype=torch.int32),
        "history_mask": torch.zeros(b, h, dtype=torch.bool),
    }
    o["cards"][:, :5, 0] = torch.tensor([1, 3, 5, 7, 99])
    o["card_mask"][:, :5] = True
    o["global"][:, 0], o["global"][:, 1], o["global"][:, 2], o["global"][:, 3] = 8000, 3500, 3, 0x4
    o["global"][:, 21] = 11
    # options: activate card 3 effect 0 (row 1), effect 1, a summon, and pass
    o["actions"][:, 0] = torch.tensor([int(Kind.ACTIVATE), 1, 3, 3, 0, 0, 11, 2, 0, 0])
    o["actions"][:, 1] = torch.tensor([int(Kind.ACTIVATE), 1, 3, 3, 1, 0, 11, 2, 0, 0])
    o["actions"][:, 2] = torch.tensor([int(Kind.SUMMON), 2, 5, 0, 0, 0, 11, 2, 0, 0])
    o["actions"][:, 3] = torch.tensor([int(Kind.TO_END), -1, 0, 0, 0, 0, 11, 0, 0, 0])
    o["action_mask"][:, :4] = True
    o["chain"][:, 0] = torch.tensor([1, 3, 3, 0, 1])
    o["chain_mask"][:, 0] = True
    return o


def _net(**kw):
    torch.manual_seed(0)
    return PolicyNetV3(_emb(), d=16, layers=4, heads=2, dropout=0.0, **kw).eval()


def test_forward_shapes_and_masking():
    net = _net()
    lg = net(_batch())
    assert lg.shape == (3, 64)
    assert torch.isfinite(lg[:, :4]).all() and torch.isinf(lg[:, 4:]).all()


def test_trim_does_not_change_logits():
    net, obs = _net(), _batch()
    a = net(obs)
    net.trim = False
    assert torch.allclose(a, net(obs), atol=1e-5)


def test_activation_counts_and_normal_summon_reach_the_logits():
    net, obs = _net(), _batch()
    base = net(obs)
    o2 = {k: v.clone() for k, v in obs.items()}
    o2["history"][:, 0, :10] = obs["actions"][:, 0]  # I activated card 3 effect 0 this turn
    o2["history"][:, 0, 10], o2["history"][:, 0, 11] = 0, 0
    o2["history_mask"][:, 0] = True
    cnt = net._opt_counts(o2["actions"].long(), o2["history"].long(), o2["history_mask"])
    cnt0 = net._opt_counts(obs["actions"].long(), obs["history"].long(), obs["history_mask"])
    assert not torch.allclose(cnt[:, 0], cnt0[:, 0]) and torch.allclose(cnt[:, 1:], cnt0[:, 1:])
    assert not torch.allclose(net(o2)[:, :4], base[:, :4])
    g0 = net._glob_cat(obs["global"].long(), obs["history"].long(), obs["history_mask"])
    o3 = {k: v.clone() for k, v in obs.items()}
    o3["history"][:, 0, 0], o3["history_mask"][:, 0] = int(Kind.SUMMON), True
    g1 = net._glob_cat(o3["global"].long(), o3["history"].long(), o3["history_mask"])
    assert not torch.allclose(g0, g1)


def test_desc_lookup_and_lua_bits():
    net = _net()
    hi, lo = torch.tensor([3, 3, 0, 9, 0]), torch.tensor([0, 1, 2, 0, 0])
    out = net.desc(hi, lo)
    assert torch.equal(out[4], torch.zeros(16))  # desc 0 -> nothing
    # (0, 2) and (5, 0) share string row 2 but only (3, *) has Lua: different vectors for different strings
    assert not torch.allclose(out[0], out[1]) and not torch.allclose(out[2], out[3])
    b = lua_bits(_emb()["lua_feats"])
    assert b.shape[1] == 16 + 34 + 13 + 32 + 5 + 1 + 4 + 1 and b[1, 16 + 33] == 1
    assert phase_index(torch.tensor([0, 1, 4, 0x200])).tolist() == [0, 1, 3, 10]


def test_new_modules_get_gradients():
    net = PolicyNetV3(_emb(), d=16, layers=4, heads=2, dropout=0.0).train()
    lg = net(_batch())
    torch.log_softmax(lg[:, :4], -1)[:, 0].sum().backward()
    for name in ("desc.str_proj.0.weight", "desc.lua_proj.0.weight", "eff_keys.proj.weight",
                 "card_sent.0.attn.in_proj_weight", "card_sent.1.attn.in_proj_weight", "opt_sent.attn.in_proj_weight",
                 "g_phase.weight", "ctx.1.weight"):  # fmt: skip
        g = dict(net.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name


def test_make_policy_v1_is_plain_policynet_and_v3_heads(tmp_path):
    emb = _emb()
    v1 = make_policy(emb, {"d": 16, "layers": 1, "heads": 2})
    assert type(v1) is PolicyNet
    ref = PolicyNet(emb, d=16, layers=1, heads=2)
    assert v1.state_dict().keys() == ref.state_dict().keys()
    s, lua = tmp_path / "s.npz", tmp_path / "l.npz"
    np.savez(s, keys=emb["str_keys"], rows=emb["str_rows"], vecs=emb["str_vecs"])
    np.savez(lua, keys=emb["lua_keys"], feats=emb["lua_feats"])
    base = {k: emb[k] for k in ("codes", "card", "effects", "n_effects")}
    v3 = make_policy(base, {"d": 16, "layers": 4, "heads": 2, "v3": True, "desc_strings": s, "desc_lua": lua},
                     value=True, q=True, dropout=0.0).eval()  # fmt: skip
    assert isinstance(v3, PolicyNetV3)
    lg, v = v3.forward_value(_batch())
    lg2, q = v3.forward_q(_batch())
    assert v.shape == (3,) and torch.allclose(lg, lg2) and torch.isfinite(q).all()
