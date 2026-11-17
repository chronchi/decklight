"""ONNX export parity on a tiny net (offline-app spike S2, product nets in scripts/onnx_parity.py)."""

import numpy as np
import pytest
import torch

torch.backends.mha.set_fastpath_enabled(False)

from decklight.model.onnx_export import (
    CODE_COLUMNS,
    codes_to_rows,
    example_obs,
    export_nets,
    project_table,
    raw_table,
    sorted_codes,
)

ORT = pytest.importorskip("onnxruntime")


def _tiny_emb(seed=0, n=24):
    rng = np.random.default_rng(seed)
    codes = np.sort(rng.choice(np.arange(100, 9999), size=n, replace=False)).astype(np.int64)
    return {
        "codes": codes,
        "card": rng.normal(size=(n, 8)).astype(np.float16),
        "effects": rng.normal(size=(n, 2, 8)).astype(np.float16),
        "n_effects": np.ones(n, dtype=np.int64),
    }


def _tiny_nets(emb):
    from decklight.model.policy import PolicyNet

    torch.manual_seed(0)
    policy = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0).eval()
    torch.manual_seed(1)
    value = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=True).eval()
    return policy, value


def _obs_with_codes(codes, batch=4, seed=3):
    rng = np.random.default_rng(seed)
    obs = {k: v.numpy() for k, v in example_obs(batch, n_cards=12, n_chain=4, n_opt=8, n_hist=6).items()}
    pool = np.concatenate([[0, -5, 424242], codes])
    obs["cards"][:, :, 0] = rng.choice(pool, size=obs["cards"][:, :, 0].shape)
    obs["card_mask"] = np.ones_like(obs["card_mask"], dtype=bool)
    obs["chain"][:, :, 1] = rng.choice(pool, size=obs["chain"][:, :, 1].shape)
    obs["chain_mask"] = np.ones_like(obs["chain_mask"], dtype=bool)
    obs["actions"][:, :, 2:4] = rng.choice(pool, size=obs["actions"][:, :, 2:4].shape)
    obs["actions"][:, :, 1] = rng.integers(-1, 12, size=obs["actions"][:, :, 1].shape)
    obs["action_mask"] = np.ones_like(obs["action_mask"], dtype=bool)
    obs["history"][:, :, 2:4] = rng.choice(pool, size=obs["history"][:, :, 2:4].shape)
    obs["history_mask"] = np.ones_like(obs["history_mask"], dtype=bool)
    return obs


def test_codes_to_rows_unknown_zero_negative():
    codes = np.array([100, 200, 300], dtype=np.int64)
    obs = {
        "cards": np.array([[[100], [0], [-7], [999], [300]]], dtype=np.int32),
        "actions": np.zeros((1, 1, 10), dtype=np.int32),
        "chain": np.zeros((1, 1, 5), dtype=np.int32),
        "history": np.zeros((1, 1, 13), dtype=np.int32),
    }
    out = codes_to_rows(obs, codes)
    assert out["cards"].ravel().tolist() == [1, 0, 0, 0, 3]
    assert out is not obs and (obs["cards"].ravel() == [100, 0, -7, 999, 300]).all()  # copy, input kept
    # torch path keeps device/dtype
    t = {k: torch.from_numpy(v) for k, v in obs.items()}
    out_t = codes_to_rows(t, codes)
    assert isinstance(out_t["cards"], torch.Tensor) and out_t["cards"].dtype == torch.int32
    assert out_t["cards"].ravel().tolist() == [1, 0, 0, 0, 3]


def test_code_columns_cover_all_text_lookups():
    """Every model code read is a listed column (cards c0, chain c1, actions/history c2-3)."""
    import inspect

    from decklight.model import policy as P

    src = inspect.getsource(P.ActionEmbed.forward) + inspect.getsource(P.PolicyNet.encode_cards)
    assert "for i in (0, 2, 3, 4, 5, 6, 7, 8, 9)" in src  # ActionEmbed: code, effect_code cols
    assert "self.text(code)" in src and "self.text(eff_code)" in src
    assert "c[..., 0]" in src  # encode_cards
    assert "ch[..., 1]" in inspect.getsource(P.PolicyNet.encode)  # chain handler
    got = sorted((c["key"], c["col"]) for c in CODE_COLUMNS)
    assert got == [("actions", 2), ("actions", 3), ("cards", 0), ("chain", 1), ("history", 2), ("history", 3),
                   ("obs_history", 2), ("obs_history", 3)]


def test_tiny_export_parity(tmp_path):
    emb = _tiny_emb()
    codes = sorted_codes(emb)
    assert (codes == np.asarray(emb["codes"])).all()
    policy, value = _tiny_nets(emb)
    raw = raw_table(emb)
    assert raw.shape == (len(codes), 16) and raw.dtype == np.float32
    paths = export_nets(policy, value, (len(codes) + 1, 16, 16), tmp_path,
                        ex=example_obs(2, n_cards=12, n_chain=4, n_opt=8, n_hist=6))

    obs = _obs_with_codes(codes)
    rows = codes_to_rows(obs, codes)
    table = project_table(policy.text, raw)
    table_v = project_table(value.text, raw)
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obs.items()}
    with torch.no_grad():
        ref_logits = policy(t).numpy()
        ref_value = value.forward_value(t)[1].numpy()

    providers = ["CPUExecutionProvider"]
    text_full = np.concatenate([policy.text.unknown.detach().float().numpy().reshape(1, -1), raw])
    got_text = ORT.InferenceSession(str(paths["card_proj"]), providers=providers).run(
        ["text"], {"raw": np.ascontiguousarray(text_full)})[0]
    assert got_text.shape == table.shape
    assert float(np.abs(got_text - table).max()) <= 1e-5

    feed = {k: np.ascontiguousarray(rows[k]) for k in
            ("cards", "card_mask", "global", "chain", "chain_mask", "actions", "action_mask",
             "history", "history_mask")}
    feed["text"] = np.ascontiguousarray(table)
    pol = ORT.InferenceSession(str(paths["policy"]), providers=providers)
    got_logits = pol.run(["logits"], {i.name: feed[i.name] for i in pol.get_inputs()})[0]
    assert got_logits.shape == ref_logits.shape
    assert float(np.abs(got_logits - ref_logits).max()) <= 1e-5

    feed_v = dict(feed)
    feed_v["text"] = np.ascontiguousarray(table_v)
    val = ORT.InferenceSession(str(paths["value"]), providers=providers)
    got_value = val.run(["value"], {i.name: feed_v[i.name] for i in val.get_inputs()})[0]
    assert got_value.shape == ref_value.shape
    assert float(np.abs(got_value - ref_value).max()) <= 1e-5


def test_trimmed_inputs_and_negative_bitfields(tmp_path):
    """The graph has dynamic token axes: trimmed obs (trim_obs_np, the twin of policy.trim_obs) give torch's logits.
    Bit-field columns carry negative ints (bit 31 set): the exported `//`, `%` must keep `>>`, `&` semantics."""
    from decklight.model.onnx_export import pad_logits, trim_obs_np
    from decklight.model.policy import trim_obs

    emb = _tiny_emb()
    codes = sorted_codes(emb)
    policy, value = _tiny_nets(emb)
    paths = export_nets(policy, value, (len(codes) + 1, 16, 16), tmp_path,
                        ex=example_obs(2, n_cards=40, n_chain=16, n_opt=24, n_hist=16))
    rng = np.random.default_rng(5)
    obs = {k: v.numpy() for k, v in example_obs(3, n_cards=40, n_chain=16, n_opt=24, n_hist=16).items()}
    pool = np.concatenate([[0, -5], codes])
    obs["cards"][..., 0] = rng.choice(pool, size=obs["cards"].shape[:2])
    for col in (7, 8, 9, 13, 18, 19):  # bit fields and packed stats, some negative
        obs["cards"][..., col] = rng.integers(-(2**31), 2**31 - 1, size=obs["cards"].shape[:2], dtype=np.int64)
    obs["actions"][..., 2:4] = rng.choice(pool, size=obs["actions"][..., 2:4].shape)
    obs["actions"][..., 1] = rng.integers(-1, 9, size=obs["actions"].shape[:2])
    for key, used in (("card_mask", (9, 17, 3)), ("chain_mask", (0, 2, 0)), ("action_mask", (3, 11, 1)),
                      ("history_mask", (4, 0, 9))):
        for b, n in enumerate(used):
            obs[key][b, :n] = True
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obs.items()}
    trimmed_t, _ = trim_obs(t)
    trimmed, n_opt = trim_obs_np(codes_to_rows(obs, codes))
    assert {k: v.shape for k, v in trimmed.items()} == {k: tuple(v.shape) for k, v in trimmed_t.items()}
    assert trimmed["cards"].shape[1] == 24 and n_opt == 24  # 17 used -> rounded up to 24 (TRIM_TO 8)
    with torch.no_grad():
        ref_logits = policy(t).numpy()
        ref_value = value.forward_value(t)[1].numpy()
    providers = ["CPUExecutionProvider"]
    feed = dict(trimmed, text=project_table(policy.text, raw_table(emb)))
    pol = ORT.InferenceSession(str(paths["policy"]), providers=providers)
    got = pad_logits(pol.run(["logits"], {i.name: feed[i.name] for i in pol.get_inputs()})[0], n_opt)
    legal = obs["action_mask"]
    assert (np.isneginf(got) == ~legal).all()
    assert float(np.abs(got[legal] - ref_logits[legal]).max()) <= 1e-5
    feed["text"] = project_table(value.text, raw_table(emb))
    val = ORT.InferenceSession(str(paths["value"]), providers=providers)
    got_v = val.run(["value"], {i.name: feed[i.name] for i in val.get_inputs()})[0]
    assert float(np.abs(got_v - ref_value).max()) <= 1e-5
