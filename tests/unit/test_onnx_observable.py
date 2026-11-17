"""ONNX export of an observable-history PolicyNet: same v1 graph, metadata says the history source."""

import numpy as np
import pytest
import torch

torch.backends.mha.set_fastpath_enabled(False)

from decklight.model.onnx_export import (
    codes_to_rows,
    example_obs,
    export_nets,
    project_table,
    raw_table,
    read_metadata,
    sorted_codes,
    with_history_source,
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
    # A different window, so feeding the wrong source is visible.
    obs["obs_history"] = obs["history"].copy()
    obs["obs_history"][:, :, 2:4] = rng.choice(pool, size=obs["obs_history"][:, :, 2:4].shape)
    obs["obs_history_mask"] = np.ones_like(obs["history_mask"], dtype=bool)
    return obs


def _nets(emb, observable: bool):
    from decklight.model.policy import PolicyNet

    torch.manual_seed(0)
    policy = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, observable=observable).eval()
    torch.manual_seed(1)
    value = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=True).eval()
    return policy, value


def test_observable_export_metadata_and_parity(tmp_path):
    emb = _tiny_emb()
    codes = sorted_codes(emb)
    policy, value = _nets(emb, observable=True)
    paths = export_nets(policy, value, (len(codes) + 1, 16, 16), tmp_path,
                        ex=example_obs(2, n_cards=12, n_chain=4, n_opt=8, n_hist=6))

    assert read_metadata(paths["policy"]).get("history_source") == "obs_history"
    assert read_metadata(paths["value"]).get("history_source") == "history"  # not observable

    obs = _obs_with_codes(codes)
    rows = codes_to_rows(obs, codes)
    table = project_table(policy.text, raw_table(emb))
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obs.items()}
    with torch.no_grad():
        ref_logits = policy(t).numpy()  # observable_history picks obs_history itself

    feed = {k: np.ascontiguousarray(v) for k, v in with_history_source(rows, "obs_history").items()}
    feed["text"] = np.ascontiguousarray(table)
    pol = ORT.InferenceSession(str(paths["policy"]), providers=["CPUExecutionProvider"])
    got = pol.run(["logits"], {i.name: feed[i.name] for i in pol.get_inputs()})[0]
    assert got.shape == ref_logits.shape
    legal = obs["action_mask"]
    assert float(np.abs(got[legal] - ref_logits[legal]).max()) <= 1e-5

    # The history window must not give the same answer (the gate would not catch a wrong source otherwise).
    wrong = {k: np.ascontiguousarray(v) for k, v in rows.items()}
    wrong["text"] = feed["text"]
    bad = pol.run(["logits"], {i.name: wrong[i.name] for i in pol.get_inputs()})[0]
    assert float(np.abs(bad[legal] - ref_logits[legal]).max()) > 1e-4


def test_plain_policy_metadata_is_history(tmp_path):
    emb = _tiny_emb(seed=4)
    codes = sorted_codes(emb)
    policy, value = _nets(emb, observable=False)
    paths = export_nets(policy, value, (len(codes) + 1, 16, 16), tmp_path,
                        ex=example_obs(2, n_cards=12, n_chain=4, n_opt=8, n_hist=6))
    assert read_metadata(paths["policy"]).get("history_source") == "history"
    assert read_metadata(paths["value"]).get("history_source") == "history"
