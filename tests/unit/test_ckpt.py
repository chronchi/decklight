"""load_state gives identical tensors for .pt (flat or {"model": ...}) and .safetensors."""

import torch
from safetensors.torch import save_file

from decklight.model.ckpt import load_state


def test_load_state_pt_safetensors_equal(tmp_path):
    state = {"w1": torch.randn(4, 4), "w2": torch.zeros(3, dtype=torch.int64)}
    flat = tmp_path / "flat.pt"
    torch.save(state, flat)
    wrapped = tmp_path / "wrapped.pt"
    torch.save({"model": state, "epoch": 3}, wrapped)
    sf = tmp_path / "w.safetensors"
    save_file(state, sf)
    a, b, c = load_state(flat), load_state(wrapped), load_state(sf)
    assert set(a) == set(b) == set(c) == {"w1", "w2"}
    for k in a:
        assert torch.equal(a[k], b[k]) and torch.equal(a[k], c[k])
    assert c["w1"].dtype == torch.float32 and c["w2"].dtype == torch.int64
