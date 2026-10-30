"""Checkpoint loader: .safetensors (pure data) or legacy torch .pt files.

Every pipeline entry point that takes a ``--ckpt`` reads through :func:`load_state`,
so forge pack weights (``policy/weights.safetensors``) work everywhere a ``best.pt``
used to.
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint file and return the flat state dict.

    ``.safetensors`` files load with ``safetensors.torch.load_file`` (no pickle);
    anything else loads with ``torch.load(weights_only=True)`` and unwraps a
    ``{"model": ...}`` envelope when one is present.
    """
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(p), device="cpu")
    else:
        state = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
    return state
