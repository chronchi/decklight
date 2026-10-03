"""A trained BC PolicyNet as a gauntlet policy (argmax over legal options, or sampling with a temperature).

    pol = BCPolicy("runs/<run>/bc/best.pt", "data/cards/emb/qwen3-0.6b-d512.npz")   # args read from metrics.jsonl
    actions = pol(buf, rows)                                                         # VecEnv buffers, like policies.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ygo_zero.model.bc_data import OBS_KEYS
from ygo_zero.model.policy import PolicyNet


def _run_args(ckpt: Path) -> dict:
    m = ckpt.parent / "metrics.jsonl"
    if m.is_file():
        for raw in m.read_text().splitlines():
            rec = json.loads(raw)
            if "args" in rec:
                return rec["args"]
    return {}


class BCPolicy:
    def __init__(self, ckpt: str | Path, emb: str | Path, temperature: float = 0.0, seed: int = 0,
                 threads: int = 2, batch: int = 256, device: str = "cpu") -> None:  # fmt: skip
        ckpt = Path(ckpt)
        a = _run_args(ckpt)
        torch.set_num_threads(threads)
        self.model = PolicyNet(dict(np.load(emb)), d=a.get("d", 256), layers=a.get("layers", 4),
                               heads=a.get("heads", 4))  # fmt: skip
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model", state)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.temperature, self.batch = temperature, batch
        self.rng = np.random.default_rng(seed)
        self.name = f"bc:{ckpt.parent.parent.name}"

    @torch.no_grad()
    def logits(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        t = {k: torch.from_numpy(np.ascontiguousarray(obs[k])).to(self.device) for k in OBS_KEYS}
        return self.model(t).float().cpu().numpy()

    def __call__(self, buf: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
        out = np.zeros(len(rows), np.int32)
        for i in range(0, len(rows), self.batch):
            r = rows[i : i + self.batch]
            lg = self.logits({k: buf[k][r] for k in OBS_KEYS})
            if self.temperature <= 0:
                out[i : i + len(r)] = lg.argmax(1)
            else:
                z = lg / self.temperature
                z = np.exp(z - z.max(1, keepdims=True))
                z /= z.sum(1, keepdims=True)
                out[i : i + len(r)] = [self.rng.choice(len(p), p=p) for p in z]
        return out
