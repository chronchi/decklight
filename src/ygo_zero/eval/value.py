"""A trained value PolicyNet as a search scorer: P(the acting player wins | state).

    net = ValueNet("runs/<run>/value/best.pt", "data/cards/emb/qwen3-0.6b-d512.npz")
    v = net.values(obs_batch)  # np.ndarray in [-1, 1], actor's view
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ygo_zero.eval.bc_policy import _run_args
from ygo_zero.model.bc_data import KNOWN_KEYS, OBS_KEYS
from ygo_zero.model.policy import PolicyNet


class ValueNet:
    def __init__(self, ckpt: str | Path, emb: str | Path, device: str = "cpu",
                 batch: int = 512, threads: int = 2) -> None:  # fmt: skip
        ckpt = Path(ckpt)
        a = _run_args(ckpt)
        if a and not a.get("value", False):
            raise ValueError(f"{ckpt} is not a value checkpoint (metrics args lack value: true)")
        torch.set_num_threads(threads)
        self.hist = int(a.get("hist", 0) or 0)
        self.known = bool(a.get("known", False))
        self.model = PolicyNet(dict(np.load(emb)), d=a.get("d", 256), layers=a.get("layers", 4),
                               heads=a.get("heads", 4), hist=self.hist, known=self.known,
                               value=True)  # fmt: skip
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model", state)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.batch = batch

    def _tensor(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        if self.known and not all(k in obs for k in KNOWN_KEYS):
            raise KeyError("this checkpoint was trained with --known: the obs needs card_known/known_code")
        if self.hist > 0:
            if "hist_long" not in obs:
                raise KeyError("this checkpoint was trained with --hist: the obs needs hist_long/hist_long_mask")
            t = {k: torch.from_numpy(np.ascontiguousarray(obs[k])).to(self.device) for k in OBS_KEYS
                 if k not in ("history", "history_mask")}
            t["history"] = torch.from_numpy(np.ascontiguousarray(obs["hist_long"][:, : self.hist])).to(self.device)
            t["history_mask"] = torch.from_numpy(np.ascontiguousarray(obs["hist_long_mask"][:, : self.hist])).to(self.device)
            if self.known:
                for k in KNOWN_KEYS:
                    t[k] = torch.from_numpy(np.ascontiguousarray(obs[k])).to(self.device)
        else:
            keys = tuple(OBS_KEYS) + (tuple(KNOWN_KEYS) if self.known else ())
            t = {k: torch.from_numpy(np.ascontiguousarray(obs[k])).to(self.device) for k in keys}
        return t

    @torch.no_grad()
    def values(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Value in [-1, 1] for each row of a batched obs (actor's view)."""
        n = next(iter(obs.values())).shape[0]
        out = np.zeros(n, np.float32)
        for i in range(0, n, self.batch):
            t = self._tensor({k: v[i : i + self.batch] for k, v in obs.items()})
            _, v = self.model.forward_value(t)
            out[i : i + self.batch] = v.float().cpu().numpy()
        return np.clip(out, -1, 1)
