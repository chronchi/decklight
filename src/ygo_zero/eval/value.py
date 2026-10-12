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
from ygo_zero.model.policy_v3 import make_policy


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
        self.model = make_policy(emb, a, value=True)
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


class PrivValueNet:
    """A privileged critic (train_rl --priv-critic: PolicyNet known=True + value head) as a search scorer.

    Each obs row must carry `priv_code` (CppEnv obs do): the critic sees that world's hidden cards through
    `critic_obs`. Meant for honest search, where every rollout runs in a sampled world, so revealing the sampled
    world's cards leaks nothing about the real game. Same output as ValueNet.values: [-1, 1], actor's view.
    """

    def __init__(self, ckpt: str | Path, emb: str | Path, device: str = "cpu",
                 batch: int = 512, threads: int = 2) -> None:  # fmt: skip
        from ygo_zero.eval.turn1 import load_critic_local, run_args

        ckpt = Path(ckpt)
        a = run_args(ckpt)
        if not a or str(a.get("rl", {}).get("priv_critic", "0")) != "1":
            raise ValueError(f"{ckpt}: metrics args next to it lack rl.priv_critic=1 (not a privileged critic)")
        torch.set_num_threads(threads)
        self.hist = int(a.get("hist", 0) or 0)
        self.model = load_critic_local(ckpt, dict(np.load(emb)), a)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.batch = batch

    @torch.no_grad()
    def values(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from ygo_zero.rl.critic import critic_obs

        if "priv_code" not in obs:
            raise KeyError("PrivValueNet needs priv_code in the obs (CppEnv obs carry it)")
        n = obs["priv_code"].shape[0]
        out = np.zeros(n, np.float32)
        for i in range(0, n, self.batch):
            sl = {k: v[i : i + self.batch] for k, v in obs.items()}
            b: dict[str, np.ndarray] = {}
            for k in OBS_KEYS:
                if k in ("history", "history_mask") and self.hist > 0:
                    b[k] = sl["hist_long" if k == "history" else "hist_long_mask"][:, : self.hist]
                else:
                    b[k] = sl[k]
            for k in KNOWN_KEYS:
                if k in sl:
                    b[k] = sl[k]
            cob = critic_obs(b, sl["priv_code"])
            t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(self.device) for k, v in cob.items()}
            out[i : i + self.batch] = self.model.forward_value(t)[1].float().cpu().numpy()
        return np.clip(out, -1, 1)
