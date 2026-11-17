"""Model adapters: one observation -> option probabilities, per-option features, attention and token groups.

The current BCPolicy / PolicyNet(V3) is wrapped with forward hooks, exactly like `scripts/audit/opp_probe.py`'s
`Tap` (no model code is changed). Architectures with no attention return `attn=None`; the attention section is then
skipped. Other models implement `ModelAdapter.run` directly or reuse `HookedAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from decklight.diag.common import softmax_options
from decklight.model.bc_data import KNOWN_KEYS, OBS_KEYS


@dataclass
class AdapterOutput:
    probs: np.ndarray  # [n_opt] legal-option softmax
    option_feats: np.ndarray  # [n_opt, F]
    global_feat: np.ndarray  # [G]
    attn: np.ndarray | None = None  # [n_opt, n_tokens]
    token_groups: list[str] | None = None  # [n_tokens]


class ModelAdapter:
    """Statistical interface used by every section."""

    name: str = "model"

    def run(self, obs: dict) -> AdapterOutput:  # pragma: no cover - interface
        raise NotImplementedError


_LOC = {1: "deck", 2: "hand", 3: "mzone", 4: "szone", 5: "gy", 6: "banished", 7: "extra", 8: "overlay"}


def token_groups_from_obs(obs: dict) -> list[str]:
    """Token names in encoder order: global, cards, chain, history (same layout as opp_probe.groups)."""
    cards = np.asarray(obs["cards"])[0].astype(int)
    out = ["global"]
    for c in cards:
        out.append(f"{'opp' if c[3] else 'my'}_{_LOC.get(int(c[1]), 'other')}")
    out += ["chain"] * np.asarray(obs["chain"]).shape[1]
    h = np.asarray(obs["history"])[0].astype(int)
    out += [f"hist_{'opp' if x[10] else 'my'}_{'now' if x[11] == 0 else 'old'}" for x in h]
    return out


class HookedAdapter(ModelAdapter):
    """Wrap a torch policy with encode/cross/head modules. `forward` turns one batched obs into logits."""

    def __init__(self, model: torch.nn.Module, forward=None, *, device: str = "cpu", name: str = "model"):
        self.model = model.to(torch.device(device)).eval()
        self.device = torch.device(device)
        self.name = name
        self._forward = forward
        self.cap: dict = {}
        self._hooks = []
        enc = getattr(model, "encode", None)
        if enc is not None:
            def encode(obs):
                out = enc(obs)
                self.cap["obs"] = out[0]
                return out

            model.encode = encode
        cross = getattr(model, "cross", None)
        if cross is not None:
            self._hooks.append(
                cross.register_forward_hook(
                    lambda _m, _i, o: self.cap.__setitem__(
                        "attn", (o[1].detach() if isinstance(o, tuple) and len(o) > 1 else None)
                    )
                )
            )
        head = getattr(model, "head", None)
        if head is not None:
            self._hooks.append(
                head.register_forward_hook(lambda _m, i, _o: self.cap.__setitem__("feat", i[0].detach()))
            )

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _default_forward(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        t = {k: torch.from_numpy(np.ascontiguousarray(batch[k])).to(self.device) for k in OBS_KEYS if k in batch}
        with torch.no_grad():
            return self.model(t).float().cpu().numpy()

    @torch.no_grad()
    def run(self, obs: dict) -> AdapterOutput:
        self.cap = {}
        batch = {k: np.asarray(v)[None] for k, v in obs.items()}
        logits = (self._forward or self._default_forward)(batch)
        probs = softmax_options(np.asarray(logits)[0], np.asarray(obs["action_mask"]))
        feat = self.cap.get("feat")
        if feat is None:
            raise RuntimeError(f"{self.name}: no per-option feature capture (model has no `head`)")
        feat = feat[0].float().numpy()
        d = feat.shape[1] // 3
        glob = feat[0, -d:] if d > 0 else feat[0, :0]
        attn = self.cap.get("attn")
        attn = attn[0].float().numpy() if attn is not None else None
        groups = token_groups_from_obs(self.cap["obs"]) if "obs" in self.cap else None
        if attn is not None and groups is not None and len(groups) != attn.shape[-1]:
            groups = (groups + ["pad"] * attn.shape[-1])[: attn.shape[-1]]
        return AdapterOutput(probs=probs, option_feats=feat, global_feat=glob, attn=attn,
                             token_groups=groups)  # fmt: skip


class BCPolicyAdapter(HookedAdapter):
    """A BCPolicy checkpoint (v1 or v3); `name` is the run/checkpoint name for reports."""

    def __init__(self, ckpt: str | Path, emb: str | Path, *, device: str = "cpu", threads: int = 2,
                 name: str | None = None):  # fmt: skip
        from decklight.eval.bc_policy import BCPolicy

        self.policy = BCPolicy(ckpt, emb, device=device, threads=threads)
        fwd = lambda batch: self.policy.logits(batch)
        super().__init__(self.policy.model, fwd, device=device,
                         name=name or Path(ckpt).parent.parent.name)  # fmt: skip


def obs_for_policy(obs: dict) -> dict:
    """Drop non-model keys (e.g. `_player`) before batching for a policy."""
    return {k: v for k, v in obs.items() if k in OBS_KEYS or k in KNOWN_KEYS}
