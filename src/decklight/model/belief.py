"""Belief head: P(card c of the opponent's list is among its hidden cards | actor's view) (PLAN §0a "Hidden-information
bound" (b)).

The trunk is a PolicyNet (init from a policy checkpoint); each candidate code queries the encoder output (cards,
chain, history tokens) by cross-attention. The output is a residual on the logit of the hypergeometric baseline the
determinizer uses today (scripts/build_belief_shards.py), zero-initialized, so an untrained head reproduces the
baseline exactly and any loss below it is information the model extracted from the state and history.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from decklight.model.policy import PolicyNet

EPS = 1e-4


class BeliefNet(nn.Module):
    def __init__(self, trunk: PolicyNet, heads: int = 4):
        super().__init__()
        self.trunk = trunk
        d = trunk.glob_tok.shape[0]
        self.copies = nn.Embedding(5, d)  # copies of the code in the list (1-3, clamped)
        self.q_norm = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, obs: dict[str, torch.Tensor], cand: torch.Tensor, cand_n: torch.Tensor,
                base_p: torch.Tensor) -> torch.Tensor:  # fmt: skip
        """Logits [B, K] of "code cand[:, k] is hidden"; padded candidates (cand == 0) are returned but meaningless."""
        _, enc, keep, _ = self.trunk.encode(obs)
        q = self.q_norm(self.trunk.text(cand.long()) + self.copies(cand_n.long().clamp(0, 4)))
        att, _ = self.cross(q, enc, enc, key_padding_mask=~keep)
        g = enc[:, :1].expand(-1, q.shape[1], -1)
        resid = self.mlp(torch.cat([q + att, g, q * g], -1)).squeeze(-1)
        return torch.logit(base_p.float().clamp(EPS, 1 - EPS)) + resid


def scored(cand: torch.Tensor, base_p: torch.Tensor) -> torch.Tensor:
    """Entries that carry information: a real candidate whose baseline is neither 0 nor 1 (those are determined by
    the visible state: every copy seen, or more hidden slots than other cards)."""
    return (cand > 0) & (base_p > 0) & (base_p < 1)


def bce(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Summed BCE over the scored entries m."""
    return F.binary_cross_entropy_with_logits(logits[m], y[m].float(), reduction="sum")


def baseline_bce(y: torch.Tensor, base_p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    lg = torch.logit(base_p.float().clamp(EPS, 1 - EPS))
    return bce(lg, y, m)


def init_trunk(trunk: PolicyNet, state: dict[str, torch.Tensor]) -> list[str]:
    """Load policy weights into a trunk that may have extra parts (known_emb, hist_pos, a wider hist_age). New
    parts start as no-ops (zero known_emb / hist_pos); a wider hist_age copies the old rows and repeats the last
    bucket. Returns the names that did not come from `state`."""
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    own = trunk.state_dict()
    fresh = []
    for k, v in own.items():
        if k in state and state[k].shape == v.shape:
            own[k] = state[k]
        elif k == "hist_age.weight" and k in state:
            old = state[k]
            new = old[-1:].expand(v.shape[0], -1).clone()
            new[: old.shape[0]] = old
            own[k] = new
        else:
            if k.startswith(("known_emb", "hist_pos")):
                own[k] = torch.zeros_like(v)
            fresh.append(k)
    missing_from_trunk = [k for k in state if k not in own]
    if missing_from_trunk:
        raise KeyError(f"checkpoint has parameters the trunk lacks: {missing_from_trunk[:5]}")
    trunk.load_state_dict(own)
    return fresh
