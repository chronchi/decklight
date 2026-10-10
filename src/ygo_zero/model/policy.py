"""Policy network v1 for behaviour cloning (PLAN Phase 4).

Inputs are the raw int32 observation arrays (docs/obs_spec.md, docs/action_spec.md), batched:
cards [B,160,21] + card_mask, global [B,24], chain [B,16,5] + chain_mask, actions [B,64,10] + action_mask,
history [B,32,13] + history_mask. Output: one logit per action slot (illegal slots = -inf).

Card identity comes only from frozen text embeddings (ygo_zero.cards.embed: whole-card vector + mean of the
effect vectors), looked up by passcode, so unseen cards work; there is no per-ID table. Codes missing from the table
and hidden cards (code 0) share a learned "unknown" vector.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_KINDS = 32
TRIM_TO = 8  # trimmed lengths round up to a multiple of this (fewer distinct shapes for kernels / compile)


def _used(mask: torch.Tensor) -> int:
    """Length of the prefix of axis 1 that holds every set entry of the batch (rounded up to TRIM_TO, ≥ 1)."""
    any_ = mask.any(0)
    n = int(any_.nonzero().max().item()) + 1 if bool(any_.any()) else 1
    return min(mask.shape[1], -(-n // TRIM_TO) * TRIM_TO)


def trim_obs(obs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    """Drop the all-padding tail of the card, chain, history and option axes for this batch (NOTEBOOK 2026-10-06:
    a decision uses ~76 of 160 card rows, 0.3 of 16 chain links and 4.7 of 64 options). Masked positions never
    reach a real output (attention key masks, option mask), so the logits are unchanged; returns the trimmed obs and
    the full option count to pad the logits back to."""
    o = dict(obs)
    n_opt = obs["action_mask"].shape[1]
    lc, lch, lh, la = (_used(obs[k]) for k in ("card_mask", "chain_mask", "history_mask", "action_mask"))
    for k in ("cards", "card_mask", "card_known", "known_code"):
        if k in o:
            o[k] = o[k][:, :lc]
    o["chain"], o["chain_mask"] = o["chain"][:, :lch], o["chain_mask"][:, :lch]
    o["history"], o["history_mask"] = o["history"][:, :lh], o["history_mask"][:, :lh]
    o["actions"], o["action_mask"] = o["actions"][:, :la], o["action_mask"][:, :la]
    return o, n_opt


N_MSGS = 64
MAX_ARG = 64


def _bits(x: torch.Tensor, n: int) -> torch.Tensor:
    """Low n bits of an int tensor as floats [..., n]."""
    shifts = torch.arange(n, device=x.device)
    return ((x.long().unsqueeze(-1) >> shifts) & 1).float()


class CardText(nn.Module):
    """Frozen code -> text vector table (row 0 = unknown, learned), projected to d."""

    def __init__(
        self, codes: np.ndarray, card: np.ndarray, effects: np.ndarray, n_effects: np.ndarray, d: int
    ):
        super().__init__()
        n_eff = np.maximum(n_effects.astype(np.float32), 1.0)[:, None]
        eff_mean = effects.astype(np.float32).sum(1) / n_eff
        table = np.concatenate([card.astype(np.float32), eff_mean], axis=1)  # [N, 2*dim]
        self.register_buffer("codes", torch.from_numpy(codes.astype(np.int64)), persistent=False)
        self.register_buffer("table", torch.from_numpy(table), persistent=False)
        self.unknown = nn.Parameter(torch.zeros(table.shape[1]))
        self.proj = nn.Sequential(nn.Linear(table.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.p_unknown = 0.0

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        c = code.long().clamp(min=0)
        idx = torch.searchsorted(self.codes, c).clamp(max=len(self.codes) - 1)
        hit = (self.codes[idx] == c) & (c > 0)
        vec = torch.where(hit.unsqueeze(-1), self.table[idx], self.unknown.expand(*c.shape, -1))
        if self.training and self.p_unknown > 0.0:
            drop = torch.rand(c.shape, device=c.device) < self.p_unknown
            vec = torch.where(drop.unsqueeze(-1), self.unknown.expand(*c.shape, -1), vec)
        return self.proj(vec)


class ActionEmbed(nn.Module):
    """Option row [.., 10] -> d (shared by legal options and history entries)."""

    def __init__(self, text: CardText, d: int):
        super().__init__()
        self.text = text
        self.kind = nn.Embedding(N_KINDS, d)
        self.msg = nn.Embedding(N_MSGS, d)
        self.eff = nn.Embedding(32, d)
        self.arg = nn.Embedding(MAX_ARG + 1, d)
        self.loc = nn.Embedding(16, d)
        self.ctrl = nn.Embedding(3, d)
        self.step = nn.Embedding(16, d)
        self.eff_owner = nn.Linear(d, d)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        kind, code, eff_code, eff_idx, arg, msg, loc, ctrl, step = (
            a[..., i] for i in (0, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        return (
            self.kind(kind.clamp(0, N_KINDS - 1))
            + self.text(code)
            + self.eff_owner(self.text(eff_code))
            + self.eff(eff_idx.clamp(0, 31))
            + self.arg(arg.clamp(0, MAX_ARG))
            + self.msg(msg.clamp(0, N_MSGS - 1))
            + self.loc(loc.clamp(0, 15))
            + self.ctrl((ctrl + 1).clamp(0, 2))
            + self.step(step.clamp(0, 15))
        )


class PolicyNet(nn.Module):
    def __init__(
        self,
        emb: dict[str, np.ndarray],
        d: int = 256,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        card_dropout: float = 0.0,
        hist: int = 0,
        known: bool = False,
        value: bool = False,
        q: bool = False,
    ):
        """`hist`: long-memory window (obs v2, ADR 0005). 0 = v1 exactly (same parameters, old state dicts
        load with strict=True); > 0 adds a learned recency-position embedding and widens the turns-ago
        embedding to 32 buckets. `known`: card-knowledge memory (obs v2). False = exactly the step-A
        parameters (old state dicts load with strict=True); True adds a card_known embedding and feeds
        `known_code` to CardText where `card_known == 1`. `value`: adds a state-value head on the global token
        (RL); `forward` is unchanged, `forward_value` returns (logits, value in [-1, 1]). `q`: adds a
        per-option Q head on the same per-option encoding the logits head uses (VRPO); `q=False` (default)
        leaves parameters, state dict and outputs bit-identical. With `value=True, q=True`,
        `forward_q` returns Q(s,a) = V(s) + residual(s,a) with a zero-initialized residual, so
        Q(s,.) == V(s) exactly at init; with `value=False, q=True` it returns the residual alone
        (zeros at init). Illegal options are finite (never NaN); callers mask them via the logits."""
        super().__init__()
        self.hist = hist
        self.trim = (
            True  # drop batch padding before the encoder (trim_obs); False = the full fixed-size inputs
        )
        self.known = known
        self.text = CardText(emb["codes"], emb["card"], emb["effects"], emb["n_effects"], d)
        self.text.p_unknown = card_dropout
        if known:
            self.known_emb = nn.Embedding(
                3, d
            )  # card_known flag (0 unknown/hidden, 1 remembered, 2 revealed)
        self.loc = nn.Embedding(16, d)
        self.seq = nn.Embedding(33, d)
        self.ctrl = nn.Embedding(2, d)
        self.owner = nn.Embedding(2, d)
        self.pos = nn.Embedding(8, d)
        self.pub = nn.Embedding(2, d)
        self.sel = nn.Embedding(2, d)
        self.lvl = nn.Embedding(16, d)
        self.rank = nn.Embedding(16, d)
        self.link = nn.Embedding(8, d)
        self.bits = nn.Linear(32 + 8 + 32 + 9 + 16, d)  # type, attribute, race, link markers, status
        self.stats = nn.Linear(6, d)  # atk, def, base atk, base def, lscale, rscale (scaled)
        self.glob_num = nn.Linear(22, d)
        self.glob_msg = nn.Embedding(N_MSGS, d)
        self.glob_tok = nn.Parameter(torch.zeros(d))
        self.action = ActionEmbed(self.text, d)
        self.chain_ctrl = nn.Embedding(2, d)
        self.hist_actor = nn.Embedding(2, d)
        if hist > 0:
            self.hist_pos = nn.Embedding(hist, d)  # recency position: 0 = most recent
            self.hist_age = nn.Embedding(32, d)  # turns ago, clamped
        else:
            self.hist_age = nn.Embedding(8, d)
        self.type_emb = nn.Embedding(4, d)  # token type: global, card, chain, history
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.act_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        if value:
            self.value_head = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1), nn.Tanh()
            )
        self.has_q = bool(q)
        if q:
            self.q_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
            # Zero only the output layer: residual == 0 at init, yet the hidden layer still gets gradients
            # (zeroing both layers would leave every weight with a zero gradient, only the last bias would move).
            nn.init.zeros_(self.q_head[-1].weight)
            nn.init.zeros_(self.q_head[-1].bias)

    def encode_cards(
        self,
        cards: torch.Tensor,
        card_known: torch.Tensor | None = None,
        known_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = cards.long()
        code = c[..., 0]
        extra = 0.0
        if self.known:
            if card_known is None or known_code is None:
                raise KeyError(
                    "this PolicyNet was built with known=True: the obs needs card_known/known_code"
                )
            k = card_known.long().clamp(0, 2)
            code = torch.where(k == 1, known_code.long(), code)  # remembered identity, not "hidden"
            extra = self.known_emb(k)
        stats = torch.stack(
            [c[..., 14], c[..., 15], c[..., 16], c[..., 17], c[..., 18] & 0xFF, (c[..., 18] >> 8) & 0xFF], -1
        ).float()
        stats = torch.where(stats < 0, torch.zeros_like(stats), stats) / torch.tensor(
            [3000.0, 3000.0, 3000.0, 3000.0, 13.0, 13.0], device=c.device
        )
        bits = torch.cat(
            [
                _bits(c[..., 7], 32),
                _bits(c[..., 8], 8),
                _bits(c[..., 9], 32),
                _bits(c[..., 13], 9),
                _bits(c[..., 19], 16),
            ],
            -1,
        )
        return (
            self.text(code)
            + self.loc(c[..., 1].clamp(0, 15))
            + self.seq(c[..., 2].clamp(0, 32))
            + self.ctrl(c[..., 3].clamp(0, 1))
            + self.owner(c[..., 4].clamp(0, 1))
            + self.pos(c[..., 5].clamp(0, 7))
            + self.pub(c[..., 6].clamp(0, 1))
            + self.lvl(c[..., 10].clamp(0, 15))
            + self.rank(c[..., 11].clamp(0, 15))
            + self.link(c[..., 12].clamp(0, 7))
            + self.bits(bits)
            + self.stats(stats)
            + self.sel(c[..., 20].clamp(0, 1))
            + extra
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._run(obs)[0]

    def forward_value(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits, enc = self._run(obs)
        return logits, self.value_head(enc[:, 0]).squeeze(-1)

    def forward_q(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits, Q[B, n_opt]): Q(s,a) = V(s) + q_head(feat) when value_head exists, else q_head(feat).

        The residual head is zero-initialized, so with a value head Q(s,.) == V(s) exactly at init
        on legal options. Illegal/padded options come out as 0 (finite, never NaN/inf); callers mask
        them via the logits (pi == 0 there, so Vbar/sampling are unaffected). Requires `q=True`."""
        if not getattr(self, "has_q", False) or not hasattr(self, "q_head"):
            raise AttributeError("this PolicyNet was built with q=False: no Q head (use q=True)")
        logits, resid, enc, _, mask = self._run_features(obs, want_q=True)
        assert resid is not None
        if hasattr(self, "value_head"):
            v = self.value_head(enc[:, 0]).squeeze(-1)
            q = v.unsqueeze(1) + resid
        else:
            q = resid
        q = torch.where(mask, q, q.detach() * 0.0)  # illegal/pad -> 0, never NaN/inf
        return logits, q

    def _run(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _, enc, _, _ = self._run_features(obs, want_q=False)
        return logits, enc

    def encode(self, obs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, int]:
        """Encoder pass shared by every head: (trimmed obs, enc [B, T, d] with enc[:, 0] the global token, keep
        [B, T] (True = real token), full option count before trimming (0 when trim is off))."""
        n_opt = 0
        if self.trim:
            obs, n_opt = trim_obs(obs)
        cards, glob = obs["cards"], obs["global"].long()
        B = cards.shape[0]
        card_tok = (
            self.encode_cards(cards, obs.get("card_known"), obs.get("known_code")) + self.type_emb.weight[1]
        )
        num = glob[:, :22].float()
        num = torch.cat([num[:, :2] / 8000.0, num[:, 2:20] / 40.0, num[:, 20:22] / 64.0], -1)
        g = (
            self.glob_tok
            + self.glob_num(num)
            + self.glob_msg(glob[:, 21].clamp(0, N_MSGS - 1))
            + self.type_emb.weight[0]
        )
        ch = obs["chain"].long()
        chain_tok = self.text(ch[..., 1]) + self.chain_ctrl(ch[..., 4].clamp(0, 1)) + self.type_emb.weight[2]
        h = obs["history"].long()
        if self.hist > 0:
            age = self.hist_age(h[..., 11].clamp(0, 31))
            pos = self.hist_pos.weight[: h.shape[1]]
        else:
            age = self.hist_age(h[..., 11].clamp(0, 7))
            pos = 0.0
        hist_tok = (
            self.action(h[..., :10]) + self.hist_actor(h[..., 10].clamp(0, 1)) + age
            + self.type_emb.weight[3] + pos
        )  # fmt: skip
        tokens = torch.cat([g.unsqueeze(1), card_tok, chain_tok, hist_tok], 1)
        keep = torch.cat(
            [
                torch.ones(B, 1, dtype=torch.bool, device=cards.device),
                obs["card_mask"],
                obs["chain_mask"],
                obs["history_mask"],
            ],
            1,
        )
        enc = self.encoder(tokens, src_key_padding_mask=~keep)
        return obs, enc, keep, n_opt

    def _run_features(self, obs: dict[str, torch.Tensor], want_q: bool = False
                      ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int, torch.Tensor]:
        """Shared encoder + per-option features. Returns (logits, resid_or_None, enc, n_opt, legal_mask).

        `resid` is the raw q_head output (before adding V) when want_q, else None. `legal_mask` is
        [B, n_opt_full] (True = legal). Bit-identical to the old `_run` when want_q=False."""
        obs, enc, keep, n_opt = self.encode(obs)
        cards = obs["cards"]
        a = obs["actions"].long()
        act = self.action(a)
        # the option's own card, as encoded in context (row index into the card tokens; -1 = none)
        row = a[..., 1]
        n_cards = cards.shape[1]
        card_ctx = torch.gather(
            enc[:, 1 : 1 + n_cards], 1, row.clamp(0, n_cards - 1).unsqueeze(-1).expand(-1, -1, enc.shape[-1])
        )
        card_ctx = card_ctx * (row >= 0).unsqueeze(-1)
        q = self.act_norm(act + card_ctx)
        att, _ = self.cross(q, enc, enc, key_padding_mask=~keep)
        glob_ctx = enc[:, :1].expand(-1, a.shape[1], -1)
        feat = torch.cat([q + att, card_ctx, glob_ctx], -1)
        logits = self.head(feat).squeeze(-1)
        logits = logits.masked_fill(~obs["action_mask"], float("-inf"))
        resid = self.q_head(feat).squeeze(-1) if want_q else None
        legal = obs["action_mask"]
        if n_opt > logits.shape[1]:
            pad = n_opt - logits.shape[1]
            logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
            legal = torch.nn.functional.pad(legal, (0, pad), value=False)
            if resid is not None:
                resid = torch.nn.functional.pad(resid, (0, pad), value=0.0)
        return logits, resid, enc, n_opt, legal
