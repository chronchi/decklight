"""Policy network v3 (retrain plan v3, stage B): PolicyNet plus the inputs the v1 encoder cannot see.

Same observation arrays as v1 (no obs change: obs v3 was dropped, NOTEBOOK 2026-10-11). Additions:
- **effect keys**: options, chain links and history entries carry `desc = (code << 20) | idx`; v3 embeds the string
  it names (frozen text vector of card str1..16 / system strings, `cards.effect_strings`) plus the Lua record of the
  effect with that description (type, category, range, property, count limit, cond/cost/target flags);
- **effect sentences**: each card token cross-attends to its card's effect-sentence vectors (≤ 8, frozen) after
  encoder layers 2 and 4; each option attends to the sentences of the card that owns its effect;
- **categorical globals**: phase, turn, went first, whose turn, chain length, LP bins, zone counts and "Normal
  Summon used this turn" (me / opponent) as embeddings, plus a context vector added to every option query (v1 feeds
  them as one linear map of raw numbers /40);
- **activation counts**: per option, how often that exact effect (code, desc) was activated this turn and in the
  history window, by me and by the opponent; per card, my activations of that code this turn;
- **history phase** (column 12) embedded.

Card identity still comes only from frozen text (no per-ID table). v1 checkpoints are untouched: v3 is a subclass,
selected by the run arg `v3` through `make_policy`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ygo_zero.engine.actions import Kind
from ygo_zero.model.policy import PolicyNet, trim_obs

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRINGS = ROOT / "data" / "cards" / "emb" / "strings-qwen3-0.6b-d512.npz"
DEFAULT_LUA = ROOT / "data" / "cards" / "lua_desc.npz"
# Custom cards: point the product at merged tables without touching checkpoints' stored args.
STRINGS_ENV = "YGO_STRINGS"
LUA_ENV = "YGO_LUA"
N_LUA = 16 + 34 + 13 + 32 + 5 + 1 + 4 + 1


def lua_bits(feats: np.ndarray) -> np.ndarray:
    """lua_desc rows (cards.effect_strings.LUA_COLS) -> float features [N, N_LUA]."""
    f = feats.astype(np.int64)

    def bits(x: np.ndarray, n: int) -> np.ndarray:
        return ((x[:, None] >> np.arange(n)) & 1).astype(np.float32)

    cnt = np.eye(5, dtype=np.float32)[np.clip(f[:, 4], 0, 4)]
    out = np.concatenate([bits(f[:, 0], 16), bits(f[:, 1], 34), bits(f[:, 2], 13), bits(f[:, 3], 32), cnt,
                          f[:, 5:6].astype(np.float32), bits(f[:, 6], 4), np.ones((len(f), 1), np.float32)], 1)  # fmt: skip
    assert out.shape[1] == N_LUA
    return out


def load_desc_tables(strings: Path | str | None = None, lua: Path | str | None = None) -> dict[str, np.ndarray]:
    """The extra arrays v3 needs, to merge into the card-embedding dict passed to PolicyNetV3.

    Explicit args win; else `$YGO_STRINGS` / `$YGO_LUA` (merged tables for custom cards); else the
    stock `data/cards` files.
    """
    s = np.load(strings or os.environ.get(STRINGS_ENV) or DEFAULT_STRINGS)
    y = np.load(lua or os.environ.get(LUA_ENV) or DEFAULT_LUA)
    return {"str_keys": s["keys"], "str_rows": s["rows"], "str_vecs": s["vecs"], "lua_keys": y["keys"],
            "lua_feats": y["feats"]}  # fmt: skip


def phase_index(p: torch.Tensor) -> torch.Tensor:
    """PHASE_* bit value -> 1 + bit position (0 = none), clamped to 15."""
    p = p.long().clamp(min=0)
    idx = torch.zeros_like(p)
    for b in range(15):
        idx = torch.where((p >> b) & 1 == 1, torch.full_like(p, b + 1), idx)
    return idx


class DescEmbed(nn.Module):
    """(desc_hi, desc_lo) -> d: projected string vector + projected Lua record (0 / learned 'none' when absent)."""

    def __init__(self, emb: dict[str, np.ndarray], d: int):
        super().__init__()
        vecs = emb["str_vecs"].astype(np.float32)
        self.register_buffer("str_keys", torch.from_numpy(emb["str_keys"].astype(np.int64)), persistent=False)
        self.register_buffer("str_rows", torch.from_numpy(emb["str_rows"].astype(np.int64)), persistent=False)
        self.register_buffer("str_vecs", torch.from_numpy(vecs), persistent=False)
        self.register_buffer("lua_keys", torch.from_numpy(emb["lua_keys"].astype(np.int64)), persistent=False)
        self.register_buffer("lua_feats", torch.from_numpy(lua_bits(emb["lua_feats"])), persistent=False)
        self.no_str = nn.Parameter(torch.zeros(vecs.shape[1]))
        self.str_proj = nn.Sequential(nn.Linear(vecs.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.lua_proj = nn.Sequential(nn.Linear(N_LUA, d), nn.GELU(), nn.Linear(d, d))
        self.no_lua = nn.Parameter(torch.zeros(d))
        self.p_unknown = 0.0

    @staticmethod
    def _find(keys: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        i = torch.searchsorted(keys, k).clamp(max=len(keys) - 1)
        return i, keys[i] == k

    def forward(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        hi, lo = hi.long().clamp(min=0), lo.long().clamp(min=0)
        k = (hi << 20) | lo
        some = k > 0
        i, hit = self._find(self.str_keys, k)
        hit = hit & some
        if self.training and self.p_unknown > 0.0:  # same dropout as unknown card codes (new strings at test time)
            hit = hit & (torch.rand(k.shape, device=k.device) >= self.p_unknown)
        vec = torch.where(hit.unsqueeze(-1), self.str_vecs[self.str_rows[i]], self.no_str.expand(*k.shape, -1))
        j, lhit = self._find(self.lua_keys, k)
        lua = torch.where((lhit & some).unsqueeze(-1), self.lua_proj(self.lua_feats[j]), self.no_lua.expand(*k.shape, -1))
        return (self.str_proj(vec) + lua) * some.unsqueeze(-1)


class EffectKeys(nn.Module):
    """code -> (keys [.., 1 + slots, d], pad mask): a learned null slot + the card's effect-sentence vectors."""

    def __init__(self, emb: dict[str, np.ndarray], d: int):
        super().__init__()
        self.register_buffer("codes", torch.from_numpy(emb["codes"].astype(np.int64)), persistent=False)
        self.register_buffer("effects", torch.from_numpy(emb["effects"].astype(np.float16)), persistent=False)
        self.register_buffer("n_eff", torch.from_numpy(emb["n_effects"].astype(np.int64)), persistent=False)
        self.slots = int(emb["effects"].shape[1])  # MAX_EFFECTS of the embedding build (8)
        self.proj = nn.Linear(emb["effects"].shape[-1], d)
        self.pos = nn.Embedding(self.slots, d)
        self.null = nn.Parameter(torch.zeros(d))

    def forward(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = code.long().clamp(min=0)
        idx = torch.searchsorted(self.codes, c).clamp(max=len(self.codes) - 1)
        hit = (self.codes[idx] == c) & (c > 0)
        flat = torch.where(hit, idx, torch.full_like(idx, -1)).reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)  # project each distinct card once
        safe = uniq.clamp(min=0)
        k = self.proj(self.effects[safe].float()) + self.pos.weight
        n = torch.where(uniq >= 0, self.n_eff[safe], torch.zeros_like(safe))
        pad = torch.arange(self.slots, device=c.device) >= n.unsqueeze(-1)  # True = no sentence
        k = torch.cat([self.null.expand(len(uniq), 1, -1), k], 1)[inv]
        pad = torch.cat([torch.zeros(len(uniq), 1, dtype=torch.bool, device=c.device), pad], 1)[inv]
        return k.reshape(*c.shape, self.slots + 1, -1), pad.reshape(*c.shape, self.slots + 1)


class SentenceAttn(nn.Module):
    """x [.., d] attends to its card's sentence keys; pre-norm residual."""

    CHUNK = 32768

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, keys: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        q = self.norm(x).reshape(-1, 1, shp[-1])
        k = keys.reshape(q.shape[0], -1, shp[-1])
        m = pad.reshape(q.shape[0], -1)
        # one attention row per token: batch 512 x ~160 cards exceeds the CUDA efficient-attention batch limit (65,535;
        # stage-b-20261013-0544 crashed at step 1), so long batches go in chunks
        out = torch.cat([self.attn(q[i : i + self.CHUNK], k[i : i + self.CHUNK], k[i : i + self.CHUNK],
                                   key_padding_mask=m[i : i + self.CHUNK], need_weights=False)[0]
                         for i in range(0, q.shape[0], self.CHUNK)])  # fmt: skip
        return x + out.reshape(shp)


N_COUNTS = 14  # global 6..19
CNT_MAX = 40
LP_BINS = 21  # 500 LP per bin, last bin = 10000+
ACT = int(Kind.ACTIVATE)
NS_KINDS = (int(Kind.SUMMON), int(Kind.MSET))


class PolicyNetV3(PolicyNet):
    SENT_AFTER = (1, 3)  # encoder layer indices (0-based) after which card tokens read their sentences

    def __init__(self, emb: dict[str, np.ndarray], d: int = 256, layers: int = 4, heads: int = 4, dropout: float = 0.1,
                 card_dropout: float = 0.0, **kw):  # fmt: skip
        super().__init__(emb, d=d, layers=layers, heads=heads, dropout=dropout, card_dropout=card_dropout, **kw)
        self.desc = DescEmbed(emb, d)
        self.desc.p_unknown = card_dropout
        self.eff_keys = EffectKeys(emb, d)
        self.card_sent = nn.ModuleList(SentenceAttn(d, heads, dropout) for _ in self.SENT_AFTER)
        self.opt_sent = SentenceAttn(d, heads, dropout)
        self.g_phase = nn.Embedding(16, d)
        self.g_turn = nn.Embedding(32, d)
        self.g_first = nn.Embedding(2, d)
        self.g_mine = nn.Embedding(2, d)
        self.g_chain = nn.Embedding(17, d)
        self.g_lp = nn.Embedding(2 * LP_BINS, d)
        self.g_cnt = nn.Embedding(N_COUNTS * (CNT_MAX + 1), d)
        self.g_ns = nn.Embedding(4, d)  # Normal Summon used this turn: [me 0/1] + [opp 2/3]
        self.ctx = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.opt_cnt = nn.Embedding(4 * 4, d)  # (me/opp x this turn/window) x count 0..3
        self.card_cnt = nn.Embedding(4, d)  # my activations of this code this turn, 0..3
        self.hist_phase = nn.Embedding(16, d)

    # ---- helpers -------------------------------------------------------------------------------------------
    def _glob_cat(self, glob: torch.Tensor, h: torch.Tensor, hmask: torch.Tensor) -> torch.Tensor:
        lp = torch.stack([(glob[:, 0].clamp(min=0) // 500).clamp(max=LP_BINS - 1),
                          LP_BINS + (glob[:, 1].clamp(min=0) // 500).clamp(max=LP_BINS - 1)], 1)  # fmt: skip
        off = torch.arange(N_COUNTS, device=glob.device) * (CNT_MAX + 1)
        cnt = glob[:, 6:20].clamp(0, CNT_MAX) + off
        this_turn = hmask & (h[..., 11] == 0)
        ns = this_turn & ((h[..., 0] == NS_KINDS[0]) | (h[..., 0] == NS_KINDS[1]))
        ns_me = (ns & (h[..., 10] == 0)).any(1).long()
        ns_op = (ns & (h[..., 10] == 1)).any(1).long()
        return (self.g_phase(phase_index(glob[:, 3])) + self.g_turn(glob[:, 2].clamp(0, 31))
                + self.g_first(glob[:, 4].clamp(0, 1)) + self.g_mine(glob[:, 5].clamp(0, 1))
                + self.g_chain(glob[:, 20].clamp(0, 16)) + self.g_lp(lp).sum(1) + self.g_cnt(cnt).sum(1)
                + self.g_ns(ns_me) + self.g_ns(2 + ns_op))  # fmt: skip

    def _card_code(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        code = obs["cards"][..., 0].long()
        if self.known:
            code = torch.where(obs["card_known"].long() == 1, obs["known_code"].long(), code)
        return code

    # ---- encoder -------------------------------------------------------------------------------------------
    def encode(self, obs: dict[str, torch.Tensor]):
        n_opt = 0
        if self.trim:
            obs, n_opt = trim_obs(obs)
        cards, glob = obs["cards"], obs["global"].long()
        B, n_cards = cards.shape[0], cards.shape[1]
        h, hmask = obs["history"].long(), obs["history_mask"]
        this_turn_me = hmask & (h[..., 11] == 0) & (h[..., 10] == 0) & (h[..., 0] == ACT)
        code = self._card_code(obs)
        used = ((code.unsqueeze(-1) == h[..., 2].unsqueeze(1)) & this_turn_me.unsqueeze(1)).sum(-1)
        card_tok = (self.encode_cards(cards, obs.get("card_known"), obs.get("known_code")) + self.type_emb.weight[1]
                    + self.card_cnt(used.clamp(0, 3)))  # fmt: skip
        num = glob[:, :22].float()
        num = torch.cat([num[:, :2] / 8000.0, num[:, 2:20] / 40.0, num[:, 20:22] / 64.0], -1)
        g = (self.glob_tok + self.glob_num(num) + self.glob_msg(glob[:, 21].clamp(0, 63)) + self.type_emb.weight[0]
             + self._glob_cat(glob, h, hmask))  # fmt: skip
        ch = obs["chain"].long()
        chain_tok = (self.text(ch[..., 1]) + self.chain_ctrl(ch[..., 4].clamp(0, 1)) + self.type_emb.weight[2]
                     + self.desc(ch[..., 2], ch[..., 3]))  # fmt: skip
        if self.hist > 0:
            age = self.hist_age(h[..., 11].clamp(0, 31))
            pos = self.hist_pos.weight[: h.shape[1]]
        else:
            age = self.hist_age(h[..., 11].clamp(0, 7))
            pos = 0.0
        hist_tok = (self.action(h[..., :10]) + self.hist_actor(h[..., 10].clamp(0, 1)) + age + self.type_emb.weight[3]
                    + pos + self.desc(h[..., 3], h[..., 4]) + self.hist_phase(phase_index(h[..., 12])))  # fmt: skip
        tokens = torch.cat([g.unsqueeze(1), card_tok, chain_tok, hist_tok], 1)
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=cards.device), obs["card_mask"],
                          obs["chain_mask"], hmask], 1)  # fmt: skip
        keys, pad = self.eff_keys(code)
        x = tokens
        for i, layer in enumerate(self.encoder.layers):
            x = layer(x, src_key_padding_mask=~keep)
            if i in self.SENT_AFTER:
                j = self.SENT_AFTER.index(i)
                cx = self.card_sent[j](x[:, 1 : 1 + n_cards], keys, pad)
                x = torch.cat([x[:, :1], cx, x[:, 1 + n_cards :]], 1)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        return obs, x, keep, n_opt

    # ---- option head ---------------------------------------------------------------------------------------
    def _opt_counts(self, a: torch.Tensor, h: torch.Tensor, hmask: torch.Tensor) -> torch.Tensor:
        has = (a[..., 3] > 0) | (a[..., 4] > 0)
        same = ((a[..., 2].unsqueeze(-1) == h[..., 2].unsqueeze(1)) & (a[..., 3].unsqueeze(-1) == h[..., 3].unsqueeze(1))
                & (a[..., 4].unsqueeze(-1) == h[..., 4].unsqueeze(1))
                & (hmask & (h[..., 0] == ACT)).unsqueeze(1) & has.unsqueeze(-1))  # fmt: skip
        now = (h[..., 11] == 0).unsqueeze(1)
        out = 0.0
        for s, (actor, turn_only) in enumerate(((0, True), (0, False), (1, True), (1, False))):
            m = same & (h[..., 10] == actor).unsqueeze(1)
            if turn_only:
                m = m & now
            out = out + self.opt_cnt(s * 4 + m.sum(-1).clamp(0, 3))
        return out

    def _run_features(self, obs: dict[str, torch.Tensor], want_q: bool = False):
        obs, enc, keep, n_opt = self.encode(obs)
        cards = obs["cards"]
        a = obs["actions"].long()
        h, hmask = obs["history"].long(), obs["history_mask"]
        act = self.action(a) + self.desc(a[..., 3], a[..., 4]) + self._opt_counts(a, h, hmask)
        owner = torch.where(a[..., 3] > 0, a[..., 3], a[..., 2])  # card whose effect this is
        keys, pad = self.eff_keys(owner)
        act = self.opt_sent(act, keys, pad)
        row = a[..., 1]
        n_cards = cards.shape[1]
        card_ctx = torch.gather(enc[:, 1 : 1 + n_cards], 1,
                                row.clamp(0, n_cards - 1).unsqueeze(-1).expand(-1, -1, enc.shape[-1]))  # fmt: skip
        card_ctx = card_ctx * (row >= 0).unsqueeze(-1)
        q = self.act_norm(act + card_ctx + self.ctx(enc[:, :1]))
        att, _ = self.cross(q, enc, enc, key_padding_mask=~keep)
        glob_ctx = enc[:, :1].expand(-1, a.shape[1], -1)
        feat = torch.cat([q + att, card_ctx, glob_ctx], -1)
        logits = self.head(feat).squeeze(-1).masked_fill(~obs["action_mask"], float("-inf"))
        resid = self.q_head(feat).squeeze(-1) if want_q else None
        legal = obs["action_mask"]
        if n_opt > logits.shape[1]:
            pad_n = n_opt - logits.shape[1]
            logits = nn.functional.pad(logits, (0, pad_n), value=float("-inf"))
            legal = nn.functional.pad(legal, (0, pad_n), value=False)
            if resid is not None:
                resid = nn.functional.pad(resid, (0, pad_n), value=0.0)
        return logits, resid, enc, n_opt, legal


ARCH_KEYS = ("d", "layers", "heads", "hist", "known", "v3", "desc_strings", "desc_lua")  # copied into derived runs' args


def make_policy(emb: dict[str, np.ndarray] | str | Path, a: dict, **kw) -> PolicyNet:
    """Build the PolicyNet a run's args describe (`a` = metrics.jsonl args; `kw` overrides, e.g. value=True).

    Every constructor site goes through here so a v3 checkpoint loads wherever a v1 one does."""
    emb = dict(np.load(emb)) if isinstance(emb, (str, Path)) else dict(emb)
    base = {"d": a.get("d", 256), "layers": a.get("layers", 4), "heads": a.get("heads", 4),
            "hist": int(a.get("hist", 0) or 0), "known": bool(a.get("known", False))}  # fmt: skip
    base.update(kw)
    if a.get("v3"):
        emb.update(load_desc_tables(a.get("desc_strings") or None, a.get("desc_lua") or None))
        return PolicyNetV3(emb, **base)
    return PolicyNet(emb, **base)
