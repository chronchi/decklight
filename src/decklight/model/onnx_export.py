"""ONNX export of the product policy/value nets (offline-app milestone A0, spike S2).

Design (the contract the Rust core relies on):
- Two graphs per network. ``card_proj.onnx`` is the frozen text MLP ``proj`` alone:
  ``raw`` float32 [M, R] -> ``text`` float32 [M, d]. The app runs it once at load
  (and once per card-pack row batch) and builds the lookup table as
  ``text = concat([proj(unknown), proj(raw_rows)])``; the raw ``unknown`` vector
  ships next to the graphs as ``unknown.npy`` (it is a learned checkpoint
  parameter, not part of the embedding npz, so the graphs alone cannot produce
  row 0).
- ``policy.onnx`` / ``value.onnx`` take every obs tensor the model reads (same
  names, shapes and dtypes as the training obs) with the code columns replaced
  by row indices (see :data:`CODE_COLUMNS` / :func:`codes_to_rows`), plus
  ``text`` float32 [T, d] (row 0 = ``proj(unknown)``, row ``i >= 1`` = card
  ``i``'s projected vector; axis 0 is dynamic so card packs can grow T).
  Inside the graph, card lookups are plain ``Gather`` on ``text`` by row index:
  no ``searchsorted``, no embedding table in the graph, so new cards need no
  new model file.

Only PolicyNet v1 (``hist=0``, ``known=False``, not v3) is supported: that is
what the product checkpoints are, and the v1 encoder replicas below would
silently diverge on v3 inputs. Anything else raises.

An observable-history net (``PolicyNet(observable=True)``) exports the same v1
graph: only the source of the ``history`` / ``history_mask`` inputs changes.
The policy graph's ``metadata_props`` carries ``history_source`` = ``obs_history``
for such a net (``history`` otherwise, and absent means ``history`` for old
files), so the runtime feeds the client-visible window into the same inputs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from decklight.model.bc_data import OBS_KEYS
from decklight.model.policy import N_KINDS, TRIM_TO, CardText

#: Every obs tensor column that holds a card code and is passed to
#: ``CardText`` (directly, or through ``ActionEmbed`` which both the option
#: head and the history encoder share). ``cards`` col 0 (``encode_cards``),
#: ``chain`` col 1 (chain link handler), ``actions`` cols 2-3,
#: ``history`` cols 2-3 (``code`` and ``effect_code`` via ``ActionEmbed``) and
#: the observable window ``obs_history`` cols 2-3 (fed into ``history`` for an
#: observable net). Columns holding card *rows* (``actions``/``history`` col 1,
#: ``chain`` col 0) are indices into the card tokens, not codes, and are untouched.
CODE_COLUMNS = [
    {"key": "cards", "col": 0},
    {"key": "chain", "col": 1},
    {"key": "actions", "col": 2},
    {"key": "actions", "col": 3},
    {"key": "history", "col": 2},
    {"key": "history", "col": 3},
    {"key": "obs_history", "col": 2},
    {"key": "obs_history", "col": 3},
]

CODES_VERSION = 1


def sorted_codes(emb: dict[str, np.ndarray]) -> np.ndarray:
    """The card-code vocabulary, int64 and sorted (the npz already is)."""
    return np.asarray(emb["codes"]).astype(np.int64).ravel()


def raw_table(emb: dict[str, np.ndarray]) -> np.ndarray:
    """Frozen text rows ``[N, R]`` (whole-card vector + mean effect vector).

    Mirrors ``CardText.__init__`` exactly: ``R = 2 * dim``.
    """
    card = np.asarray(emb["card"]).astype(np.float32)
    effects = np.asarray(emb["effects"]).astype(np.float32)
    n_eff = np.maximum(np.asarray(emb["n_effects"]).astype(np.float32), 1.0)[:, None]
    eff_mean = effects.sum(1) / n_eff
    return np.concatenate([card, eff_mean], axis=1)


def codes_to_rows(
    obs: dict[str, np.ndarray] | dict[str, torch.Tensor], codes: np.ndarray
) -> dict[str, np.ndarray] | dict[str, torch.Tensor]:
    """Copy of ``obs`` with every :data:`CODE_COLUMNS` entry replaced by a row index.

    ``codes`` is the sorted int64 vocabulary; row ``1 + position`` holds that
    card's projected vector, row 0 is the unknown vector. Codes missing from
    the table, ``code <= 0`` (hidden / none) and negative codes all map to 0.
    Accepts numpy or torch obs values (torch stays on its device/dtype).
    """
    codes = np.asarray(codes).astype(np.int64).ravel()
    out = dict(obs)
    for entry in CODE_COLUMNS:
        key, col = entry["key"], entry["col"]
        if key not in out:
            continue
        arr = out[key]
        if isinstance(arr, torch.Tensor):
            c = arr[..., col].to(torch.int64)
            idx = torch.searchsorted(torch.from_numpy(codes).to(c.device), c.clamp(min=0))
            idx = idx.clamp(max=len(codes) - 1)
            hit = (torch.from_numpy(codes).to(c.device)[idx] == c.clamp(min=0)) & (c > 0)
            rows = torch.where(hit, idx + 1, torch.zeros_like(idx))
            cp = arr.clone()
            cp[..., col] = rows.to(arr.dtype)
            out[key] = cp
        else:
            a = np.asarray(arr)
            c = a[..., col].astype(np.int64)
            idx = np.searchsorted(codes, np.maximum(c, 0)).astype(np.int64)
            idx = np.minimum(idx, len(codes) - 1)
            hit = (codes[idx] == np.maximum(c, 0)) & (c > 0)
            rows = np.where(hit, idx + 1, 0)
            cp = a.copy()
            cp[..., col] = rows.astype(a.dtype)
            out[key] = cp
    return out


def project_table(text_mod: CardText, raw: np.ndarray) -> np.ndarray:
    """Full lookup table ``[N + 1, d]``: row 0 = ``proj(unknown)``, then ``proj(raw)``."""
    unk = text_mod.unknown.detach().float().cpu().numpy().reshape(1, -1)
    with torch.no_grad():
        proj = text_mod.proj.float().eval()
        t0 = proj(torch.from_numpy(unk)).numpy()
        t1 = proj(torch.from_numpy(np.ascontiguousarray(raw))).numpy()
    return np.concatenate([t0, t1], axis=0).astype(np.float32)


def history_source(net: nn.Module) -> str:
    """The obs key the exported ``history`` / ``history_mask`` inputs must be fed from."""
    return "obs_history" if getattr(net, "observable", False) else "history"


def with_history_source(
    obs: dict[str, np.ndarray] | dict[str, torch.Tensor], source: str
) -> dict[str, np.ndarray] | dict[str, torch.Tensor]:
    """Copy of ``obs`` with ``history`` / ``history_mask`` taken from ``source``.

    The ONNX graph always has ``history`` / ``history_mask`` inputs; for an observable net the runtime
    feeds ``obs_history`` into them. ``source == "history"`` leaves the dict unchanged.
    """
    if source == "history":
        return obs
    if source not in ("obs_history",):
        raise ValueError(f"unknown history source {source!r}")
    if "obs_history" not in obs:
        raise KeyError(f"history source {source!r}: the obs needs obs_history/obs_history_mask")
    return {**obs, "history": obs["obs_history"], "history_mask": obs["obs_history_mask"]}


def read_metadata(path: Path) -> dict[str, str]:
    """The ONNX ``metadata_props`` of a graph (custom keys such as ``history_source``)."""
    import onnx

    model = onnx.load(str(path))
    return {p.key: p.value for p in model.metadata_props}


def _set_metadata(path: Path, props: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(path))
    del model.metadata_props[:]
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))


def _check_v1(net: nn.Module) -> None:
    if type(net).__name__ != "PolicyNet" or int(getattr(net, "hist", 0)) != 0 or bool(
        getattr(net, "known", False)
    ):
        raise ValueError(
            f"ONNX export supports PolicyNet v1 only (hist=0, known=False, not v3); got {type(net).__name__} "
            f"(hist={getattr(net, 'hist', '?')}, known={getattr(net, 'known', '?')})"
        )


def _gather_text(table: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``table[row]``: a plain ``Gather``. Rows come from :func:`codes_to_rows`, so they are in ``[0, T)``; an
    out-of-range row is a caller bug and the runtime rejects it (no clamp: a clamp bound to the dynamic table size
    is a ``Clip`` that tract cannot type, NOTEBOOK 2026-11-01)."""
    return table[rows.long()]


def _action_embed(net: nn.Module, a: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``ActionEmbed.forward`` with ``Gather`` in place of ``CardText``."""
    act = net.action
    kind, code, eff_code, eff_idx, arg, msg, loc, ctrl, step = (
        a[..., i] for i in (0, 2, 3, 4, 5, 6, 7, 8, 9)
    )
    return (
        act.kind(kind.clamp(0, N_KINDS - 1))
        + _gather_text(table, code)
        + act.eff_owner(_gather_text(table, eff_code))
        + act.eff(eff_idx.clamp(0, 31))
        + act.arg(arg.clamp(0, 64))
        + act.msg(msg.clamp(0, 63))
        + act.loc(loc.clamp(0, 15))
        + act.ctrl((ctrl + 1).clamp(0, 2))
        + act.step(step.clamp(0, 15))
    )


def _bits(x: torch.Tensor, n: int) -> torch.Tensor:
    """Low n bits of an int tensor as floats [..., n].

    ``policy.py`` uses ``((x >> s) & 1)``; the torch ONNX exporter has no
    symbolic for bitwise AND on ints, so this uses ``(u // 2**s) % 2`` on
    ``u = x mod 2**32``, the field's unsigned 32-bit pattern (n <= 32). ``u`` is
    never negative, so ONNX's truncating integer ``Div`` equals torch's floor
    division (on a negative ``x`` it would not: tests/unit/test_onnx_export.py).
    """
    u = _u32(x)
    div = u.unsqueeze(-1) // (2 ** torch.arange(n, device=x.device))
    return (div % 2).float()


def _u32(x: torch.Tensor) -> torch.Tensor:
    """int32 field -> its unsigned bit pattern as int64 (ONNX ``Mod`` with fmod=0 keeps the divisor's sign)."""
    return x.long() % 4294967296


def _encode_cards(net: nn.Module, cards: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``PolicyNet.encode_cards`` with ``Gather`` in place of ``CardText`` (known=False)."""
    c = cards.int()
    # policy.py: ``c[..., 18] & 0xFF`` / ``(c[..., 18] >> 8) & 0xFF``; same rewrite as _bits.
    stats = torch.stack(
        [c[..., 14], c[..., 15], c[..., 16], c[..., 17], _u32(c[..., 18]) % 256, (_u32(c[..., 18]) // 256) % 256],
        -1,
    ).float()
    stats = torch.where(stats < 0, torch.zeros_like(stats), stats) / torch.tensor(
        [3000.0, 3000.0, 3000.0, 3000.0, 13.0, 13.0], device=c.device
    )
    bits = torch.cat(
        [_bits(c[..., 7], 32), _bits(c[..., 8], 8), _bits(c[..., 9], 32),
         _bits(c[..., 13], 9), _bits(c[..., 19], 16)], -1,
    )  # fmt: skip
    return (
        _gather_text(table, c[..., 0])
        + net.loc(c[..., 1].clamp(0, 15))
        + net.seq(c[..., 2].clamp(0, 32))
        + net.ctrl(c[..., 3].clamp(0, 1))
        + net.owner(c[..., 4].clamp(0, 1))
        + net.pos(c[..., 5].clamp(0, 7))
        + net.pub(c[..., 6].clamp(0, 1))
        + net.lvl(c[..., 10].clamp(0, 15))
        + net.rank(c[..., 11].clamp(0, 15))
        + net.link(c[..., 12].clamp(0, 7))
        + net.bits(bits)
        + net.stats(stats)
        + net.sel(c[..., 20].clamp(0, 1))
    )


def _mha(m: nn.MultiheadAttention, q: torch.Tensor, kv: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """``m(q, kv, kv, key_padding_mask=pad)[0]`` (batch_first, no dropout) written out.

    The legacy exporter traces ``F.multi_head_attention_forward`` with the token count frozen into a ``Reshape``
    (``tgt_len``), which breaks the dynamic token axes; here every shape comes from the inputs."""
    d, h = m.embed_dim, m.num_heads
    hd = d // h
    w, b = m.in_proj_weight, m.in_proj_bias

    def heads(x: torch.Tensor) -> torch.Tensor:  # [B, L, d] -> [B, h, L, hd]
        return x.unflatten(-1, (h, hd)).transpose(1, 2)

    qh = heads(nn.functional.linear(q, w[:d], b[:d])) * hd**-0.5
    kh = heads(nn.functional.linear(kv, w[d : 2 * d], b[d : 2 * d]))
    vh = heads(nn.functional.linear(kv, w[2 * d :], b[2 * d :]))
    att = (qh @ kh.transpose(-1, -2)).masked_fill(pad[:, None, None, :], float("-inf")).softmax(-1)
    return m.out_proj((att @ vh).transpose(1, 2).flatten(2))


def _encoder(enc: nn.TransformerEncoder, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """``enc(x, src_key_padding_mask=pad)`` for pre-norm layers (``norm_first=True``, as PolicyNet builds them)."""
    for layer in enc.layers:
        if not layer.norm_first:
            raise ValueError("ONNX export expects norm_first encoder layers")
        y = layer.norm1(x)
        x = x + _mha(layer.self_attn, y, y, pad)
        x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
    return enc.norm(x) if enc.norm is not None else x


def _encode(net: nn.Module, cards, card_mask, glob, chain, chain_mask, history, history_mask, table):
    """``PolicyNet.encode`` with ``trim`` off (fixed sizes) and ``Gather`` text lookups."""
    B = cards.shape[0]
    card_tok = _encode_cards(net, cards, table) + net.type_emb.weight[1]
    g = glob.int()
    num = g[:, :22].float()
    num = torch.cat([num[:, :2] / 8000.0, num[:, 2:20] / 40.0, num[:, 20:22] / 64.0], -1)
    gtok = net.glob_tok + net.glob_num(num) + net.glob_msg(g[:, 21].clamp(0, 63)) + net.type_emb.weight[0]
    ch = chain.int()
    chain_tok = (
        _gather_text(table, ch[..., 1]) + net.chain_ctrl(ch[..., 4].clamp(0, 1)) + net.type_emb.weight[2]
    )
    h = history.int()
    age = net.hist_age(h[..., 11].clamp(0, 7))
    hist_tok = (
        _action_embed(net, h[..., :10], table)
        + net.hist_actor(h[..., 10].clamp(0, 1))
        + age
        + net.type_emb.weight[3]
    )
    tokens = torch.cat([gtok.unsqueeze(1), card_tok, chain_tok, hist_tok], 1)
    keep = torch.cat(
        [torch.ones(B, 1, dtype=torch.bool, device=cards.device), card_mask, chain_mask, history_mask], 1
    )
    enc = _encoder(net.encoder, tokens, ~keep)
    return enc, keep


def _logits_from(net: nn.Module, enc, keep, cards, actions, action_mask, table):
    a = actions.int()
    act = _action_embed(net, a, table)
    n_cards = cards.shape[1]
    # policy.py clamps the row to n_cards - 1; a legal option's row is a real card (< the used length, which
    # trim_obs_np checks), so only the lower clamp is kept (a Clip bound to a dynamic size breaks tract).
    card_ctx = torch.gather(
        enc[:, 1 : 1 + n_cards],
        1,
        a[..., 1].clamp(min=0).long().unsqueeze(-1).expand(-1, -1, enc.shape[-1]),
    )
    card_ctx = card_ctx * (a[..., 1] >= 0).unsqueeze(-1)
    q = net.act_norm(act + card_ctx)
    att = _mha(net.cross, q, enc, ~keep)
    glob_ctx = enc[:, :1] + 0.0 * q  # broadcast over options (tract cannot infer the Expand of policy.py)
    feat = torch.cat([q + att, card_ctx, glob_ctx], -1)
    logits = net.head(feat).squeeze(-1)
    return logits.masked_fill(~action_mask, float("-inf"))


class ExportCardProj(nn.Module):
    """The frozen text MLP ``proj`` alone: ``raw`` [M, R] -> ``text`` [M, d]."""

    def __init__(self, text: CardText):
        super().__init__()
        self.proj = text.proj

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.proj(raw.float())


class ExportPolicy(nn.Module):
    """Policy logits from row-index obs + the projected text table."""

    def __init__(self, net: nn.Module):
        super().__init__()
        _check_v1(net)
        net.eval()
        net.requires_grad_(False)  # trace bakes weights as constants
        net.trim = False  # fixed-size graph; trim is a no-op on logits (policy.py)
        object.__setattr__(self, "src", net)  # not a submodule: keeps the frozen table out of the graph

    def forward(self, cards, card_mask, glob, chain, chain_mask, actions, action_mask,
                history, history_mask, text) -> torch.Tensor:  # fmt: skip
        net = self.src
        enc, keep = _encode(net, cards, card_mask, glob, chain, chain_mask, history, history_mask, text)
        return _logits_from(net, enc, keep, cards, actions, action_mask, text)


class ExportValue(nn.Module):
    """State value from row-index obs + the projected text table."""

    def __init__(self, net: nn.Module):
        super().__init__()
        _check_v1(net)
        if not hasattr(net, "value_head"):
            raise ValueError("this checkpoint has no value head (need a value checkpoint)")
        net.eval()
        net.requires_grad_(False)  # trace bakes weights as constants
        net.trim = False
        object.__setattr__(self, "src", net)

    def forward(self, cards, card_mask, glob, chain, chain_mask, actions, action_mask,
                history, history_mask, text) -> torch.Tensor:  # fmt: skip
        net = self.src
        enc, _ = _encode(net, cards, card_mask, glob, chain, chain_mask, history, history_mask, text)
        return net.value_head(enc[:, 0]).squeeze(-1)


POLICY_INPUTS = list(OBS_KEYS) + ["text"]

#: Axis 1 of these inputs may be cut to the batch's used length (``trim_obs`` in policy.py).
TOKEN_AXES = {"cards": "n_cards", "card_mask": "n_cards", "chain": "n_chain", "chain_mask": "n_chain",
              "actions": "n_opt", "action_mask": "n_opt", "history": "n_hist", "history_mask": "n_hist"}


def trim_obs_np(obs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    """numpy twin of ``policy.trim_obs`` (same lengths, rounded up to ``TRIM_TO``); returns the trimmed obs and the
    full option count. Pad the logits back with :func:`pad_logits`."""
    def used(mask: np.ndarray) -> int:
        any_ = np.asarray(mask).any(0)
        n = int(np.flatnonzero(any_).max()) + 1 if any_.any() else 1
        return min(mask.shape[1], -(-n // TRIM_TO) * TRIM_TO)

    o = dict(obs)
    n_opt = obs["action_mask"].shape[1]
    lens = {"n_cards": used(obs["card_mask"]), "n_chain": used(obs["chain_mask"]),
            "n_hist": used(obs["history_mask"]), "n_opt": used(obs["action_mask"])}
    for k, ax in TOKEN_AXES.items():
        if k in o:
            o[k] = np.ascontiguousarray(o[k][:, : lens[ax]])
    rows = o["actions"][..., 1][o["action_mask"]]
    if rows.size and int(rows.max()) >= lens["n_cards"]:
        raise ValueError(f"a legal option points at card row {int(rows.max())} >= {lens['n_cards']} used rows")
    return o, n_opt


def pad_logits(logits: np.ndarray, n_opt: int) -> np.ndarray:
    """Trimmed logits [B, n] back to [B, n_opt] with -inf on the cut options."""
    out = np.full((logits.shape[0], n_opt), -np.inf, dtype=logits.dtype)
    out[:, : logits.shape[1]] = logits
    return out

_DTYPE_EXAMPLE = {
    "cards": torch.int32,
    "card_mask": torch.bool,
    "global": torch.int32,
    "chain": torch.int32,
    "chain_mask": torch.bool,
    "actions": torch.int32,
    "action_mask": torch.bool,
    "history": torch.int32,
    "history_mask": torch.bool,
}


def example_obs(
    batch: int = 2,
    n_cards: int = 160,
    n_chain: int = 16,
    n_opt: int = 64,
    n_hist: int = 32,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Zero obs with the global row valid (trace-safe example inputs)."""
    return {
        "cards": torch.zeros(batch, n_cards, 21, dtype=torch.int32),
        "card_mask": torch.zeros(batch, n_cards, dtype=torch.bool),
        "global": torch.zeros(batch, 24, dtype=torch.int32),
        "chain": torch.zeros(batch, n_chain, 5, dtype=torch.int32),
        "chain_mask": torch.zeros(batch, n_chain, dtype=torch.bool),
        "actions": torch.zeros(batch, n_opt, 10, dtype=torch.int32),
        "action_mask": torch.zeros(batch, n_opt, dtype=torch.bool),
        "history": torch.zeros(batch, n_hist, 13, dtype=torch.int32),
        "history_mask": torch.zeros(batch, n_hist, dtype=torch.bool),
    }


def _prune_initializers(path: Path) -> int:
    """Drop initializers no node reads (e.g. the frozen text table); returns the count removed."""
    import onnx

    model = onnx.load(str(path))
    used: set[str] = set()
    for node in model.graph.node:
        used.update(node.input)
    kept = [i for i in model.graph.initializer if i.name in used]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        onnx.save(model, str(path))
    return removed


def export_nets(
    policy_net: nn.Module,
    value_net: nn.Module,
    dims: tuple[int, int, int],
    out_dir: Path,
    opset: int = 17,
    ex: dict[str, torch.Tensor] | None = None,
) -> dict[str, Path]:
    """Export the three graphs; returns the written paths (no sidecars)."""
    import torch.onnx

    n_table, d, r = dims
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = out_dir / "policy.onnx"
    value_path = out_dir / "value.onnx"
    proj_path = out_dir / "card_proj.onnx"

    ex = dict(ex) if ex is not None else example_obs()
    args = tuple(ex[k] for k in OBS_KEYS)
    text_ex = torch.zeros(n_table, d, dtype=torch.float32)
    # Token axes are dynamic too: the caller trims the all-padding tail first (:func:`trim_obs_np`, as
    # ``PolicyNet.trim`` does), which is ~2-3x faster than the fixed padded shapes.
    dyn = {k: {0: "batch", **({1: TOKEN_AXES[k]} if k in TOKEN_AXES else {})} for k in OBS_KEYS}
    dyn["text"] = {0: "table"}

    # The fused ``_transformer_encoder_layer_fwd`` has no ONNX symbolic; the
    # slow math path is functionally identical (checked by the parity gates).
    prev_fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        ep, ev = ExportPolicy(policy_net), ExportValue(value_net)
        torch.onnx.export(
            ep, args + (text_ex,), str(policy_path),
            input_names=POLICY_INPUTS, output_names=["logits"], dynamic_axes={**dyn, "logits": {0: "batch", 1: "n_opt"}},
            opset_version=opset, dynamo=False,
        )  # fmt: skip
        torch.onnx.export(
            ev, args + (text_ex,), str(value_path),
            input_names=POLICY_INPUTS, output_names=["value"], dynamic_axes={**dyn, "value": {0: "batch"}},
            opset_version=opset, dynamo=False,
        )  # fmt: skip
    finally:
        torch.backends.mha.set_fastpath_enabled(prev_fast)
    # NOTE: one projection graph per checkpoint: policy and value were trained
    # separately and their proj/unknown weights differ (max |dW| ~ 0.04), so a
    # single card_proj.onnx cannot serve both inference graphs.
    torch.onnx.export(
        ExportCardProj(policy_net.text), (torch.zeros(8, r, dtype=torch.float32),), str(proj_path),
        input_names=["raw"], output_names=["text"],
        dynamic_axes={"raw": {0: "rows"}, "text": {0: "rows"}}, opset_version=opset, dynamo=False,
    )  # fmt: skip
    proj_value_path = out_dir / "card_proj_value.onnx"
    torch.onnx.export(
        ExportCardProj(value_net.text), (torch.zeros(8, r, dtype=torch.float32),), str(proj_value_path),
        input_names=["raw"], output_names=["text"],
        dynamic_axes={"raw": {0: "rows"}, "text": {0: "rows"}}, opset_version=opset, dynamo=False,
    )  # fmt: skip
    for p in (policy_path, value_path, proj_path, proj_value_path):
        _prune_initializers(p)
    # The graph is identical for an observable net; the metadata names the obs key its history inputs come from
    # (absent = "history", so old exports keep working).
    _set_metadata(policy_path, {"history_source": history_source(policy_net)})
    _set_metadata(value_path, {"history_source": history_source(value_net)})
    return {"policy": policy_path, "value": value_path, "card_proj": proj_path,
            "card_proj_value": proj_value_path}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
