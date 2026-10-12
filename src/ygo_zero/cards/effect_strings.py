"""Effect-string tables keyed by the engine's description id (retrain plan v3, stage 0f; model v3 input).

Options and chain links carry a description `desc = (code << 20) | idx` (obs columns `desc >> 20`, `desc & 0xFFFFF`):
- `code > 0`: card string `str{idx+1}` of card `code` (BabelCDB `texts`), e.g. Ash Blossom str1 "Negate that effect";
- `code == 0`: system string `idx` (`!system` lines of edopro-strings.conf), e.g. 1 "Normal Summon".

Two tables, both keyed by `key = (code << 20) | idx` (= the raw description):
- `strings_<tag>.npz`: one embedding per string (same model as card texts; identical texts share one computation);
- `lua_desc.npz`: per card string, the OR of the Lua EffectRecords whose `SetDescription(aux.Stringid(id, idx))` is
  that string (type, category, range, property, count limit, cond/cost/target flags). Ground truth from the scripts
  the engine runs; descriptions not set in Lua (system strings, hints) have no row.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from ygo_zero.cards.embed import EMB_DIR, MODEL, ROOT, pick_device
from ygo_zero.cards.lua_effects import OFFICIAL_DIR, load_constants, parse_script
from ygo_zero.engine.resources import default_cdbs

STRINGS_CONF = ROOT / "third_party" / "edopro-strings.conf"
# lua_desc columns (int64): bit fields are expanded by the model
LUA_COLS = ("type", "category", "range", "property", "count_n", "count_flags", "flags", "n_records")
F_COND, F_COST, F_TARGET, F_OP, F_OATH = 1, 2, 4, 8, 16


def desc_key(code: int, idx: int) -> int:
    return (int(code) << 20) | int(idx)


def card_strings(cdbs: list[Path] | None = None) -> dict[int, str]:
    """key -> text for every non-empty str1..str16 (later cdb files override earlier ones)."""
    out: dict[int, str] = {}
    for path in cdbs if cdbs is not None else default_cdbs():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            cols = ", ".join(f"str{i}" for i in range(1, 17))
            for row in con.execute(f"SELECT id, {cols} FROM texts"):
                for i, s in enumerate(row[1:]):
                    if s and s.strip():
                        out[desc_key(row[0], i)] = s.strip()
        finally:
            con.close()
    return out


def system_strings(path: Path = STRINGS_CONF) -> dict[int, str]:
    """key (= system id, code 0) -> text of every `!system <id> <text>` line."""
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("!system "):
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[1].isdigit() and parts[2].strip():
                out[desc_key(0, int(parts[1]))] = parts[2].strip()
    return out


def all_strings() -> dict[int, str]:
    return {**system_strings(), **card_strings()}


def lua_desc_table(scripts_dir: Path = OFFICIAL_DIR) -> tuple[np.ndarray, np.ndarray]:
    """(keys int64 [N], feats int64 [N, len(LUA_COLS)]) over every official card script."""
    consts = load_constants()
    rows: dict[int, list[int]] = {}
    for p in sorted(scripts_dir.glob("c*.lua")):
        stem = p.stem[1:]
        if not stem.isdigit():
            continue
        code = int(stem)
        for r in parse_script(p.read_text(encoding="utf-8", errors="replace"), code, consts):
            if r.description is None:
                continue
            k = desc_key(code, r.description)
            f = rows.setdefault(k, [0] * len(LUA_COLS))
            f[0] |= r.type
            f[1] |= r.category
            f[2] |= r.range
            f[3] |= r.property
            if r.count_limit:
                f[4] = max(f[4], int(r.count_limit[0]))
                f[5] |= int(len(r.count_limit) > 1)  # once per turn by card id / flags (hard OPT)
            f[6] |= (F_COND * r.has_condition) | (F_COST * r.has_cost) | (F_TARGET * r.has_target) | (
                F_OP * r.has_operation)  # fmt: skip
            f[7] += 1
    keys = np.array(sorted(rows), dtype=np.int64)
    feats = np.array([rows[int(k)] for k in keys], dtype=np.int64).reshape(len(keys), len(LUA_COLS))
    return keys, feats


def build_lua(out: Path | None = None) -> Path:
    t0 = time.time()
    keys, feats = lua_desc_table()
    if len(keys) == 0:
        raise SystemExit("ALERT effect_strings: 0 Lua descriptions parsed")
    out = out or EMB_DIR.parent / "lua_desc.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=keys, feats=feats, cols=np.array(LUA_COLS))
    print(f"wrote {out}: {len(keys)} descriptions, {time.time() - t0:.0f}s", flush=True)
    return out


def build_strings(dim: int = 512, batch: int = 128, limit: int | None = None, tag: str | None = None,
                  out_dir: Path | None = None) -> Path:  # fmt: skip
    """Embed every card/system string -> `<out_dir>/strings-<tag>.npz` (keys, vecs fp16 L2-normalised)."""
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "hf"))
    from sentence_transformers import SentenceTransformer

    strings = all_strings()
    keys = sorted(strings)[:limit]
    uniq: dict[str, int] = {}
    rows = [uniq.setdefault(strings[k], len(uniq)) for k in keys]
    table = list(uniq)
    device = pick_device()
    print(f"device {device}; {len(keys)} strings, {len(table)} unique", flush=True)
    model = SentenceTransformer(MODEL, device=device, truncate_dim=dim)
    t0 = time.time()
    vecs = model.encode(table, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False).astype(np.float16)  # fmt: skip
    if not np.isfinite(vecs).all() or (np.abs(vecs).sum(1) == 0).any():
        raise SystemExit("ALERT effect_strings: non-finite or zero string vectors")
    out_dir = out_dir or EMB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"strings-{tag or f'qwen3-0.6b-d{dim}'}.npz"
    np.savez_compressed(out, keys=np.array(keys, dtype=np.int64), rows=np.array(rows, dtype=np.int32), vecs=vecs,
                        model=MODEL, dim=dim)  # fmt: skip
    print(f"wrote {out}: {len(keys)} keys, {len(table)} vectors, {time.time() - t0:.0f}s", flush=True)
    return out


class DescTables:
    """Lookup of desc keys -> (string row + 1 or 0, lua row + 1 or 0) for the model's embedding tables."""

    def __init__(self, strings: Path, lua: Path) -> None:
        z = np.load(strings)
        self.str_keys, self.str_rows, self.vecs = z["keys"], z["rows"], z["vecs"]
        y = np.load(lua)
        self.lua_keys, self.lua_feats = y["keys"], y["feats"]

    @staticmethod
    def _lookup(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
        i = np.searchsorted(sorted_keys, keys).clip(0, max(len(sorted_keys) - 1, 0))
        return np.where(sorted_keys[i] == keys, i + 1, 0) if len(sorted_keys) else np.zeros_like(keys)

    def string_rows(self, hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
        """Vector row + 1 per (desc_hi, desc_lo) (0 = no string)."""
        k = (np.asarray(hi, dtype=np.int64) << 20) | np.asarray(lo, dtype=np.int64)
        j = self._lookup(self.str_keys, k)
        return np.where(j > 0, self.str_rows[np.maximum(j - 1, 0)].astype(np.int64) + 1, 0)

    def lua_rows(self, hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
        k = (np.asarray(hi, dtype=np.int64) << 20) | np.asarray(lo, dtype=np.int64)
        return self._lookup(self.lua_keys, k)
