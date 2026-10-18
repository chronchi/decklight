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
from ygo_zero.engine.resources import default_cdbs, extra_script_dirs

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


def set_names(path: Path = STRINGS_CONF) -> dict[int, str]:
    """setcode -> archetype name of every `!setname 0x<code> <name>` line."""
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("!setname "):
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[2].strip():
                try:
                    out[int(parts[1], 16)] = parts[2].split("\t")[0].strip()
                except ValueError:
                    continue
    return out


def all_strings() -> dict[int, str]:
    return {**system_strings(), **card_strings()}


def _lua_rows(scripts_dir: Path, consts: dict) -> dict[int, list[int]]:
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
    return rows


def lua_desc_table(scripts_dir: Path = OFFICIAL_DIR,
                   extra_dirs: list[Path] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(keys int64 [N], feats int64 [N, len(LUA_COLS)]) over every official card script.

    `extra_dirs` (default: every `YGO_EXTRA_CARDS` `<dir>/script`) adds those scripts too; a script
    name that already exists in `scripts_dir` is skipped (same rule as
    `engine.resources.index_scripts`). With no extra dirs the result is byte-identical to before.
    """
    consts = load_constants()
    rows = _lua_rows(scripts_dir, consts)
    official = {p.name for p in scripts_dir.glob("c*.lua")}
    for d in extra_dirs if extra_dirs is not None else extra_script_dirs():
        if d == scripts_dir:
            continue
        for p in sorted(d.glob("c*.lua")):
            if p.name in official or not p.stem[1:].isdigit():
                continue
            official.add(p.name)
            code = int(p.stem[1:])
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
                    f[5] |= int(len(r.count_limit) > 1)
                f[6] |= (F_COND * r.has_condition) | (F_COST * r.has_cost) | (F_TARGET * r.has_target) | (
                    F_OP * r.has_operation)  # fmt: skip
                f[7] += 1
    keys = np.array(sorted(rows), dtype=np.int64)
    feats = np.array([rows[int(k)] for k in keys], dtype=np.int64).reshape(len(keys), len(LUA_COLS))
    return keys, feats


def merge_lua_desc(base: Path, out: Path, extra_dirs: list[Path] | None = None,
                   in_place: bool = False) -> Path:
    """Merge `base` lua_desc.npz with the extra dirs' scripts -> `out` (never touches `base`).

    Only the extra scripts are parsed (official parsing is skipped: `base` already holds it).
    Base keys keep their rows; keys only in the extras are appended (resorted: the table stays
    sorted for `DescTables._lookup`). A script name that also exists in the official dir is
    skipped (same rule as `engine.resources.index_scripts`). Refuses to overwrite
    `data/cards/lua_desc.npz` unless `in_place` is passed.
    """
    import time as _time

    default = (EMB_DIR.parent / "lua_desc.npz").resolve()
    if Path(out).resolve() == default and not in_place:
        raise SystemExit("ALERT effect_strings: refusing to overwrite data/cards/lua_desc.npz "
                         "(pass --in-place or --out)")
    t0 = _time.time()
    y = np.load(base)
    old_keys, old_feats = np.array(y["keys"]), np.array(y["feats"])
    consts = load_constants()
    official = {p.name for p in OFFICIAL_DIR.glob("c*.lua")}
    have = {int(k) for k in old_keys}
    rows: dict[int, list[int]] = {}
    for d in extra_dirs if extra_dirs is not None else extra_script_dirs():
        if d == OFFICIAL_DIR:
            continue
        for p in sorted(d.glob("c*.lua")):
            if p.name in official or not p.stem[1:].isdigit():
                continue
            official.add(p.name)
            code = int(p.stem[1:])
            for r in parse_script(p.read_text(encoding="utf-8", errors="replace"), code, consts):
                if r.description is None or desc_key(code, r.description) in have:
                    continue
                k = desc_key(code, r.description)
                f = rows.setdefault(k, [0] * len(LUA_COLS))
                f[0] |= r.type
                f[1] |= r.category
                f[2] |= r.range
                f[3] |= r.property
                if r.count_limit:
                    f[4] = max(f[4], int(r.count_limit[0]))
                    f[5] |= int(len(r.count_limit) > 1)
                f[6] |= (F_COND * r.has_condition) | (F_COST * r.has_cost) | (F_TARGET * r.has_target) | (
                    F_OP * r.has_operation)  # fmt: skip
                f[7] += 1
    new_keys = np.array(sorted(rows), dtype=np.int64)
    new_feats = np.array([rows[int(k)] for k in new_keys], dtype=np.int64).reshape(len(new_keys),
                                                                                   len(LUA_COLS)) if rows else \
        np.zeros((0, len(LUA_COLS)), dtype=np.int64)
    all_keys = np.concatenate([old_keys, new_keys])
    all_feats = np.concatenate([old_feats, new_feats])
    order = np.argsort(all_keys)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=all_keys[order], feats=all_feats[order], cols=np.array(LUA_COLS))
    print(f"merged {base} + {len(new_keys)} extra descriptions -> {out} ({_time.time() - t0:.0f}s)",
          flush=True)
    return out


def build_lua(out: Path | None = None, extra_dirs: list[Path] | None = None,
              base: Path | None = None, in_place: bool = False) -> Path:
    """Build the lua_desc table.

    Full build (default): parse the official scripts dir (plus `YGO_EXTRA_CARDS` dirs when set)
    -> `data/cards/lua_desc.npz` (or `out`).
    Incremental (`base` given): parse only the extra dirs and merge into `base` -> `out` (default:
    next to the original, `<stem>-custom.npz`). Never overwrites `data/cards/lua_desc.npz` unless
    `--in-place` is passed.
    """
    if base is not None:
        default_out = Path(base).parent / f"{Path(base).stem}-custom.npz"
        return merge_lua_desc(Path(base), out or default_out, extra_dirs=extra_dirs, in_place=in_place)
    default = (EMB_DIR.parent / "lua_desc.npz").resolve()
    out = Path(out) if out is not None else EMB_DIR.parent / "lua_desc.npz"
    effective = extra_dirs if extra_dirs is not None else extra_script_dirs()
    if out.resolve() == default and effective and not in_place:
        raise SystemExit("ALERT effect_strings: refusing to overwrite data/cards/lua_desc.npz "
                         "with an extras build (pass --in-place or --out)")
    t0 = time.time()
    keys, feats = lua_desc_table(extra_dirs=extra_dirs)
    if len(keys) == 0:
        raise SystemExit("ALERT effect_strings: 0 Lua descriptions parsed")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=keys, feats=feats, cols=np.array(LUA_COLS))
    print(f"wrote {out}: {len(keys)} descriptions, {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="Build or incrementally extend the lua_desc table.")
    _ap.add_argument("--out", type=Path, default=None, help="output npz (incremental default: <base>-custom.npz)")
    _ap.add_argument("--extra", type=Path, action="append", default=None,
                     help="<dir>/script to parse (default: YGO_EXTRA_CARDS dirs); repeatable")
    _ap.add_argument("--base", type=Path, default=None, help="incremental mode: merge into this npz")
    _ap.add_argument("--in-place", action="store_true", help="allow overwriting data/cards/lua_desc.npz")
    _a = _ap.parse_args()
    build_lua(out=_a.out, extra_dirs=_a.extra, base=_a.base, in_place=_a.in_place)


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


def missing_string_keys(base: str | Path) -> tuple[list[int], dict[int, str]]:
    """String keys in the current card DBs (extras included) + system strings absent from `base`.

    No model load: for `--only-missing --strings --dry-run`. Returns (missing keys, key -> text).
    """
    z = np.load(base)
    have = {int(k) for k in z["keys"]}
    strings = all_strings()
    missing = sorted(k for k in strings if k not in have)
    return missing, {k: strings[k] for k in missing}


def append_string_embeddings(base: str | Path, out: str | Path, dim: int = 512,
                             batch: int = 128, encode=None) -> Path:
    """Embed the strings missing from `base` and write the merged npz to `out` (same schema).

    Base vecs are kept verbatim (appended to, never reordered), so every base key maps to a
    byte-identical vector; keys stay sorted for `DescTables._lookup`. When nothing is missing,
    `out` is still written (a copy of `base`). `encode` maps unique new texts to fp16 vectors
    (default: the Qwen model; tests pass a fake).
    """
    import time as _time

    t0 = _time.time()
    z = np.load(base)
    if int(z["dim"]) != dim:
        raise SystemExit(f"ALERT effect_strings: base dim {int(z['dim'])} != --dim {dim}")
    base_keys = [int(k) for k in z["keys"]]
    base_rows = [int(r) for r in z["rows"]]
    base_vecs = np.array(z["vecs"])
    missing, _texts = missing_string_keys(base)
    strings = all_strings()
    text2idx: dict[str, int] = {}
    for k, r in zip(base_keys, base_rows):
        if (t := strings.get(k)) is not None:
            text2idx.setdefault(t, r)
    new_texts: list[str] = []
    for k in missing:
        if strings[k] not in text2idx:
            text2idx[strings[k]] = len(base_vecs) + len(new_texts)
            new_texts.append(strings[k])
    new_vecs = None
    if new_texts:
        if encode is None:
            os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "hf"))
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL, device=pick_device(), truncate_dim=dim)

            def encode(texts: list[str]) -> np.ndarray:
                return model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float16)

        new_vecs = np.asarray(encode(new_texts), dtype=np.float16)
        if new_vecs.shape != (len(new_texts), dim) or not np.isfinite(new_vecs).all() or \
                (np.abs(new_vecs).sum(1) == 0).any():
            raise SystemExit("ALERT effect_strings: non-finite, zero or misshaped string vectors")
    all_keys = sorted(set(base_keys) | set(missing))
    have_rows = dict(zip(base_keys, base_rows))
    rows = [have_rows[k] if k in have_rows else text2idx[strings[k]] for k in all_keys]
    vecs = np.concatenate([base_vecs, new_vecs]) if new_vecs is not None else base_vecs
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=np.array(all_keys, dtype=np.int64),
                        rows=np.array(rows, dtype=np.int32), vecs=vecs, model=MODEL, dim=dim)
    print(f"merged {base} + {len(missing)} strings -> {out} ({_time.time() - t0:.0f}s)", flush=True)
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
