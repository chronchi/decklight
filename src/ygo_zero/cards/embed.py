"""Card text embeddings with an open-weight model (PLAN Phase 2, layer 2, first version).

Every card in BabelCDB gets
- `card`: one vector for the whole card ("name. type line. text"), and
- `effects`: up to MAX_EFFECTS vectors, one per effect sentence of the text (PSCT effects end with a period;
  "●" bullets stay with their effect), padded with zeros, plus `n_effects`.
Cards with the same text (alternate artworks) share one computation. Vectors are L2-normalised float16, truncated to
`dim` (Qwen3-Embedding is trained Matryoshka-style, so prefixes are valid embeddings).

Cache: `data/cards/emb/<tag>.npz` (codes, card, effects, n_effects). The model lives in `HF_HOME=data/cache/hf`.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from ygo_zero.engine.resources import default_cdbs

ROOT = Path(__file__).resolve().parents[3]
EMB_DIR = ROOT / "data" / "cards" / "emb"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_EFFECTS = 8

_PEND = re.compile(r"\[\s*Pendulum Effect\s*\]|\[\s*Monster Effect\s*\]|-{5,}")


def card_texts(cdbs: list[Path] | None = None) -> dict[int, tuple[str, str]]:
    """code -> (name, desc) over all BabelCDB files (later files override earlier ones)."""
    out: dict[int, tuple[str, str]] = {}
    for path in cdbs if cdbs is not None else default_cdbs():
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for code, name, desc in con.execute("SELECT id, name, desc FROM texts"):
                out[int(code)] = (name or "", (desc or "").replace("\r\n", "\n"))
        finally:
            con.close()
    return out


def split_effects(desc: str) -> list[str]:
    """Effect sentences of a card text: pendulum/monster sections, then sentences; '●' bullet lines join the
    sentence before them. Material lines (no period, first line of Extra Deck monsters) stay as their own segment."""
    parts: list[str] = []
    for section in _PEND.split(desc):
        section = section.strip()
        if not section:
            continue
        lines = [ln.strip() for ln in section.split("\n") if ln.strip()]
        buf: list[str] = []
        for ln in lines:
            if ln.startswith("●") and buf:
                buf[-1] += " " + ln
                continue
            # sentences: split after a period followed by a space and an uppercase letter / quote
            for s in re.split(r"(?<=\.)\s+(?=[A-Z\"(])", ln):
                if s:
                    buf.append(s)
        parts.extend(buf)
    return [p for p in parts if len(p) > 3]


def pick_device() -> str:
    """'cuda' when a GPU is usable (a tiny op succeeds: old GPUs may lack kernels in current wheels), else 'cpu'."""
    import torch

    if torch.cuda.is_available():
        try:
            (torch.ones(2, device="cuda") * 2).sum().item()
            return "cuda"
        except Exception as e:  # noqa: BLE001 - unsupported arch etc.
            print(f"cuda unusable ({e}); using cpu", flush=True)
    return "cpu"


def build(dim: int = 512, batch: int = 32, limit: int | None = None, codes: list[int] | None = None,
          tag: str | None = None, log_every: int = 50, out_dir: Path | None = None) -> Path:  # fmt: skip
    """Embed every card (or `codes`) and write `<out_dir or data/cards/emb>/<tag>.npz`."""
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "hf"))
    from sentence_transformers import SentenceTransformer

    texts = card_texts()
    keys = sorted(codes if codes is not None else texts)
    keys = [k for k in keys if k in texts][:limit]
    device = pick_device()
    print(f"device {device}", flush=True)
    model = SentenceTransformer(MODEL, device=device, truncate_dim=dim)
    uniq: dict[str, int] = {}  # text -> row in the unique-text table
    card_rows: list[int] = []
    eff_rows: list[list[int]] = []
    for code in keys:
        name, desc = texts[code]
        whole = f"{name}. {desc}"
        card_rows.append(uniq.setdefault(whole, len(uniq)))
        eff_rows.append([uniq.setdefault(e, len(uniq)) for e in split_effects(desc)[:MAX_EFFECTS]])
    table = list(uniq)
    vecs = np.zeros((len(table), dim), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(table), batch):
        chunk = table[i : i + batch]
        v = model.encode(chunk, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True)
        vecs[i : i + len(chunk)] = v.astype(np.float16)
        if (i // batch) % log_every == 0:
            done = i + len(chunk)
            rate = done / max(time.time() - t0, 1e-6)
            print(
                f"{done}/{len(table)} texts, {rate:.1f}/s, eta {(len(table) - done) / rate / 60:.0f} min",
                flush=True,
            )
    n = len(keys)
    card = vecs[card_rows]
    effects = np.zeros((n, MAX_EFFECTS, dim), dtype=np.float16)
    n_eff = np.zeros(n, dtype=np.int8)
    for j, rows in enumerate(eff_rows):
        if rows:
            effects[j, : len(rows)] = vecs[rows]
            n_eff[j] = len(rows)
    out_dir = out_dir or EMB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag or f'qwen3-0.6b-d{dim}'}.npz"
    np.savez_compressed(out, codes=np.array(keys, dtype=np.int64), card=card, effects=effects, n_effects=n_eff,
                        model=MODEL, dim=dim)  # fmt: skip
    print(f"wrote {out}: {n} cards, {len(table)} unique texts, {time.time() - t0:.0f}s", flush=True)
    return out


def missing_card_codes(base: str | Path) -> tuple[list[int], dict[int, str]]:
    """Codes in the current card DBs (`YGO_EXTRA_CARDS` included) absent from `base` npz, with names.

    No model load: for `--only-missing --dry-run`.
    """
    z = np.load(base)
    have = {int(c) for c in z["codes"]}
    texts = card_texts()
    missing = sorted(c for c in texts if c not in have)
    return missing, {c: texts[c][0] for c in missing}


def append_card_embeddings(base: str | Path, out: str | Path, dim: int = 512,
                            batch: int = 32, encode=None) -> Path:
    """Embed the codes missing from `base` and write the merged npz to `out` (same schema as `build`).

    Base rows are kept verbatim (same order, same values); new codes are appended sorted. When
    nothing is missing, `out` is still written (a copy of `base`). `encode` maps unique new texts
    to fp16 vectors (default: the Qwen model; tests pass a fake so no model ever loads on laptop).
    """
    import time as _time

    t0 = _time.time()
    z = np.load(base)
    if int(z["dim"]) != dim:
        raise SystemExit(f"ALERT embed: base dim {int(z['dim'])} != --dim {dim}")
    base_codes = [int(c) for c in z["codes"]]
    missing, _names = missing_card_codes(base)
    texts = card_texts()
    uniq: dict[str, int] = {}
    card_rows: list[int] = []
    eff_rows: list[list[int]] = []
    for code in missing:
        name, desc = texts[code]
        card_rows.append(uniq.setdefault(f"{name}. {desc}", len(uniq)))
        eff_rows.append([uniq.setdefault(e, len(uniq)) for e in split_effects(desc)[:MAX_EFFECTS]])
    vecs = None
    if uniq:
        if encode is None:
            os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "hf"))
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL, device=pick_device(), truncate_dim=dim)

            def encode(texts: list[str]) -> np.ndarray:
                return model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float16)

        vecs = np.asarray(encode(list(uniq)), dtype=np.float16)
        if vecs.shape != (len(uniq), dim) or not np.isfinite(vecs).all() or \
                (np.abs(vecs).sum(1) == 0).any():
            raise SystemExit("ALERT embed: non-finite, zero or misshaped card vectors")
    n_new = len(missing)
    card = np.concatenate([np.array(z["card"]),
                           vecs[card_rows] if n_new else np.zeros((0, dim), np.float16)])
    eff = np.concatenate([np.array(z["effects"]),
                          np.zeros((n_new, MAX_EFFECTS, dim), np.float16)]) if n_new else \
        np.array(z["effects"])
    n_eff = np.array(z["n_effects"])
    if n_new:
        assert vecs is not None
        n_new_eff = np.zeros(n_new, np.int8)
        for j, rows in enumerate(eff_rows):
            if rows:
                eff[len(n_eff) + j, : len(rows)] = vecs[rows]
                n_new_eff[j] = len(rows)
        n_eff = np.concatenate([n_eff, n_new_eff])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, codes=np.array(base_codes + missing, dtype=np.int64), card=card,
                        effects=eff, n_effects=n_eff, model=MODEL, dim=dim)
    print(f"merged {base} + {n_new} cards -> {out} ({_time.time() - t0:.0f}s)", flush=True)
    return out


class CardEmbeddings:
    """Lookup of the cached vectors by card code (unknown / hidden codes -> zeros, flagged).

    `path` or `$YGO_EMB` (a merged npz covering custom cards) or the newest stock build.
    """

    def __init__(self, path: Path | None = None) -> None:
        path = path or os.environ.get("YGO_EMB") or max(EMB_DIR.glob("qwen3-0.6b-d*.npz"))
        z = np.load(path)
        self.codes = z["codes"]
        self.card = z["card"]
        self.effects = z["effects"]
        self.n_effects = z["n_effects"]
        self.dim = int(z["dim"])
        self.index = {int(c): i for i, c in enumerate(self.codes)}

    def rows(self, codes: np.ndarray) -> np.ndarray:
        """Row index per code (−1 if unknown); feed `row + 1` to an embedding table whose row 0 is 'unknown'."""
        return np.array([self.index.get(int(c), -1) for c in np.ravel(codes)], dtype=np.int64).reshape(
            np.shape(codes)
        )
