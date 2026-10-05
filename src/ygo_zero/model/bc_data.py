"""BC training data: replay demo records into observation shards, load them, split by line.

A shard is one `.npz` per record file with the stacked observation arrays (docs/obs_spec.md) plus
`action` (chosen index), `weight` (stated 1.0 / inferred 0.5), `line` (base line id: the variant suffix `#h<k>` and deck
tags stripped, so all hand variants of one line land in the same split), `deck` and `src`. Shards are temporary
training artifacts (AGENTS.md storage rules): rebuild them from the records, never commit them.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import zipfile
from pathlib import Path

import numpy as np

OBS_KEYS = (
    "cards",
    "card_mask",
    "global",
    "chain",
    "chain_mask",
    "actions",
    "action_mask",
    "history",
    "history_mask",
)

# Option-row columns (docs/action_spec.md): 0 kind, 1 card row, 2 code, 3 effect_code,
# 4 effect_idx, 5 arg, 6 msg, 7 location, 8 controller, 9 step.
PLACE_KIND = 19
_SINGLETON_KINDS = frozenset({11, 17, 18})  # PASS, FINISH, CANCEL: never equivalent to anything else


def equivalent_mask(
    actions: np.ndarray, action: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Bool [64] mask of options equivalent to the chosen option `action`.

    Option j is equivalent to the chosen option a when their action rows are equal on every
    column EXCEPT the ones that only identify a position among identical things: the card-row
    index (col 1) and, for PLACE-kind options (kind 19), the zone/sequence argument (col 5).
    Concretely, j != a is equivalent when: same kind (col 0), same code (col 2), same effect
    code/index (cols 3-4), same msg (col 6), same location (col 7), same controller (col 8),
    same step (col 9), and -- for non-PLACE kinds -- same arg (col 5). The chosen option itself
    is always included. Options of kind PASS (11), FINISH (17) or CANCEL (18) are never marked
    equivalent to anything else (a one-hot of `action` is returned when the choice itself is
    one of those). Illegal slots (per `mask`, when given) are never marked equivalent.
    """
    n = actions.shape[0]
    equiv = np.zeros(n, dtype=bool)
    equiv[action] = True
    a = actions[action]
    kind_a = int(a[0])
    if kind_a in _SINGLETON_KINDS:
        return equiv
    same = (
        (actions[:, 0] == kind_a)
        & (actions[:, 2] == a[2])
        & (actions[:, 3] == a[3])
        & (actions[:, 4] == a[4])
        & (actions[:, 6] == a[6])
        & (actions[:, 7] == a[7])
        & (actions[:, 8] == a[8])
        & (actions[:, 9] == a[9])
    )
    if kind_a != PLACE_KIND:
        same &= actions[:, 5] == a[5]
    for k in _SINGLETON_KINDS:
        same &= actions[:, 0] != k
    if mask is not None:
        same &= np.asarray(mask, dtype=bool)
    same[action] = True  # always include the choice itself, even if masked out
    return same


_ENGINE = None


def history_window(hlog, start, end, me, turn, H, salient=True):
    """Rebuild history windows from a compact log (the env transform, vectorised within each sample).

    hlog [S, 14]: concatenated per-record logs (obs.history_log format: stored option row in cols 0-9,
    actor / turn / phase / public in cols 10-13). start/end/me/turn [B]: each sample's record slice
    [start, end) and viewer (acting seat, duel turn). Returns (hist [B, H, 13] int32, mask [B, H] bool),
    most-recent-first. `salient` skips PASS / PLACE entries (obs v2 `hist_long`); salient=False with H=32
    reproduces the v1 `history` window exactly.
    """
    from ygo_zero.engine.obs import HISTORY_COLS, SALIENT_SKIP

    hlog = np.asarray(hlog, dtype=np.int32)
    start = np.asarray(start, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    me = np.asarray(me, dtype=np.int64)
    turn = np.asarray(turn, dtype=np.int64)
    skip = np.array([int(k) for k in SALIENT_SKIP])
    n = len(start)
    hist = np.zeros((n, H, HISTORY_COLS), np.int32)
    mask = np.zeros((n, H), bool)
    for i in range(n):
        rows = hlog[start[i] : end[i]]
        if salient and len(rows):
            rows = rows[~np.isin(rows[:, 0], skip)]
        take = rows[max(0, len(rows) - H) :][::-1]  # most recent first
        k = len(take)
        if not k:
            continue
        base = take[:, :10].copy()
        actor, public = take[:, 10], take[:, 13]
        flip = base[:, 8] >= 0
        base[flip, 8] ^= actor[flip] ^ me[i]
        hide = (actor != me[i]) & (public == 0)
        base[np.ix_(hide, [2, 3, 4])] = 0
        hist[i, :k, :10] = base
        hist[i, :k, 10] = actor ^ me[i]
        hist[i, :k, 11] = turn[i] - take[:, 11]
        hist[i, :k, 12] = take[:, 12]
        mask[i, :k] = True
    return hist, mask


def _init() -> None:
    global _ENGINE
    from ygo_zero.engine.duel import Engine

    _ENGINE = Engine()


def base_line(line_id: str) -> str:
    return line_id.split("#")[0].split("+")[0]


def _shard_one(job: tuple[str, str, str]) -> tuple[str, int]:
    from ygo_zero.demos.dataset import iter_samples

    src_file, out_file, src_tag = job
    cols: dict[str, list] = {k: [] for k in OBS_KEYS}
    action, weight, line, deck, kind, equiv = [], [], [], [], [], []
    hlog_rows: list = []  # per-record logs in first-seen order (shared refs from iter_samples meta)
    hlog_start, hlog_end, seath, turnh = [], [], [], []
    log_off: dict[int, int] = {}
    alive: list = []  # keeps every keyed log referenced: id() of a freed array can be reused by a later record's log
    running = 0
    has_log = True
    for obs, a, w, meta in iter_samples(_ENGINE, Path(src_file)):
        for k in OBS_KEYS:
            cols[k].append(obs[k])
        equiv.append(equivalent_mask(obs["actions"], a, obs.get("action_mask")))
        action.append(a)
        weight.append(w)
        line.append(base_line(meta["line_id"]))
        deck.append(meta["deck"])
        kind.append(meta.get("kind", "unknown"))
        log = meta.get("hlog")
        if log is None or "hlog_end" not in meta:
            has_log = False
        else:
            key = id(log)
            if key not in log_off:
                log_off[key] = running
                alive.append(log)
                hlog_rows.append(np.asarray(log, dtype=np.int32))
                running += len(log)
            hlog_start.append(log_off[key])
            hlog_end.append(log_off[key] + int(meta["hlog_end"]))
            seath.append(int(meta["me"]))
            turnh.append(int(meta["turn"]))
    if not action:
        return out_file, 0
    extra = {}
    if has_log:
        extra = {
            "hlog": np.concatenate(hlog_rows) if hlog_rows else np.zeros((0, 14), np.int32),
            "hlog_start": np.array(hlog_start, dtype=np.int64),
            "hlog_end": np.array(hlog_end, dtype=np.int64),
            "me": np.array(seath, dtype=np.int32),
            "turn": np.array(turnh, dtype=np.int32),
        }
    np.savez_compressed(
        out_file,
        **{k: np.stack(v) for k, v in cols.items()},
        action=np.array(action, dtype=np.int64),
        equiv=np.stack(equiv),
        weight=np.array(weight, dtype=np.float32),
        line=np.array(line),
        deck=np.array(deck),
        kind=np.array(kind),
        src=np.array([src_tag] * len(action)),
        **extra,
    )
    return out_file, len(action)


def build_shards(sources: dict[str, list[Path]], out_dir: Path, jobs: int) -> int:
    """sources: tag -> record files. Writes one shard per file into out_dir; returns the sample count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = []
    for tag, files in sources.items():
        for f in files:
            # the full path makes names unique: planned-s1/verified/x and planned-s2/verified/x collided before
            # (later files silently overwrote earlier ones; fixed 2026-10-03)
            rel = Path(f).resolve()
            try:
                rel = rel.relative_to(Path(__file__).resolve().parents[3] / "data" / "demos")
            except ValueError:
                pass
            flat = "~".join(rel.with_suffix("").parts[-6:])
            work.append((str(f), str(out_dir / f"{tag}__{flat}.npz"), tag))
    total = 0
    with mp.get_context("fork").Pool(max(1, jobs), initializer=_init) as pool:
        for out, n in pool.imap_unordered(_shard_one, work):
            total += n
            print(f"{Path(out).name}: {n}", flush=True)
    return total


def _npz_headers(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Shape and dtype of every array in an .npz without decompressing the data."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                major, _ = np.lib.format.read_magic(f)
                read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
                shape, _, dtype = read(f)
            out[name.removesuffix(".npy")] = (shape, dtype)
    return out


def load_shards(shard_dir: Path) -> dict[str, np.ndarray]:
    """All shards of a directory, concatenated. Current shards (kind, equiv, hlog everywhere) are read one file at a
    time into preallocated arrays: peak RAM ~ the result, not twice it (15 GB boxes OOMed on 619k samples, 2026-10-05).
    Older or mixed directories take the simple path below."""
    files = sorted(shard_dir.glob("*.npz"))
    heads = [_npz_headers(f) for f in files]
    keys = set(heads[0]) if heads else set()
    if heads and all(set(h) == keys for h in heads) and {"kind", "equiv", "hlog"} <= keys:
        n = {k: sum(h[k][0][0] for h in heads) for k in keys}
        out = {k: np.empty((n[k], *heads[0][k][0][1:]), np.result_type(*(h[k][1] for h in heads))) for k in keys}
        pos = dict.fromkeys(keys, 0)
        hlog_off = 0
        for f in files:
            with np.load(f) as z:
                for k in keys:
                    a = z[k]
                    if k in ("hlog_start", "hlog_end"):
                        a = a + hlog_off
                    out[k][pos[k] : pos[k] + len(a)] = a
                    pos[k] += len(a)
                hlog_off += len(z["hlog"])
        return out
    parts = [dict(np.load(p)) for p in files]
    for p in parts:  # backward compat: shards written before kind tags existed
        if "kind" not in p:
            p["kind"] = np.array(["unknown"] * len(p["action"]))
    for p in parts:  # backward compat: shards written before equiv masks existed
        if "equiv" not in p and "actions" in p:
            n = len(p["action"])
            eq = np.zeros((n, p["actions"].shape[1]), dtype=bool)
            eq[np.arange(n), p["action"]] = True
            p["equiv"] = eq
    if any("hlog" in p for p in parts) and not all("hlog" in p for p in parts):
        raise ValueError("mixed shards with and without hlog in one directory: rebuild them together")
    if "hlog" in parts[0]:  # offset each file's record-relative rows into the concatenated log
        off, starts, ends = 0, [], []
        for p in parts:
            starts.append(p["hlog_start"] + off)
            ends.append(p["hlog_end"] + off)
            off += len(p["hlog"])
        out = {"hlog_start": np.concatenate(starts), "hlog_end": np.concatenate(ends)}
        return {**{k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in out}, **out}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def split_masks(
    data: dict[str, np.ndarray], val_frac: float = 0.1, holdout_decks: tuple[str, ...] = (),
    kind_val_frac: dict[str, float] | None = None,
) -> dict:
    """Boolean masks train / val / holdout. val = a stable hash of the base line id (all variants together);
    holdout = whole decks never trained on (generalisation to unseen decks). kind_val_frac overrides val_frac for
    samples of the given kinds (e.g. {"gold": 0.2}: more of the scarce hand-played games kept out to evaluate on).
    The hash threshold only grows with the fraction, so a line in val at 0.1 stays in val at 0.2."""
    hold = (
        np.isin(data["deck"], list(holdout_decks)) if holdout_decks else np.zeros(len(data["action"]), bool)
    )
    h = np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 1000 for s in data["line"]])
    thr = np.full(len(h), int(val_frac * 1000))
    for kd, f in (kind_val_frac or {}).items():
        thr[data["kind"] == kd] = int(f * 1000)
    val = (h < thr) & ~hold
    return {"train": ~val & ~hold, "val": val, "holdout": hold}
