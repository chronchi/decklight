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

# Card-knowledge memory (obs v2, docs/obs_spec.md): stored in shards when the env produced it.
KNOWN_KEYS = ("card_known", "known_code")

# Gumbel-style soft targets (train_bc --soft-targets): per-sample search candidates and their
# selection scores. Fixed width K (current search data: k <= 4 candidates); -1-padded indices.
SOFT_K = 8

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
    from decklight.engine.obs import HISTORY_COLS, SALIENT_SKIP

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
    from decklight.engine.duel import Engine

    _ENGINE = Engine()


def base_line(line_id: str) -> str:
    return line_id.split("#")[0].split("+")[0]


def load_exclusions(specs: list[str]) -> list[tuple[str, set[str], set[str]]]:
    """Retrain stage D exclusions. Each spec is `<globs>@<file>`: records in files matching the globs are dropped
    when their line_id is listed in the file. `<globs>` is comma-separated; a `!` glob excludes (e.g.
    `data/demos/*,!*/strict-d/*`: every record file except the strict re-verified ones). A `.jsonl` file lists exact record ids (`{"line": id}` rows, the
    hand-trap drops of scripts/audit/handtrap_class.py); a `.json` file maps source -> {"lines": [base line ids]}
    (scripts/audit/strict_junk.py: lines replaced by their strict re-verification), matched on the base id."""
    import json

    out = []
    for spec in specs:
        glob_, _, f = spec.rpartition("@")
        if not glob_ or not f:
            raise ValueError(f"bad exclusion spec {spec!r} (want <glob>@<file>)")
        text = Path(f).read_text()
        if f.endswith(".jsonl"):
            ids = {json.loads(x)["line"] for x in text.splitlines() if x.strip()}
            out.append((glob_, ids, set()))
        else:
            out.append((glob_, set(), {b for v in json.loads(text).values() for b in v["lines"]}))
        if not out[-1][1] and not out[-1][2]:
            raise ValueError(f"exclusion file {f} lists no lines")
    return out


def _skip_for(src_file: str, excl: list[tuple[str, set[str], set[str]]]):
    import fnmatch

    def hit(globs: str) -> bool:
        pos = [g for g in globs.split(",") if not g.startswith("!")]
        neg = [g[1:] for g in globs.split(",") if g.startswith("!")]
        return any(fnmatch.fnmatch(src_file, g) for g in pos) and not any(fnmatch.fnmatch(src_file, g) for g in neg)

    rules = [(ids, bases) for g, ids, bases in excl if hit(g)]
    if not rules:
        return None
    return lambda rec: any(rec.get("line_id") in ids or base_line(rec.get("line_id") or "") in bases
                           for ids, bases in rules)  # fmt: skip


def _shard_one(job: tuple[str, str, str]) -> tuple[str, int]:
    from decklight.demos.dataset import iter_samples

    src_file, out_file, src_tag, *rest = job
    strict = bool(rest[0]) if rest else False
    excl = rest[1] if len(rest) > 1 else []
    cols: dict[str, list] = {k: [] for k in OBS_KEYS}
    known_cols: dict[str, list] = {k: [] for k in KNOWN_KEYS}
    has_known: bool | None = None  # all samples of one file either carry the keys or not
    action, weight, line, deck, kind, equiv = [], [], [], [], [], []
    soft_idx_rows: list = []  # per-sample search candidates (option indices, -1 padded to SOFT_K)
    soft_q_rows: list = []  # per-sample selection scores aligned with soft_idx (0 padded)
    hlog_rows: list = []  # per-record logs in first-seen order (shared refs from iter_samples meta)
    hlog_start, hlog_end, seath, turnh = [], [], [], []
    log_off: dict[int, int] = {}
    alive: list = []  # keeps every keyed log referenced: id() of a freed array can be reused by a later record's log
    running = 0
    has_log = True
    for obs, a, w, meta in iter_samples(_ENGINE, Path(src_file), strict=strict, skip=_skip_for(src_file, excl)):
        for k in OBS_KEYS:
            cols[k].append(obs[k])
        hk = all(k in obs for k in KNOWN_KEYS)
        if has_known is None:
            has_known = hk
        elif hk != has_known:
            raise ValueError(f"mixed obs with/without {KNOWN_KEYS} in one file: {src_file}")
        if hk:
            for k in KNOWN_KEYS:
                known_cols[k].append(obs[k])
        equiv.append(equivalent_mask(obs["actions"], a, obs.get("action_mask")))
        alt = meta.get("alt")
        if alt:
            mask = obs.get("action_mask")
            eq = equiv[-1]
            for b in alt:
                b = int(b)
                if 0 <= b < len(eq) and (mask is None or bool(mask[b])):
                    eq[b] = True
        action.append(a)
        weight.append(w)
        si = np.full(SOFT_K, -1, dtype=np.int16)
        sq = np.zeros(SOFT_K, dtype=np.float16)
        soft = meta.get("soft")
        if soft is not None:
            cands, scores = soft
            if len(cands) > SOFT_K:
                raise ValueError(f"record has {len(cands)} search candidates > SOFT_K={SOFT_K}: {src_file}")
            si[: len(cands)] = np.asarray(cands, dtype=np.int16)
            sq[: len(scores)] = np.asarray(scores, dtype=np.float16)
        soft_idx_rows.append(si)
        soft_q_rows.append(sq)
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
    if has_known:
        extra.update({k: np.stack(v).astype({"card_known": np.uint8, "known_code": np.int32}[k])
                      for k, v in known_cols.items()})
    if has_log:
        extra.update({
            "hlog": np.concatenate(hlog_rows) if hlog_rows else np.zeros((0, 14), np.int32),
            "hlog_start": np.array(hlog_start, dtype=np.int64),
            "hlog_end": np.array(hlog_end, dtype=np.int64),
            "me": np.array(seath, dtype=np.int32),
            "turn": np.array(turnh, dtype=np.int32),
        })
    np.savez_compressed(
        out_file,
        **{k: np.stack(v) for k, v in cols.items()},
        action=np.array(action, dtype=np.int64),
        equiv=np.stack(equiv),
        soft_idx=np.stack(soft_idx_rows),
        soft_q=np.stack(soft_q_rows),
        weight=np.array(weight, dtype=np.float32),
        line=np.array(line),
        deck=np.array(deck),
        kind=np.array(kind),
        src=np.array([src_tag] * len(action)),
        **extra,
    )
    return out_file, len(action)


def build_shards(sources: dict[str, list[Path]], out_dir: Path, jobs: int, strict: bool = False,
                 exclude: list[str] | None = None) -> int:
    """sources: tag -> record files. Writes one shard per file into out_dir; returns the sample count.
    `strict`: inferred labels weigh 0 except choices among equals (demos/dataset.py strict_inferred_weight).
    `exclude`: `<path glob>@<file>` specs (load_exclusions); globs match the record file path as given."""
    excl = load_exclusions(exclude or [])
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
            work.append((str(f), str(out_dir / f"{tag}__{flat}.npz"), tag, strict, excl))
    total = 0
    per_tag: dict[str, list[int]] = {}  # tag -> [files, samples, files with 0 samples]
    with mp.get_context("fork").Pool(max(1, jobs), initializer=_init) as pool:
        for out, n in pool.imap_unordered(_shard_one, work):
            total += n
            print(f"{Path(out).name}: {n}", flush=True)
            st = per_tag.setdefault(Path(out).name.split("__", 1)[0], [0, 0, 0])
            st[0], st[1], st[2] = st[0] + 1, st[1] + n, st[2] + (n == 0)
    for tag, (nf, ns, nz) in sorted(per_tag.items()):
        print(f"source {tag}: {nf} files, {ns} samples, {nz} files with 0 samples", flush=True)
        # fail loud: in a good build every source has <= 5% empty files (games); 5988 teacher files silently gave
        # 0 samples on 2026-10-06 (a kind rule matched their file names)
        if ns == 0 or nz > 0.25 * nf:
            print(f"ALERT shards: source {tag} has {nz} of {nf} files with 0 samples ({ns} samples)", flush=True)
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
    for p in parts:  # backward compat: shards written before search soft targets existed
        if "soft_idx" not in p or "soft_q" not in p:
            n = len(p["action"])
            p.setdefault("soft_idx", np.full((n, SOFT_K), -1, dtype=np.int16))
            p.setdefault("soft_q", np.zeros((n, SOFT_K), dtype=np.float16))
    if any("hlog" in p for p in parts) and not all("hlog" in p for p in parts):
        raise ValueError("mixed shards with and without hlog in one directory: rebuild them together")
    if any("card_known" in p for p in parts) and not all(
        all(k in p for k in KNOWN_KEYS) for p in parts
    ):
        raise ValueError(f"mixed shards with and without {KNOWN_KEYS} in one directory: rebuild them together")
    if "hlog" in parts[0]:  # offset each file's record-relative rows into the concatenated log
        off, starts, ends = 0, [], []
        for p in parts:
            starts.append(p["hlog_start"] + off)
            ends.append(p["hlog_end"] + off)
            off += len(p["hlog"])
        out = {"hlog_start": np.concatenate(starts), "hlog_end": np.concatenate(ends)}
        return {**{k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in out}, **out}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


_ROOT = Path(__file__).resolve().parents[3]


def eval_only_decks(root: Path = _ROOT) -> tuple[set[str], set[str]]:
    """(holdout, newcards): pool decks never trained on (`decks/pool/HOLDOUT` + kewl-tune) and the unseen-card decks
    (`decks/newcards`, never in any generator). Retrain plan v2: training must exclude both, enforced, not by flag."""
    hold = set((root / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    new = {p.stem for p in (root / "decks/newcards").glob("*.ydk")}
    return hold, new


def check_eval_decks(decks: np.ndarray, holdout_arg: tuple[str, ...] | list[str], root: Path = _ROOT) -> list[str]:
    """Problems that must stop a training run: a newcards deck anywhere in the data, or a holdout deck present in the
    data but missing from `holdout_arg` (it would land in train). Empty list = OK."""
    hold, new = eval_only_decks(root)
    present = set(np.unique(decks).tolist())
    out = [f"newcards deck in training data: {d}" for d in sorted(present & new)]
    out += [f"holdout deck not excluded (--holdout): {d}" for d in sorted((present & hold) - set(holdout_arg))]
    return out


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
