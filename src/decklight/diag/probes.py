"""Targets and probes: is the information present in the representation, and does the policy use it?

Targets are computed once per start by playing the reference policy on to the end of the turn (as
`scripts/audit/opp_probe.py`): `complete` (the opponent finishes its script) and `field` (opponent cards at the end
of the turn). Any target column a window set supplies (label-job arm outcomes) is also probed.

Every AUC/R² gets grouped nested CV (by start), a start-level permutation p-value and a bootstrap CI over starts,
on all windows and on the counterfactual pass subset (where the checkpoint's argmax is not an activation).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from decklight.diag.behaviour import ScoredFrame
from decklight.diag.common import (
    base_features,
    bootstrap_ci,
    nested_probe,
    permutation_test,
)
from decklight.diag.windows import Window, _line_turn, group_by_start

BINARY_COLUMNS = ("complete", "won", "completed", "activation")


@dataclass
class ProbeConfig:
    permutations: int = 200
    bootstrap: int = 1000
    seed: int = 0
    min_windows: int = 20
    feature_sets: tuple[str, ...] = ("base", "glob", "opt", "opt+base", "glob+base")


@dataclass
class ProbeResult:
    tables: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    targets: dict = field(default_factory=dict)
    supplied_columns: list[str] = field(default_factory=list)


def compute_targets(engine, windows: list[Window], policy, *, progress=None) -> tuple[dict, dict]:
    """Play `policy` on from every start; returns (group -> {complete, field}, replay info)."""
    from decklight.diag.windows import _audit

    LE = _audit("line_eval")
    targets, diverged, skipped = {}, 0, 0
    groups = group_by_start(windows)
    seen, next_at = 0, 100
    for i, (group, ws) in enumerate(groups.items()):
        seen += len(ws)
        if progress is not None and seen >= next_at:
            progress(f"targets {i + 1}/{len(groups)} starts ({seen}/{len(windows)} windows)")
            next_at += 100
        st = ws[0].start
        try:
            g = LE.play(engine, st, policy, policy, stop_turn=_line_turn(engine, st))
        except Exception:  # noqa: BLE001 - a broken start is counted, never kills the run
            diverged += 1
            continue
        if not st.get("script"):
            skipped += 1
            targets[group] = {"complete": None, "field": int(g["opp_field_end_turn"])}
            continue
        targets[group] = {"complete": int(g["completed"]), "field": int(g["opp_field_end_turn"])}
    return targets, {"groups": len(groups), "diverged": diverged, "no_script": skipped}


def _feature_sets(frames: list[ScoredFrame]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    base, glob, opt, valid = [], [], [], []
    for f in frames:
        j = f.top_activation()
        valid.append(j is not None)
        if j is None:
            continue
        base.append(base_features(f.obs))
        glob.append(f.out.global_feat)
        opt.append(f.out.option_feats[j])
    out = {}
    if base:
        out["base"] = np.asarray(base, float)
        out["glob"] = np.asarray(glob, float)
        out["opt"] = np.asarray(opt, float)
        out["opt+base"] = np.concatenate([out["opt"], out["base"]], 1)
        out["glob+base"] = np.concatenate([out["glob"], out["base"]], 1)
    return out, np.asarray(valid, bool)


def _target_rows(frames: list[ScoredFrame], targets: dict, column: str) -> tuple[list[ScoredFrame], np.ndarray]:
    rows, vals = [], []
    for f in frames:
        v = targets.get(f.window.group, {}).get(column)
        if column in f.window.supplied:
            v = f.window.supplied[column]
        if v is None:
            continue
        rows.append(f)
        vals.append(float(v))
    return rows, np.asarray(vals, float)


def _is_binary(v: np.ndarray) -> bool:
    return bool(np.all(np.isin(v, (0.0, 1.0))))


def _probe_kind(column: str, y: np.ndarray) -> str:
    binary = (not column.endswith("field")) and (
        column in BINARY_COLUMNS or column.endswith(("_won", "_completed")) or _is_binary(y)
    )
    return "auc" if binary else "r2"


def run_probes(
    frames: list[ScoredFrame], targets: dict, cfg: ProbeConfig, *, progress=None
) -> ProbeResult:
    res = ProbeResult(targets=targets)
    supplied = sorted({k for f in frames for k in f.window.supplied})
    res.supplied_columns = supplied
    columns = ["complete", "field", *supplied]
    planned: list[tuple[str, str]] = []
    for column in columns:
        rows, y = _target_rows(frames, targets, column)
        if len(rows) < cfg.min_windows:
            continue
        kind = _probe_kind(column, y)
        if kind == "auc" and len(np.unique(y)) < 2:
            continue
        planned.extend((column, fs) for fs in cfg.feature_sets)
    total = len(planned)
    done_n = 0
    for column in columns:
        rows, y = _target_rows(frames, targets, column)
        if len(rows) < cfg.min_windows:
            continue
        kind = _probe_kind(column, y)
        if kind == "auc" and len(np.unique(y)) < 2:
            message = f"target {column!r} has one class (n={len(y)})"
            if column in ("complete", "field"):
                res.errors.append(message)
            else:  # a supplied (descriptive) column: report, do not fail the whole diagnosis
                res.tables.setdefault(column, {})["@all"] = {"n": len(y), "skipped": True, "reason": message}
            continue
        groups = np.asarray([f.window.group for f in rows])
        feats, valid = _feature_sets(rows)
        entries = {}
        printed: set[str] = set()
        for subset in ("all", "p_passes"):
            if subset == "p_passes":
                keep = np.asarray([not f.argmax_activation for f in rows])
                if keep.sum() < cfg.min_windows or (kind == "auc" and len(np.unique(y[keep])) < 2):
                    entries[f"{column}@{subset}"] = {"n": int(keep.sum()), "skipped": True}
                    continue
            else:
                keep = np.ones(len(rows), bool)
            keep = keep & valid
            n_groups = len(np.unique(groups[keep])) if keep.any() else 0
            if (keep.sum() < cfg.min_windows or n_groups < 2
                    or (kind == "auc" and len(np.unique(y[keep])) < 2)):
                entries[f"{column}@{subset}"] = {"n": int(keep.sum()), "groups": n_groups, "skipped": True}
                continue
            sub = {"n": int(keep.sum()), "groups": n_groups,
                   "pos_rate": round(float(y[keep].mean()), 4) if kind == "auc" else None}
            for fs in cfg.feature_sets:
                if fs not in feats:
                    continue
                t0 = time.monotonic()
                cb = None
                if progress is not None:
                    cb = (lambda msg, c=column, f=fs: progress(f"  probes {c}/{f} {msg}"))
                vi = np.flatnonzero(valid)
                kv = keep[vi]
                X = feats[fs][kv]
                yy, gg = y[vi][kv], groups[vi][kv]
                nested = nested_probe(X, yy, gg, kind, seed=cfg.seed)
                param = statistics.median(nested["params"]) if nested["params"] else 1.0
                lo, hi = bootstrap_ci(
                    yy, nested["pred"], gg, kind, n_boot=cfg.bootstrap, seed=cfg.seed, progress=cb
                )
                perm = permutation_test(
                    X, yy, gg, kind, param, n_perm=cfg.permutations, seed=cfg.seed, progress=cb
                )
                sub[fs] = {
                    "score": nested["score"],
                    "ci": [lo, hi],
                    "perm_p": perm["p"],
                    "perm_null_q95": perm["null_q95"],
                    "param": round(float(param), 6),
                }
                if progress is not None and fs not in printed:
                    printed.add(fs)
                    done_n += 1
                    progress(f"probes {done_n}/{total} {column}/{fs} ({time.monotonic() - t0:.1f}s)")
            entries[f"{column}@{subset}"] = sub
        res.tables[column] = entries
    return res
