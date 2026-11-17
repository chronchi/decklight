"""Shared helpers: legal-option softmax, public count features, grouped CV, permutation/bootstrap, Spearman."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable

import numpy as np

PHASE_NAMES = {
    0x01: "draw",
    0x02: "standby",
    0x04: "main1",
    0x08: "battle_start",
    0x10: "battle_step",
    0x20: "battle",
    0x40: "damage",
    0x80: "damage_calc",
    0x100: "battle_end",
    0x200: "main2",
    0x400: "end",
}


def phase_name(phase: int) -> str:
    return PHASE_NAMES.get(int(phase), f"phase{int(phase)}")


def softmax_options(logits: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Softmax over legal options only (illegal options get probability 0)."""
    legal = np.asarray(action_mask, dtype=bool)
    out = np.zeros(legal.shape[0], dtype=np.float64)
    z = np.asarray(logits[: legal.shape[0]], dtype=np.float64).copy()
    z[~legal] = -np.inf
    finite = np.isfinite(z)
    if not finite.any():
        return out
    v = z[finite] - z[finite].max()
    e = np.exp(v)
    out[finite] = e / e.sum()
    return out


def base_features(obs: dict) -> np.ndarray:
    """Public counts read from the obs (no model), as in `scripts/audit/opp_probe.py`."""
    c = np.asarray(obs["cards"])
    m = np.asarray(obs["card_mask"]).astype(bool)
    c = c[m]
    f = [((c[:, 3] == side) & (c[:, 1] == loc)).sum() for side in (0, 1) for loc in (2, 3, 4, 5, 6)]
    h = np.asarray(obs["history"])
    hm = np.asarray(obs["history_mask"]).astype(bool)
    h = h[hm]
    now = h[:, 11] == 0
    f += [
        int(np.asarray(obs["chain_mask"]).sum()),
        int((now & (h[:, 10] == 1)).sum()),
        int((now & (h[:, 10] == 0)).sum()),
    ]
    g = np.asarray(obs["global"])[:22].astype(float)
    return np.concatenate([np.asarray(f, float), g])


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks of ties (0-based)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x))
    out[order] = np.arange(len(x))
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            out[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return out


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x, y = np.asarray(list(x), float), np.asarray(list(y), float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def _cv_splits(n: int, groups: np.ndarray, k: int):
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    k = max(2, min(k, n_groups))
    return list(GroupKFold(k).split(np.zeros(n), np.zeros(n), groups))


def _make_model(kind: str, param: float):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "auc":
        return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=3000, C=param))])
    return Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=param))])


def _score(kind: str, y: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import r2_score, roc_auc_score

    if kind == "auc":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, pred))
    if len(y) < 2 or np.ptp(y) == 0:
        return float("nan")
    return float(r2_score(y, pred))


def nested_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str,
    *,
    inner: int = 4,
    seed: int = 0,
) -> dict:
    """Grouped nested CV: outer 5 folds (by start), inner folds pick C/alpha; returns out-of-fold predictions."""
    from sklearn.model_selection import GridSearchCV

    y = np.asarray(y, float)
    pred = np.zeros(len(y))
    picked: list[float] = []
    if len(np.unique(groups)) < 2:
        return {"score": float("nan"), "pred": pred, "params": picked}
    splits = _cv_splits(len(y), groups, 5)
    for tr, te in splits:
        if kind == "auc" and len(np.unique(y[tr])) < 2:
            pred[te] = 0.0
            continue
        if len(np.unique(groups[tr])) < 2:  # too few starts to tune inside this fold: fixed regularisation
            m = _make_model(kind, 1.0).fit(X[tr], y[tr])
            picked.append(1.0)
        else:
            grid = {"m__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]} if kind == "auc" else {
                "m__alpha": [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
            }
            gs = GridSearchCV(
                _make_model(kind, 1.0),
                grid,
                scoring="roc_auc" if kind == "auc" else "r2",
                cv=_cv_splits(len(tr), groups[tr], inner),
                error_score=float("nan"),
            )
            with warnings.catch_warnings():
                from sklearn.exceptions import ConvergenceWarning, FitFailedWarning, UndefinedMetricWarning

                for cat in (FitFailedWarning, UndefinedMetricWarning, ConvergenceWarning, UserWarning):
                    warnings.simplefilter("ignore", cat)
                gs.fit(X[tr], y[tr], groups=groups[tr])
            if not math.isfinite(float(getattr(gs, "best_score_", float("nan")))):
                m = _make_model(kind, 1.0).fit(X[tr], y[tr])
                picked.append(1.0)
            else:
                picked.append(float(next(iter(gs.best_params_.values()))))
                m = gs
        pred[te] = m.predict_proba(X[te])[:, 1] if kind == "auc" else m.predict(X[te])
    return {"score": round(_score(kind, y, pred), 4), "pred": pred, "params": picked}


def cv_predict(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, param: float, folds: int = 5
) -> np.ndarray:
    """Out-of-fold predictions with a fixed regularisation (used for permutation tests)."""
    y = np.asarray(y, float)
    pred = np.zeros(len(y))
    if len(np.unique(groups)) < 2:
        return np.full(len(y), float(np.mean(y)))
    for tr, te in _cv_splits(len(y), groups, folds):
        if len(np.unique(y[tr])) < 2:
            pred[te] = float(np.mean(y[tr]))
            continue
        m = _make_model(kind, param)
        with warnings.catch_warnings():
            from sklearn.exceptions import ConvergenceWarning

            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1] if kind == "auc" else m.predict(X[te])
    return pred


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    kind: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    progress=None,
) -> tuple[float | None, float | None]:
    """Percentile CI over starts (groups resampled with replacement)."""
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    n = max(1, n_boot)
    step = max(1, n // 10)
    for i in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        s = _score(kind, y[idx], pred[idx])
        if math.isfinite(s):
            vals.append(s)
        if progress is not None and (i + 1) % step == 0:
            progress(f"bootstrap {i + 1}/{n}")
    if not vals:
        return None, None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str,
    param: float,
    *,
    n_perm: int = 200,
    seed: int = 0,
    progress=None,
) -> dict:
    """Start-level permutation test against the nested-CV out-of-fold predictions.

    The predictions are not refit for each permutation: under the null they carry no start-level signal, so
    permuting whole-start labels against them is a valid randomization test for association with the model's
    out-of-fold scores (and it is cheap enough to run 200+ times on the laptop). Each start keeps its label
    multiset: groups are permuted and every window of a target start gets its donor start's label.
    """
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    pred = cv_predict(X, y, groups, kind, param)
    obs = _score(kind, y, pred)
    uniq = np.unique(groups)
    first = np.asarray([y[groups == g][0] for g in uniq])
    null = []
    n = max(1, n_perm)
    step = max(1, n // 10)
    for i in range(n):
        donor = rng.permutation(len(uniq))
        yp = np.empty_like(y)
        for k, g in enumerate(uniq):
            yp[groups == g] = first[donor[k]]
        if len(np.unique(yp)) >= 2:
            null.append(_score(kind, yp, pred))
        if progress is not None and (i + 1) % step == 0:
            progress(f"permutation {i + 1}/{n}")
    null = np.asarray([v for v in null if math.isfinite(v)])
    if not math.isfinite(obs) or not len(null):
        return {"observed": None, "p": None, "null_mean": None, "null_q95": None}
    p = (1 + int((null >= obs).sum())) / (1 + len(null))
    return {
        "observed": round(float(obs), 4),
        "p": round(float(p), 4),
        "null_mean": round(float(null.mean()), 4),
        "null_q95": round(float(np.percentile(null, 95)), 4),
    }

def finite_tree(obj) -> bool:
    """True when every float in a nested JSON-like structure is finite."""
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(finite_tree(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(finite_tree(v) for v in obj)
    if isinstance(obj, np.floating):
        return math.isfinite(float(obj))
    if isinstance(obj, np.ndarray):
        return bool(np.isfinite(obj).all())
    return True
