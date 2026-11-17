"""Re-analysis of `opp_probe.py` outputs with the regularisation chosen by grouped CV (NOTEBOOK 2026-11-05 04:50):
RidgeCV's leave-one-out picks alpha across windows of the same start, which leaks and overfits. Nested: outer 5-fold
grouped by start, inner 4-fold grouped by start picks C / alpha. Reads <dir>/features.npz + windows.jsonl.

    uv run python scripts/audit/opp_probe_cv.py runs/opp-probe-20261105
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def nested(X, y, g, kind: str) -> tuple[float, list]:
    pred, picked = np.zeros(len(y)), []
    for tr, te in GroupKFold(5).split(X, y, g):
        if kind == "auc":
            est, grid, sc = (
                LogisticRegression(max_iter=3000),
                {"m__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
                "roc_auc",
            )
        else:
            est, grid, sc = Ridge(), {"m__alpha": [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]}, "r2"
        gs = GridSearchCV(Pipeline([("s", StandardScaler()), ("m", est)]), grid, scoring=sc, cv=GroupKFold(4))
        gs.fit(X[tr], y[tr], groups=g[tr])
        picked.append(next(iter(gs.best_params_.values())))
        pred[te] = gs.predict_proba(X[te])[:, 1] if kind == "auc" else gs.predict(X[te])
    return round(float(roc_auc_score(y, pred) if kind == "auc" else r2_score(y, pred)), 3), picked


def main() -> int:
    d = Path(sys.argv[1])
    F = np.load(d / "features.npz")
    rows = [json.loads(x) for x in (d / "windows.jsonl").read_text().splitlines()]
    g = F["start"]
    X = {k: F[k] for k in ("base", "glob", "opt")}
    X["glob+base"] = np.concatenate([X["glob"], X["base"]], 1)
    passes = np.asarray([x["p_passes"] for x in rows], bool)
    rng = np.random.default_rng(0)
    out = {}
    for tgt, kind in (("complete", "auc"), ("field", "r2")):
        for sub in ("all", "p_passes"):
            m = passes if sub == "p_passes" else np.ones(len(rows), bool)
            y = np.asarray([x[tgt] for x in rows], float)[m]
            r = {"n": int(m.sum())}
            for fs, Xf in X.items():
                r[fs] = nested(Xf[m], y, g[m], kind)
            ys = y.copy()
            for s in np.unique(g[m]):
                ys[g[m] == s] = rng.choice(y)
            r["glob_shuffled"] = nested(X["glob"][m], ys, g[m], kind)
            out[f"{tgt}@{sub}"] = r
            print(f"{tgt}@{sub}", json.dumps(r), flush=True)
    (d / "probe_cv.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
