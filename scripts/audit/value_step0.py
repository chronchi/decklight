"""Step 0 of the value-guided opponent-turn targets (docs/plans/2026-11-04-opp-turn-value-target.md, NOTEBOOK
2026-11-04 23:40, pre-registered): does the product value tell an interrupted opponent turn from a completed one?

For every non-agree window label of the label job (the 340 the seed-resampled filter played out), both arms (search's
move, P's move) are played `--variants` times exactly as in `filter_opp_windows` (same reaching variants, P at
`--temp` from the window on, the opponent's script until broken, then P), but on fresh seeds and only until the line's
turn ends. Q(arm) = mean over variants of the product value (our view) at the first observation of the next turn, or
±1 / 0 when the game ended inside the turn. dQ = Q(search) - Q(P); also the mean opponent field at the end of the turn.

    uv run python scripts/audit/value_step0.py --win data/cache/oppfilt_in --starts data/cache/line_starts/starts-v2.jsonl \
        --shard 0/1 --max-labels 1 --out /tmp/v0.json
    # analysis (laptop): --merge <dir>/s*.json --truth runs/filter-opp-20261104-1424/parts/*.json --out <dir>/step0.json

Exit 3: a mismatched record, every label of a shard failed, or (--merge) missing labels / failed > 10% / dQ all zero.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import filter_opp_windows as FO
import line_eval as LE
import window_read as WR

from decklight.demos.dataset import record_config
from decklight.engine.duel import Engine
from decklight.engine.env import PyEnv
from decklight.eval.bc_policy import BCPolicy
from decklight.search.product import load_value, resolve_product_paths
from decklight.search.turn1 import _batch

SEED_SALT = 77  # fresh seeds: the filter used [idx, k, v]


def line_turn(engine, st: dict) -> int:
    env = PyEnv(engine, record_config(st), st.get("auto_single", True))
    env.reset()
    for c in st["prefix"]:
        env.step(int(c))
    return int(env.duel.turn)


def play_arm(engine, st, pol, at, a, want, temp, seed, prefix, t0) -> dict | None:
    """`filter_opp_windows.play_arm`, stopped when the line's turn `t0` ends (None: the variant left the line)."""
    pol.temperature = 0.0
    f = FO.Resampled(pol, at, a, want, temp, np.random.default_rng(seed))
    try:
        g = LE.play(engine, st, f, pol, stop_turn=t0, keep_stop=True)
    except RuntimeError as e:
        if "different position" in str(e):
            return None
        raise
    finally:
        pol.temperature = 0.0
    if not f.hit or [int(c) for c in g["choices"][: len(prefix)]] != prefix:
        return None
    return g


def rank(x) -> np.ndarray:
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):  # average ranks of ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def analyse(a: argparse.Namespace) -> int:
    parts = [json.loads(f.read_text()) for f in a.merge]
    rows = [r for p in parts for r in p["rows"]]
    jobs, failed, bad = (sum(p[x] for p in parts) for x in ("jobs", "failed", "bad"))
    truth = {(r["rec"], r["k"]): r for f in a.truth for r in json.loads(Path(f).read_text())["rows"]}
    both = [
        (r, truth[(r["rec"], r["k"])]) for r in rows if (r["rec"], r["k"]) in truth and r["dq"] is not None
    ]
    v8 = [(r, t) for r, t in both if t["valid"] >= 8]
    dq = [r["dq"] for r, _ in v8]
    wr = [(t["search"] - t["p"]) / t["valid"] for _, t in v8]
    fq = [(r["dq"], r["dfield"]) for r, _ in both if r["dfield"] is not None]
    kept = [(r, t) for r, t in both if t["keep"] is not None and r["dq"] != 0]
    agree = sum((r["dq"] > 0) == (t["keep"] == "search") for r, t in kept)
    s = {"jobs": jobs, "rows": len(rows), "failed": failed, "bad": bad, "with_truth": len(both),
         "b_windows_valid8": len(v8), "b_spearman_dq_winrate": round(spearman(dq, wr), 3),
         "c_windows": len(fq), "c_spearman_dq_dfield": round(spearman(*zip(*fq)), 3) if len(fq) > 2 else None,
         "a_kept_nonzero": len(kept), "a_sign_agree": agree, "a_frac": round(agree / max(len(kept), 1), 3),
         "dq_zero": sum(r["dq"] == 0 for r, _ in both), "dq_mean_abs": round(float(np.mean(np.abs(dq))), 4) if dq else 0,
         "se_mean": round(float(np.mean([r["se"] for r, _ in both if r["se"] is not None])), 4) if both else None}  # fmt: skip
    s["b_pass"] = s["b_spearman_dq_winrate"] >= 0.30
    s["c_pass"] = s["c_spearman_dq_dfield"] is not None and s["c_spearman_dq_dfield"] <= -0.30
    top = sorted(both, key=lambda rt: -abs(rt[0]["dq"]))[:10]
    s["top10"] = [{"rec": r["rec"], "k": r["k"], "dq": r["dq"], "dfield": r["dfield"], "truth_s_p_valid":
                   (t["search"], t["p"], t["valid"]), "keep": t["keep"]} for r, t in top]  # fmt: skip
    a.out.write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "top10"}), flush=True)
    ok = rows and not bad and len(rows) + failed == jobs and failed <= 0.1 * jobs and s["dq_zero"] < len(both)
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--max-labels", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--merge", nargs="*", type=Path, default=None)
    ap.add_argument("--truth", nargs="*", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.merge is not None:
        return analyse(a)
    recs = [json.loads(x) for f in sorted(glob.glob(str(a.win / "*.jsonl")))
            for x in Path(f).read_text().splitlines() if x.strip()]  # fmt: skip
    starts = FO.training_starts(a.starts)
    jobs = [(ri, k) for ri, r in enumerate(recs) for k, inf in sorted(r["window_info"].items(), key=lambda t: int(t[0]))
            if inf["kind"] != "agree"]  # fmt: skip
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = jobs[i::n][: a.max_labels or None]
    ckpt, value, emb = resolve_product_paths(None, None, None)
    pol, engine = BCPolicy(ckpt, emb, threads=1, device=a.device), Engine()
    val = load_value(value, emb, device=a.device)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows, failed, bad = [], 0, 0
    for ji, (ri, ks) in enumerate(jobs):
        r, k = recs[ri], int(ks)
        inf = r["window_info"][ks]
        idx = int(r["line_id"].split("/oppwin-")[1].split("-")[0])
        st = starts[idx]
        if st["line_id"] != r["start_line_id"] or st["config"] != r["config"]:
            print(f"ALERT value_step0: record {r['line_id']} does not match start {idx}", flush=True)
            bad += 1
            continue
        try:
            me, prefix = int(st["our_seat"]), [int(c) for c in r["choices"][:k]]
            obs = FO.window_obs(engine, st, prefix)
            if obs is None:
                raise RuntimeError("the recorded prefix does not replay")
            want = {x: np.asarray(v).copy() for x, v in obs.items()}
            at = WR.our_calls_before(engine, st, r["choices"], k, me)
            t0 = line_turn(engine, st)
            vs, _ = FO.reaching_variants(engine, st, idx, prefix, want, a.variants, a.tries)
            per: dict[str, list] = {"search": [], "p": []}
            for v in vs:
                sv = st if v >= FO.TOPUP else FO.variant_start(st, v, idx)
                g = {nm: play_arm(engine, sv, pol, at, inf[nm], want, a.temp, [idx, k, v, SEED_SALT], prefix, t0)
                     for nm in ("search", "p")}  # fmt: skip
                if any(x is None or (x["stop_obs"] is None and x["won"] is None) for x in g.values()):
                    continue  # left the line, or cut at the decision cap inside the turn
                for nm, x in g.items():
                    per[nm].append(x)
            m = len(per["p"])
            if m:
                stop = [(nm, j) for nm in per for j, x in enumerate(per[nm]) if x["stop_obs"] is not None]
                vals = val(_batch([per[nm][j]["stop_obs"] for nm, j in stop])) if stop else []
                q = {
                    nm: [2.0 * x["won"] - 1.0 if x["stop_obs"] is None else None for x in per[nm]]
                    for nm in per
                }
                for (nm, j), x in zip(stop, vals, strict=True):
                    q[nm][j] = float(x) if per[nm][j]["stop_actor"] == me else -float(x)
                d = np.asarray(q["search"]) - np.asarray(q["p"])
                fs = [float(np.mean([x["opp_field_end_turn"] for x in per[nm]])) for nm in ("search", "p")]
                row = {"dq": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(m)), 4) if m > 1
                       else None, "q_search": q["search"], "q_p": q["p"], "dfield": round(fs[0] - fs[1], 3)}  # fmt: skip
            else:
                row = {"dq": None, "se": None, "q_search": [], "q_p": [], "dfield": None}
            rows.append({"rec": r["line_id"], "k": k, "orig": inf["kind"], "valid": m, **row,
                         "start_kind": st.get("kind")})  # fmt: skip
        except Exception as e:  # noqa: BLE001  counted; > 10% fails the merge
            failed += 1
            print(f"label {r['line_id']} k={k} failed: {e!r}", flush=True)
        print(f"{ji + 1}/{len(jobs)} {json.dumps(rows[-1]) if rows else ''}", flush=True)
        a.out.write_text(
            json.dumps({"rows": rows, "failed": failed, "bad": bad, "jobs": len(jobs), "partial": True})
        )
    a.out.write_text(json.dumps({"rows": rows, "failed": failed, "bad": bad, "jobs": len(jobs)}, indent=1))
    return 3 if bad or (jobs and failed == len(jobs)) else 0


if __name__ == "__main__":
    sys.exit(main())
