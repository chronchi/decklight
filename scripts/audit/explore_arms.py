"""Do uniformly random chain-0 activations lose? (NOTEBOOK 2026-11-05 15:30, user chose "test the guess offline first").

At the opponent's-turn chain-0 windows of the label job (`global[5] == 0`, `global[20] == 0`, at least one legal
ACTIVATE option), three arms are played to game end on paired seeds:
  - `p`: P's recorded move (`window_info["p"]`);
  - `search`: search's recorded move (`window_info["search"]`);
  - `rand`: `--rand-draws` uniformly random legal ACTIVATE options (what `--win-explore` samples).
Every arm of a variant shares the play seed `[idx, k, v, 91]`; a rand draw `d` is chosen with
`[idx, k, v, d, 91]`. A variant is skipped when an arm returns None or its game was cut before finishing.

    uv run python scripts/audit/explore_arms.py --win data/cache/oppfilt_in \
        --starts data/cache/line_starts/starts-v2.jsonl --shard 0/1 --max-labels 2 --out /tmp/ea/s0.json
    # merge (laptop): --merge /tmp/ea/s*.json --out /tmp/ea/summary.json

Jobs are the non-agree labels of `--win` that pass the window filter (all of them are filtered before sharding, so
the kept/dropped counts are global). Rows are written incrementally with `partial: true`. Exit 3: a record does not
match its start, every job failed, no job survives the filter, or (--merge) 0 valid windows / failed > 10% of jobs.
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
import window_alt as WA
import window_read as WR

from decklight.demos.names import CardNames
from decklight.engine.actions import Kind
from decklight.engine.duel import Engine
from decklight.eval.bc_policy import BCPolicy
from decklight.search.product import resolve_product_paths

SEED_SALT = 91


def filter_window(obs) -> tuple[bool, str]:
    """(keep, why) for the chain-0 activation windows: opponent's turn, no chain, >= 1 legal ACTIVATE."""
    if int(obs["global"][5]) != 0:
        return False, "own_turn"
    if int(obs["global"][20]) != 0:
        return False, "chain"
    mask = np.asarray(obs["action_mask"]).astype(bool)
    actions = np.asarray(obs["actions"])
    if not np.any(mask & (actions[:, 0] == int(Kind.ACTIVATE))):
        return False, "no_activate"
    return True, ""


def activate_indices(obs) -> list[int]:
    mask = np.asarray(obs["action_mask"]).astype(bool)
    actions = np.asarray(obs["actions"])
    return [int(i) for i in np.flatnonzero(mask & (actions[:, 0] == int(Kind.ACTIVATE)))]


def card_of(obs, i: int) -> int:
    """The card code of option `i` through its action card row (`actions[i][1]` -> `cards[row][0]`)."""
    row = int(np.asarray(obs["actions"])[i][1])
    cards = np.asarray(obs["cards"])
    return int(cards[row][0]) if 0 <= row < len(cards) else 0


def sign_p(b: int, w: int) -> float:
    """One-sided sign test: P(X >= b) for X ~ Bin(b + w, 1/2)."""
    return WA.sign_p(b, w)


def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _mean(xs) -> float | None:
    return float(np.mean(xs)) if len(xs) else None


def _round(x, n: int = 4):
    return None if x is None else round(x, n)


def window_stats(row: dict) -> dict:
    vs = row["variants"]
    p_w = [float(v["p"]["won"]) for v in vs]
    s_w = [float(v["search"]["won"]) for v in vs]
    r_w = [float(np.mean([d["won"] for d in v["rand"]])) for v in vs]
    p_f = [float(v["p"]["opp_field_end_turn"]) for v in vs]
    s_f = [float(v["search"]["opp_field_end_turn"]) for v in vs]
    r_f = [float(np.mean([d["opp_field_end_turn"] for d in v["rand"]])) for v in vs]
    return {
        "p_won": _mean(p_w), "search_won": _mean(s_w), "rand_won": _mean(r_w),
        "search_dwon": _mean([s - p for s, p in zip(s_w, p_w, strict=True)]),
        "rand_dwon": _mean([r - p for r, p in zip(r_w, p_w, strict=True)]),
        "p_field": _mean(p_f), "search_field": _mean(s_f), "rand_field": _mean(r_f),
        "search_dfield": _mean([s - p for s, p in zip(s_f, p_f, strict=True)]),
        "rand_dfield": _mean([r - p for r, p in zip(r_f, p_f, strict=True)]),
    }  # fmt: skip


def _pair(stats: list[dict], key: str, lower_better: bool, n_boot: int, seed: int) -> dict:
    diffs = [s[key] for s in stats if s[key] is not None]
    if not diffs:
        return {"mean": None, "ci95": None, "better": 0, "worse": 0, "sign_p": None,
                "dir": "lower_better" if lower_better else "higher_better"}  # fmt: skip
    better = sum(d < 0 for d in diffs) if lower_better else sum(d > 0 for d in diffs)
    worse = sum(d > 0 for d in diffs) if lower_better else sum(d < 0 for d in diffs)
    return {"mean": _round(_mean(diffs)), "ci95": bootstrap_ci(diffs, n_boot, seed),
            "better": int(better), "worse": int(worse), "sign_p": round(sign_p(better, worse), 4),
            "dir": "lower_better" if lower_better else "higher_better"}  # fmt: skip


def per_card(valid: list[dict]) -> list[dict]:
    tbl: dict[int, dict] = {}
    for row in valid:
        for v in row["variants"]:
            p = v["p"]["won"]
            for d in v["rand"]:
                e = tbl.setdefault(int(d["code"]), {"code": int(d["code"]), "card": d["card"], "diffs": []})
                if not e["card"]:
                    e["card"] = d["card"]
                e["diffs"].append(float(d["won"]) - float(p))
    out = [{"code": e["code"], "card": e["card"], "n_draws": len(e["diffs"]),
            "mean_won_diff": _round(_mean(e["diffs"]))} for e in tbl.values()]  # fmt: skip
    return sorted(out, key=lambda r: (-r["n_draws"], r["code"]))[:20]


def summarise(rows: list[dict], *, jobs: int = 0, failed: int = 0, bad: int = 0,
              n_boot: int = 2000, seed: int = 0) -> dict:  # fmt: skip
    valid = [r for r in rows if r.get("variants")]
    stats = [window_stats(r) for r in valid]
    n_draws = sum(len(v["rand"]) for r in valid for v in r["variants"])
    eq = sum(d["index"] == r.get("search_index") for r in valid for v in r["variants"] for d in v["rand"])
    return {
        "rows": len(rows), "valid_windows": len(valid), "jobs": jobs, "failed": failed, "bad": bad,
        "mean_win": {
            "p": _round(_mean([s["p_won"] for s in stats])),
            "search": _round(_mean([s["search_won"] for s in stats])),
            "rand": _round(_mean([s["rand_won"] for s in stats])),
            "search_minus_p": _pair(stats, "search_dwon", False, n_boot, seed),
            "rand_minus_p": _pair(stats, "rand_dwon", False, n_boot, seed + 1),
        },
        "opp_field_end_turn": {
            "p": _round(_mean([s["p_field"] for s in stats]), 3),
            "search": _round(_mean([s["search_field"] for s in stats]), 3),
            "rand": _round(_mean([s["rand_field"] for s in stats]), 3),
            "search_minus_p": _pair(stats, "search_dfield", True, n_boot, seed + 2),
            "rand_minus_p": _pair(stats, "rand_dfield", True, n_boot, seed + 3),
        },
        "per_card": per_card(valid),
        "rand_equals_search": {"draws": n_draws, "equal": int(eq),
                               "share": _round(eq / n_draws) if n_draws else None},
    }  # fmt: skip


def analyse(a: argparse.Namespace) -> int:
    parts = [json.loads(f.read_text()) for f in a.merge]
    rows = [r for p in parts for r in p["rows"]]
    jobs, failed, bad = (sum(p.get(x, 0) for p in parts) for x in ("jobs", "failed", "bad"))
    s = summarise(rows, jobs=jobs, failed=failed, bad=bad, n_boot=a.bootstrap, seed=a.boot_seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "per_card"}), flush=True)
    ok = (s["valid_windows"] > 0 and not bad and len(rows) + failed == jobs
          and failed <= 0.1 * jobs)  # fmt: skip
    if not ok:
        print(f"ALERT explore_arms: rows={len(rows)} failed={failed} bad={bad} jobs={jobs} "
              f"valid_windows={s['valid_windows']}", flush=True)  # fmt: skip
    return 0 if ok else 3


def record_config_ok(st: dict, r: dict) -> bool:
    return st["line_id"] == r["start_line_id"] and st["config"] == r["config"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=Path, default=None)
    ap.add_argument("--starts", type=Path, default=None)
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--rand-draws", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--max-labels", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--merge", nargs="*", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.merge is not None:
        return analyse(a)
    if a.win is None or a.starts is None:
        ap.error("--win and --starts are required unless --merge is given")
    if a.variants < 1 or a.rand_draws < 1:
        ap.error("--variants and --rand-draws must be >= 1")

    recs = [json.loads(x) for f in sorted(glob.glob(str(a.win / "*.jsonl")))
            for x in Path(f).read_text().splitlines() if x.strip()]  # fmt: skip
    starts = FO.training_starts(a.starts)
    nonagree = [(ri, k) for ri, r in enumerate(recs)
                for k, inf in sorted(r["window_info"].items(), key=lambda t: int(t[0]))
                if inf["kind"] != "agree"]  # fmt: skip

    engine = Engine()
    kept, dropped, bad = [], {}, 0
    for ji, (ri, ks) in enumerate(nonagree):
        r, k = recs[ri], int(ks)
        idx = int(r["line_id"].split("/oppwin-")[1].split("-")[0])
        st = starts[idx]
        if not record_config_ok(st, r):
            print(f"ALERT explore_arms: record {r['line_id']} does not match start {idx}", flush=True)
            bad += 1
            continue
        try:
            obs = FO.window_obs(engine, st, [int(c) for c in r["choices"][:k]])
            if obs is None:
                dropped["noreplay"] = dropped.get("noreplay", 0) + 1
                continue
            ok, why = filter_window(obs)
            if ok:
                kept.append((ri, ks))
            else:
                dropped[why] = dropped.get(why, 0) + 1
        except Exception as e:  # noqa: BLE001  counted; the filter must not kill the run
            dropped["error"] = dropped.get("error", 0) + 1
            print(f"filter {r['line_id']} k={k} failed: {e!r}", flush=True)
        if (ji + 1) % 50 == 0:
            print(f"filter {ji + 1}/{len(nonagree)} kept={len(kept)} dropped={dropped}", flush=True)
    print(json.dumps({"nonagree": len(nonagree), "kept": len(kept), "dropped": dropped, "bad": bad}), flush=True)
    if not kept:
        print("ALERT explore_arms: no window survives the filter", flush=True)
        return 3

    i, n = (int(x) for x in a.shard.split("/"))
    jobs = kept[i::n][: a.max_labels or None]
    ckpt, _, emb = resolve_product_paths(None, None, None)
    pol, names = BCPolicy(ckpt, emb, threads=1, device=a.device), CardNames().names
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    filt = {"nonagree": len(nonagree), "kept": len(kept), "dropped": dropped, "bad": bad}
    for ji, (ri, ks) in enumerate(jobs):
        r, k = recs[ri], int(ks)
        inf = r["window_info"][ks]
        idx = int(r["line_id"].split("/oppwin-")[1].split("-")[0])
        st, nvalid = starts[idx], 0
        try:
            me, prefix = int(st["our_seat"]), [int(c) for c in r["choices"][:k]]
            obs = FO.window_obs(engine, st, prefix)
            if obs is None:
                raise RuntimeError("the recorded prefix does not replay")
            want = {x: np.asarray(v).copy() for x, v in obs.items()}
            acts = activate_indices(want)
            if not acts:
                raise RuntimeError("no legal ACTIVATE in the recorded window")
            at = WR.our_calls_before(engine, st, r["choices"], k, me)
            vs, _ = FO.reaching_variants(engine, st, idx, prefix, want, a.variants, a.tries)
            variants = []
            for v in vs:
                sv = st if v >= FO.TOPUP else FO.variant_start(st, v, idx)
                seed = [idx, k, v, SEED_SALT]
                g = {nm: FO.play_arm(engine, sv, pol, at, inf[nm], want, a.temp, seed, prefix)
                     for nm in ("search", "p")}  # fmt: skip
                draws = []
                for d in range(a.rand_draws):
                    rng = np.random.default_rng([idx, k, v, d, SEED_SALT])
                    ai = acts[int(rng.integers(len(acts)))]
                    gr = FO.play_arm(engine, sv, pol, at, ai, want, a.temp, seed, prefix)
                    if gr is None or gr["won"] is None:
                        draws.append(None)
                        continue
                    code, label = card_of(want, ai), WR.label(want, ai, names)
                    draws.append({"won": gr["won"], "opp_field_end_turn": gr["opp_field_end_turn"],
                                  "index": int(ai), "code": code, "card": names.get(code) or label,
                                  "label": label})  # fmt: skip
                if any(x is None or x["won"] is None for x in g.values()) or any(d is None for d in draws):
                    continue
                variants.append({"v": int(v),
                                 "p": {"won": g["p"]["won"], "opp_field_end_turn": g["p"]["opp_field_end_turn"]},
                                 "search": {"won": g["search"]["won"],
                                            "opp_field_end_turn": g["search"]["opp_field_end_turn"]},
                                 "rand": draws})  # fmt: skip
            rows.append({"rec": r["line_id"], "k": k, "orig": inf["kind"], "start_kind": st.get("kind"),
                         "valid": len(variants), "search_index": int(inf["search"]),
                         "p_index": int(inf["p"]), "variants": variants})  # fmt: skip
            nvalid = len(variants)
        except Exception as e:  # noqa: BLE001  counted; > 10% fails the merge
            failed += 1
            print(f"label {r['line_id']} k={k} failed: {e!r}", flush=True)
        print(f"{ji + 1}/{len(jobs)} {r['line_id']} k={k} valid={nvalid}", flush=True)
        a.out.write_text(json.dumps({"rows": rows, "failed": failed, "bad": bad, "jobs": len(jobs),
                                     "filter": filt, "partial": True}))  # fmt: skip
    a.out.write_text(json.dumps({"rows": rows, "failed": failed, "bad": bad, "jobs": len(jobs),
                                 "filter": filt}, indent=1))  # fmt: skip
    print(json.dumps({"rows": len(rows), "failed": failed, "bad": bad, "jobs": len(jobs),
                      "valid_windows": sum(r["valid"] > 0 for r in rows)}), flush=True)  # fmt: skip
    return 3 if bad or (jobs and failed == len(jobs)) else 0


if __name__ == "__main__":
    sys.exit(main())
