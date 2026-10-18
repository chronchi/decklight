"""Fit the win-chance calibration map (isotonic per turn bucket) on scored games.

Reads value lines ``runs/score-20261016-1001/{arm}/values-*.jsonl`` (one line per
game: game_idx, seed, decks, s_seat, k[], actor[], turn[], v[] with v the
actor's-view value in [-1, 1]) and joins winners from the game records they
were scored from (``runs/value-p2-20261015-2044/{arm}/games-*.jsonl``, via
``score_decisions.iter_records``) on (game_idx, seed, decks). Target y = 1 if
the actor won; p_raw = (v + 1) / 2. One PAV map p_raw -> p per turn bucket
(<=2, 3-4, 5+).

Evaluation is 5 game-level 80/20 splits (seeds ``--seed``..``--seed`` + 4);
the gate uses the mean held-out ECE/Brier change per bucket. Decisions within
a game are correlated, so one split is too noisy to trust. The stored map is
fit on 100% of the games once the gate passes.

Writes ``src/ygo_zero/web/winchance_calib.json`` (committed) and writes the
report fresh to ``docs/annotations/2026-10-16-hppo-p2/winchance-calibration.md``
(one report per run, never appended).

Invariants (exit 2): a game with no winner; the join dropping > 1% of games;
any bucket with held-out n < 1000 on any split; a non-monotone map; any
bucket whose mean held-out ECE or Brier is worse after calibration than raw
(that bucket gets an identity map instead, and the run still exits 2 so the
regression is never silently kept).

No model work: a 1-D fit on saved numbers (laptop CPU).
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import statistics
import sys
from pathlib import Path
from typing import NoReturn

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ygo_zero.eval import winchance as W


def fail(msg: str) -> NoReturn:
    print(f"ALERT fit_winchance: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-dir", type=Path, default=ROOT / "runs" / "score-20261016-1001")
    ap.add_argument("--src", type=Path, default=ROOT / "runs" / "value-p2-20261015-2044")
    ap.add_argument("--arms", nargs="+", default=["hppo-p", "hppo-p2"])
    ap.add_argument("--out", type=Path, default=ROOT / "src" / "ygo_zero" / "web" / "winchance_calib.json")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "docs" / "annotations" / "2026-10-16-hppo-p2" / "winchance-calibration.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=float, default=0.8)
    a = ap.parse_args()
    seeds = [a.seed + i for i in range(5)]

    from score_decisions import iter_records

    # --- join winners per arm ---
    games: dict[tuple, dict] = {}  # (arm, game_idx, seed, decks) -> {winner, truncated}
    n_rec = 0
    for arm in a.arms:
        d = a.src / arm
        if not d.is_dir():
            fail(f"no such dir: {d}")
        for r in iter_records([d], 0):
            n_rec += 1
            key = (arm, r.get("game_idx"), r.get("seed"), tuple(r.get("decks", [])))
            if key in games:
                fail(f"duplicate game record {key}")
            if r.get("winner") is None:
                fail(f"game {key}: no winner in the record")
            games[key] = {"winner": int(r["winner"]), "truncated": bool(r.get("truncated"))}

    rows: list[tuple[tuple, int, float, int]] = []  # (game key, bucket, p_raw, y)
    n_val, n_drop, n_trunc = 0, 0, 0
    for arm in a.arms:
        d = a.score_dir / arm
        files = sorted(d.glob("values-*.jsonl"))
        if not files:
            fail(f"no values-*.jsonl in {d}")
        for f in files:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                g = json.loads(line)
                n_val += 1
                key = (arm, g.get("game_idx"), g.get("seed"), tuple(g.get("decks", [])))
                rec = games.get(key)
                if rec is None:
                    n_drop += 1
                    continue
                if rec["truncated"]:
                    n_trunc += 1
                    continue
                winner = rec["winner"]
                for actor, turn, v in zip(g["actor"], g["turn"], g["v"], strict=True):
                    rows.append((key, W.bucket_of(int(turn)), (float(v) + 1.0) / 2.0,
                                 1 if int(actor) == winner else 0))
    if n_val == 0:
        fail("0 value games read")
    if n_drop / n_val > 0.01:
        fail(f"join dropped {n_drop}/{n_val} games (> 1%)")
    print(f"games: {n_rec} records, {n_val} scored, {n_trunc} truncated excluded, "
          f"{n_drop} unscored dropped", flush=True)

    keys = sorted({r[0] for r in rows})
    by_bucket: dict[int, list[tuple[float, float]]] = {b: [] for b in range(len(W.BUCKETS))}
    for _, b, p_raw, y in rows:
        by_bucket[b].append((p_raw, float(y)))

    # --- 5-split evaluation per bucket (fit PAV on fit games, score held-out) ---
    maps: list[tuple[list, list]] = []
    metrics: list[dict] = []
    failed: list[str] = []
    for b in range(len(W.BUCKETS)):
        ece_raw_s, ece_cal_s, br_raw_s, br_cal_s, held_ns = [], [], [], [], []
        for s in seeds:
            rng = random.Random(s)
            order = list(keys)
            rng.shuffle(order)
            n_fit = int(len(order) * a.split)
            fit_keys, held_keys = set(order[:n_fit]), set(order[n_fit:])
            fx = np.array([r[2] for r in rows if r[0] in fit_keys and r[1] == b])
            fy = np.array([r[3] for r in rows if r[0] in fit_keys and r[1] == b])
            hx = np.array([r[2] for r in rows if r[0] in held_keys and r[1] == b])
            hy = np.array([r[3] for r in rows if r[0] in held_keys and r[1] == b])
            held_ns.append(len(hy))
            if len(fx) == 0:
                fail(f"bucket {W.bucket_name(b)}: no fit data (seed {s})")
            xs, ys = W.pav_increasing(fx, fy)
            tx, ty = W.thin_knots(xs, ys, W.MAX_KNOTS)
            h_cal = W.apply_knots(hx, tx, ty)
            ece_raw_s.append(W.ece_equal_mass(hx, hy))
            ece_cal_s.append(W.ece_equal_mass(h_cal, hy))
            br_raw_s.append(W.brier(hx, hy))
            br_cal_s.append(W.brier(h_cal, hy))
        if min(held_ns) < 1000:
            fail(f"bucket {W.bucket_name(b)}: held-out n={min(held_ns)} < 1000 on some split")
        ece_raw_m, ece_cal_m = statistics.fmean(ece_raw_s), statistics.fmean(ece_cal_s)
        br_raw_m, br_cal_m = statistics.fmean(br_raw_s), statistics.fmean(br_cal_s)
        ece_cal_sd = statistics.stdev(ece_cal_s) if len(ece_cal_s) > 1 else 0.0
        br_cal_sd = statistics.stdev(br_cal_s) if len(br_cal_s) > 1 else 0.0
        worse_ece = ece_cal_m > ece_raw_m
        worse_br = br_cal_m > br_raw_m
        if worse_ece or worse_br:
            maps.append(([0.0, 1.0], [0.0, 1.0]))  # identity: calibrated = raw, clipped
            failed.append(W.bucket_name(b))
            status = "IDENTITY (gate failed)"
        else:
            fx = np.array([p for p, _ in by_bucket[b]])
            fy = np.array([y for _, y in by_bucket[b]])
            xs, ys = W.pav_increasing(fx, fy)
            tx, ty = W.thin_knots(xs, ys, W.MAX_KNOTS)
            if len(tx) > W.MAX_KNOTS:
                fail(f"bucket {W.bucket_name(b)}: {len(tx)} knots > {W.MAX_KNOTS}")
            maps.append((tx, ty))
            status = "fitted"
        n_tot = len(by_bucket[b])
        # Reliability of the stored map on all data (the map was fit on 100%).
        pall = np.array([p for p, _ in by_bucket[b]])
        yall = np.array([y for _, y in by_bucket[b]])
        call = W.apply_knots(pall, *maps[-1])
        metrics.append({"bucket": W.bucket_name(b), "n": n_tot,
                        "n_fit_mean": round(sum(
                            len([r for r in rows if r[1] == b]) for _ in seeds) / len(seeds) * a.split),
                        "n_held_mean": round(statistics.fmean(held_ns)),
                        "n_knots": len(maps[-1][0]), "status": status,
                        "ece_raw_mean": round(ece_raw_m, 4), "ece_cal_mean": round(ece_cal_m, 4),
                        "ece_cal_sd": round(ece_cal_sd, 4),
                        "brier_raw_mean": round(br_raw_m, 4), "brier_cal_mean": round(br_cal_m, 4),
                        "brier_cal_sd": round(br_cal_sd, 4),
                        # Back-compat aliases for the single-split fields.
                        "n_fit": n_tot, "n_held": n_tot,
                        "ece_raw": round(ece_raw_m, 4), "ece_cal": round(ece_cal_m, 4),
                        "brier_raw": round(br_raw_m, 4), "brier_cal": round(br_cal_m, 4),
                        "reliability": W.reliability_table(call, yall)})

    # --- report (written fresh: one report per run) ---
    date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    lines = [f"# Win-chance calibration ({date})",
             "",
             (f"Source: {a.score_dir.name} values (arms {', '.join(a.arms)}) + winners from "
              f"{a.src.name}; {len(keys)} games, 5 game-level {a.split}/{round(1 - a.split, 2)} splits "
              f"(seeds {seeds[0]}-{seeds[-1]}), means over splits. Stored map fit on 100% of the games. "
              f"Target y = 1 if the actor won; p_raw = (v + 1) / 2. Outputs clip to "
              f"[{W.P_LO}, {W.P_HI}]."),
             "",
             "| bucket | n | knots | status | ECE raw | ECE cal mean ± sd | Brier raw | Brier cal mean ± sd |",
             "|---|---|---|---|---|---|---|---|"]
    for m in metrics:
        lines.append(f"| {m['bucket']} | {m['n']} | {m['n_knots']} | {m['status']} | "
                     f"{m['ece_raw_mean']:.4f} | {m['ece_cal_mean']:.4f} ± {m['ece_cal_sd']:.4f} | "
                     f"{m['brier_raw_mean']:.4f} | {m['brier_cal_mean']:.4f} ± {m['brier_cal_sd']:.4f} |")
    for m in metrics:
        lines += ["", f"Reliability, bucket {m['bucket']} (stored map, all data, 10 equal-mass bins):",
                  "", "| bin | n | p range | mean p | win rate |", "|---|---|---|---|---|"]
        for i, r in enumerate(m["reliability"]):
            lines.append(f"| {i} | {r['n']} | {r['p_lo']:.3f}-{r['p_hi']:.3f} | "
                         f"{r['mean_p']:.3f} | {r['win_rate']:.3f} |")
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(report)  # fresh every run: never append

    payload = {"source_run": str(a.score_dir) + " + " + str(a.src), "arms": list(a.arms), "date": date,
               "split_seeds": seeds, "split": [a.split, round(1 - a.split, 2)],
               "n_games": len(keys),
               "n_games_fit": len(keys), "n_games_held": 0,
               "clip": [W.P_LO, W.P_HI],
               "map": W.CalibMap(maps).to_dict(), "metrics": metrics}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {a.out} + wrote {a.report}", flush=True)
    if failed:
        fail(f"buckets worse after calibration (mean over {len(seeds)} splits), "
             f"stored identity map: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
