"""Free checks before M2 (docs/plans/2026-10-20-m2-review.md; user 2026-10-20: "Let's run the free checks").
No inference: value_p2 scores already stored for every decision (runs/score-20261016-1001, N2a) joined to winners,
plus the M1 fork rows.

  critic   explained variance of value_p2 against the Monte Carlo outcome (z = +1 actor wins, -1 loses), per turn
           bucket, over every decision and at the first decision of each turn (EV = 1 - Var(z - v) / Var(z));
  boot     a turn-end bootstrap's error: v at each actor's first decision of turn 2 (the bootstrap point that closes
           turn 1) minus z, overall and per actor deck (decks with >= 20 games), against the spread expected from
           noise alone;
  deal     residual deal share on the M1 rows: ICC of the 16 sampled results of the played option per state, and
           what is left after a linear fit of the state's mean result on its mean value at the end of the turn.

    python scripts/audit/m2_checks.py [--score-dir runs/score-20261016-1001] [--src runs/value-p2-20261015-2044]
        [--fork-rows runs/forkpilot-20261020-1913/rows]
Exits 2 when a join drops > 1% of games or a bucket is empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5+", 5, 10**6))


def fail(msg: str) -> None:
    print(f"ALERT m2_checks: {msg}", flush=True)
    raise SystemExit(2)


def ev(z: np.ndarray, v: np.ndarray) -> float:
    return float(1 - np.var(z - v) / np.var(z))


def load(score_dir: Path, src: Path, arms: list[str]):
    """Per game: (arm, decks, winner, [(k, actor, turn, v)])."""
    games = []
    for arm in arms:
        win = {}
        for f in sorted((src / arm).glob("games-*.jsonl")):
            for x in f.read_text().splitlines():
                if x.strip():
                    r = json.loads(x)
                    win[(r["game_idx"], r["seed"], tuple(r["decks"]))] = (
                        int(r["winner"]),
                        bool(r.get("truncated")),
                    )
        n = drop = 0
        for f in sorted((score_dir / arm).glob("values-*.jsonl")):
            for x in f.read_text().splitlines():
                if not x.strip():
                    continue
                r = json.loads(x)
                n += 1
                w = win.get((r["game_idx"], r["seed"], tuple(r["decks"])))
                if w is None or w[1] or w[0] not in (0, 1):
                    drop += 1
                    continue
                games.append((arm, r["decks"], w[0], list(zip(r["k"], r["actor"], r["turn"], r["v"]))))
        if n == 0 or drop > 0.01 * n:
            fail(f"{arm}: {drop}/{n} games dropped by the winner join")
    return games


def critic(games) -> None:
    print("critic: value_p2 vs game result (z = +-1, actor's view)")
    for name, sel in (("all decisions", None), ("turn starts", "start")):
        print(f"  {name}:")
        for b, lo, hi in BUCKETS:
            z, v = [], []
            for _, _, winner, dec in games:
                seen = set()
                for _k, actor, turn, val in dec:
                    if not lo <= turn <= hi:
                        continue
                    if sel == "start":
                        if (actor, turn) in seen:
                            continue
                        seen.add((actor, turn))
                    z.append(1.0 if actor == winner else -1.0)
                    v.append(val)
            z, v = np.asarray(z), np.asarray(v)
            if len(z) == 0:
                fail(f"empty bucket {b}")
            print(
                f"    turn {b:4s} n {len(z):7d}  EV {ev(z, v):+.3f}  corr {np.corrcoef(z, v)[0, 1]:.3f}  "
                f"sign acc {np.mean(np.sign(v) == z):.3f}  mean v {v.mean():+.3f} mean z {z.mean():+.3f}"
            )


def boot(games) -> None:
    print("boot: v at each actor's first decision of turn 2 (closes turn 1) minus z")
    err, by_deck = [], defaultdict(list)
    for _, decks, winner, dec in games:
        done = set()
        for _k, actor, turn, val in dec:
            if turn == 2 and actor not in done:
                done.add(actor)
                z = 1.0 if actor == winner else -1.0
                err.append(val - z)
                by_deck[decks[actor]].append((val, z))
    err = np.asarray(err)
    print(f"  n {len(err)}  mean (v - z) {err.mean():+.3f}  rms {np.sqrt((err**2).mean()):.3f}")
    bias, se = [], []
    for vz in by_deck.values():
        if len(vz) < 20:
            continue
        a = np.asarray(vz)
        bias.append(float((a[:, 0] - a[:, 1]).mean()))
        se.append(float((a[:, 0] - a[:, 1]).std(ddof=1) / np.sqrt(len(a))))
    bias, se = np.asarray(bias), np.asarray(se)
    # the spread of per-deck biases beyond noise: sqrt(max(0, var(bias) - mean(se^2)))
    extra = np.sqrt(max(0.0, float(bias.var() - (se**2).mean())))
    print(
        f"  decks with >= 20 games: {len(bias)}; per-deck mean (v - z): sd {bias.std():.3f}, noise-only sd "
        f"{np.sqrt((se**2).mean()):.3f}, systematic sd {extra:.3f} (win-chance pp {extra * 50:.1f}); "
        f"|bias| > 2 se: {np.mean(np.abs(bias) > 2 * se):.2f}"
    )
    worst = sorted(zip(bias, by_deck), key=lambda t: -abs(t[0]))[:5]
    print("  largest: " + ", ".join(f"{d} {b:+.2f}" for b, d in worst))


def deal(stem: Path) -> None:
    rows = [
        json.loads(x)
        for f in sorted(stem.parent.glob(f"{stem.name}-*.jsonl"))
        for x in f.read_text().splitlines()
        if x.strip()
    ]
    rows = [r for r in rows if "error" not in r]
    print(f"deal: M1 rows {len(rows)} (played option, 16 sampled results; v = mean value at end of turn)")
    for b, lo, hi in (("1", 1, 1), ("2+", 2, 10**6), ("all", 0, 10**6)):
        rs = [r for r in rows if lo <= r["turn"] <= hi]
        e = np.asarray([r["end"][0] for r in rs], float)
        v = np.asarray([np.mean(r["v1"][0]) for r in rs], float)
        tot = e.var()
        within = e.var(axis=1, ddof=1).mean()
        between = e.mean(1).var() - within / e.shape[1]
        icc = between / tot
        m = e.mean(1)
        beta = np.polyfit(v, m, 1)
        resid = m - np.polyval(beta, v)
        between_res = max(0.0, resid.var() - within / e.shape[1])
        print(
            f"  turn {b:3s} n {len(rs):3d}  ICC {icc:.2f}  r(v, mean) {np.corrcoef(v, m)[0, 1]:.2f}  "
            f"residual deal share {between_res / tot:.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-dir", type=Path, default=ROOT / "runs/score-20261016-1001")
    ap.add_argument("--src", type=Path, default=ROOT / "runs/value-p2-20261015-2044")
    ap.add_argument("--arms", nargs="+", default=["hppo-p", "hppo-p2"])
    ap.add_argument("--fork-rows", type=Path, default=ROOT / "runs/forkpilot-20261020-1913/rows")
    a = ap.parse_args()
    games = load(a.score_dir, a.src, a.arms)
    print(f"games {len(games)}")
    critic(games)
    boot(games)
    deal(a.fork_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
