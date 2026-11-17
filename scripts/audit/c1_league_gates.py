#!/usr/bin/env python3
"""Cycle-1 league segment gates (docs/lab/prereg/2026-11-10-master-cycle1.md, amendments 6-7). Engine-free.

    python3 scripts/audit/c1_league_gates.py failfast runs/.../train/Y/metrics.jsonl
    python3 scripts/audit/c1_league_gates.py gates --eval eval.json --card release_card.json --mirror card/M-mirror \
        --out gates.json

failfast (amendment 7): the tsoj win rate of the training episodes (sampled policy, train book), pooled over
iterations 31-40, against iterations 1-10 (X2's start, measured the same way). Exit 1 (stop) when the upper 95% bound
of the difference is below 0; exit 2 when either window has no tsoj games.

gates (amendment 6), each PASS/FAIL, then SEGMENT PASS|FAIL:
- tsoj: Y - X paired on the 1,568 held-out games (eval.json "paired"."Y" with --paired X) >= +3 pp, lower 95% > 0;
- no-loss: Y vs X raw mirror games (card dir M-mirror; Y is the `s_seat` side), Wilson lower 95% >= 0.47;
- retention: the card's pooled N - R over the accepted archetypes (pooled_new.diff_pp) >= +6.0 pp.
Reported only: the card's old / reserve / gauntlet rows, and early deaths in the card's new-archetype games (going
first: lost by turn 2; going second: lost by turn 3; amendment 4) for R (X) and N (Y).
Exit 2 when an input is missing or empty.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def failfast(path: Path) -> int:
    rows = [r for r in (json.loads(x) for x in path.open() if '"samples"' in x) if r.get("iter", 0) > 0]
    def pool(lo: int, hi: int) -> tuple[float, int]:
        sel = [r for r in rows if lo <= r["iter"] <= hi and r.get("ext_games", 0) > 0]
        n = sum(r["ext_games"] for r in sel)
        return (sum(r["ext_wr"] * r["ext_games"] for r in sel) / n if n else float("nan")), n
    (p0, n0), (p1, n1) = pool(1, 10), pool(31, 40)
    if not n0 or not n1:
        print(f"ALERT failfast: tsoj games it1-10 {n0}, it31-40 {n1}", flush=True)
        return 2
    d = p1 - p0
    se = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    hi = d + 1.96 * se
    verdict = "STOP" if hi < 0 else "GO"
    print(f"tsoj train win rate it1-10 {p0:.3f} (n {n0}), it31-40 {p1:.3f} (n {n1}): diff {100 * d:+.2f} pp, "
          f"upper 95% {100 * hi:+.2f} pp -> {verdict}", flush=True)  # fmt: skip
    return 1 if verdict == "STOP" else 0


def games(d: Path) -> list[dict]:
    return [json.loads(x) for f in sorted(d.glob("games-*.jsonl")) for x in f.open() if x.strip()]


def early(card: Path, tag: str) -> dict:
    """Early deaths of the S side over the card's new-archetype rows (R-* or N-* for archetypes in `new`)."""
    out = {"first_n": 0, "first_dead_t2": 0, "second_n": 0, "second_dead_t3": 0}
    for g in games(card):
        lost = g["winner"] != g["s_seat"] and not g.get("truncated")
        if g["s_seat"] == 0:  # seat 0 goes first
            out["first_n"] += 1
            out["first_dead_t2"] += int(lost and g["turns"] <= 2)
        else:
            out["second_n"] += 1
            out["second_dead_t3"] += int(lost and g["turns"] <= 3)
    return out


def gates(a: argparse.Namespace) -> int:
    for p in (a.eval, a.card):
        if not p.is_file():
            print(f"ALERT gates: missing {p}", flush=True)
            return 2
    ev, card = json.loads(a.eval.read_text()), json.loads(a.card.read_text())
    res: dict = {}
    pr = ev.get("paired", {}).get("Y")
    if pr is None:
        print("ALERT gates: eval.json has no paired Y (run gauntlet_summary with --paired X)", flush=True)
        return 2
    res["tsoj"] = {"diff_pp": round(100 * pr["diff"], 2), "ci_pp": [round(100 * x, 2) for x in pr["ci95"]],
                   "pairs": pr["pairs"], "Y": ev["Y"]["p_win"], "X": ev["X"]["p_win"]}  # fmt: skip
    res["tsoj"]["PASS"] = pr["diff"] >= 0.03 and pr["ci95"][0] > 0
    mg = games(a.mirror) if a.mirror.is_dir() else []
    if not mg:
        print(f"ALERT gates: no mirror games in {a.mirror}", flush=True)
        return 2
    k = sum(g["winner"] == g["s_seat"] for g in mg)
    lo, hi = wilson(k, len(mg))
    res["no_loss"] = {"n": len(mg), "Y_wr": round(k / len(mg), 4), "wilson": [round(lo, 4), round(hi, 4)], "PASS": lo >= 0.47}
    pn = card["pooled_new"]
    res["retention"] = {"pooled_new": pn, "PASS": pn["diff_pp"] >= 6.0}
    res["reported"] = {"old": card["pooled_old"], "reserve": card["reserve"]["stats"], "gauntlet": card["gauntlet"]["stats"],
                       "card_gates": {k: card[k] for k in ("GATE_kept", "GATE_old", "GATE_reserve", "GATE_gauntlet")}}
    cdir = a.card.parent / "card"
    for side in ("R", "N"):
        tot: dict = {}
        for arch in card["new"]:
            for k2, v in early(cdir / f"{side}-{arch}", side).items():
                tot[k2] = tot.get(k2, 0) + v
        if tot.get("first_n") and tot.get("second_n"):
            tot["first_dead_t2_rate"] = round(tot["first_dead_t2"] / tot["first_n"], 4)
            tot["second_dead_t3_rate"] = round(tot["second_dead_t3"] / tot["second_n"], 4)
        res["reported"][f"early_deaths_{'X' if side == 'R' else 'Y'}"] = tot
    res["SEGMENT_PASS"] = all(res[g]["PASS"] for g in ("tsoj", "no_loss", "retention"))
    for g in ("tsoj", "no_loss", "retention"):
        print(f"GATE {g} {'PASS' if res[g]['PASS'] else 'FAIL'}: {json.dumps({k: v for k, v in res[g].items() if k != 'PASS'})}")
    for k, v in res["reported"].items():
        print(f"reported {k}: {json.dumps(v)}")
    print(f"SEGMENT {'PASS' if res['SEGMENT_PASS'] else 'FAIL'}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("failfast")
    f.add_argument("metrics", type=Path)
    g = sub.add_parser("gates")
    g.add_argument("--eval", type=Path, required=True)
    g.add_argument("--card", type=Path, required=True)
    g.add_argument("--mirror", type=Path, required=True)
    g.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    return failfast(a.metrics) if a.cmd == "failfast" else gates(a)


if __name__ == "__main__":
    sys.exit(main())
