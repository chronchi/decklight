"""Stage P3 verdict (pre-registered NOTEBOOK 2026-10-20 "Stage P3"; user: "improved play at non inferiority at least").

Reads a pulled stage-P3 run (scripts/vast_stage_p.sh with REF = P): `p` = P3 raw vs ygo-agent, `base` = P raw vs
ygo-agent on the same deals, `paired-base.txt` (paired_sm base -> p), `principles-{p,base}-s.json`, and the head-to-head
dirs `p-hold`, `p-new` (P3 = seat s_seat vs P).

  1. improved play: P3's own-turn hand-trap waste lower than P's by > 2 SE (ht_own_t1 rate per opportunity, or the
     share of games with an own-turn HT fire), and neither ht_opp nor interrupt_opp possibly more than 10% below P's rate (diff - 2 SE < -0.10 x P's
     rate; amended before any P3 data: the plain 2 SE test failed a +6 pp model on 6k games);
  2. non-inferiority: paired P3 - P lower 90% bound >= -2 pp, each seat >= -3 pp; pooled h2h lower 90% >= 0.48.
Pass = 1 and 2; "big move" = P3 - P >= +5 pp, reported separately.

    python scripts/audit/p3_gate.py runs/stage-p3-<stamp>
Exit 0 pass, 1 fail, 2 on missing/empty inputs.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

NI_REL = 0.10  # anti-passivity non-inferiority margin, relative to P's rate (amendment, NOTEBOOK 2026-10-20)


def fail(msg: str) -> None:
    print(f"ALERT p3_gate: {msg}", flush=True)
    raise SystemExit(2)


def rate_diff(a: dict, b: dict) -> tuple[float, float]:
    """(rate_b - rate_a, SE) for two independent fires/opps proportions."""
    pa, pb = a["fires"] / max(a["opps"], 1), b["fires"] / max(b["opps"], 1)
    se = math.sqrt(pa * (1 - pa) / max(a["opps"], 1) + pb * (1 - pb) / max(b["opps"], 1))
    return pb - pa, se


def h2h(dirs: list[Path]) -> tuple[int, int]:
    w = n = 0
    for d in dirs:
        for f in sorted(d.glob("games-*.jsonl")):
            for x in f.read_text().splitlines():
                if not x.strip():
                    continue
                r = json.loads(x)
                if r.get("truncated") or r.get("winner") not in (0, 1):
                    continue
                n += 1
                w += int(r["winner"] == r["s_seat"])
    return w, n


def main() -> int:
    run = Path(sys.argv[1])
    try:
        pr = {k: json.loads((run / f"principles-{k}-s.json").read_text()) for k in ("p", "base")}
        paired = (run / "paired-base.txt").read_text().splitlines()
    except FileNotFoundError as e:
        fail(f"missing input: {e.filename}")
    pa = {}
    for line in paired[1:]:
        m = re.match(r"(\w+): n=(\d+) B-A ([+-][\d.]+) pp .*\(SE ([\d.]+)\)", line)
        if m:
            pa[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
    if "all" not in pa:
        fail("no 'all' line in paired-base.txt")
    n_games = pa["all"][0]
    p3, p = pr["p"], pr["base"]
    print(f"run {run.name}: {n_games} paired deals vs ygo-agent")

    # 1. improved play
    print("1. play (P3 - P; waste lower is better, ht_opp / interrupts lower is worse)")
    d1, s1 = rate_diff(p["ht_own_t1"], p3["ht_own_t1"])
    ga, gb = p["lines_with_own_turn_ht"] / n_games, p3["lines_with_own_turn_ht"] / n_games
    d2, s2 = gb - ga, math.sqrt(ga * (1 - ga) / n_games + gb * (1 - gb) / n_games)
    better = (d1 < -2 * s1) or (d2 < -2 * s2)
    print(
        f"   ht_own_t1 rate      P {p['ht_own_t1']['rate']:.4f} P3 {p3['ht_own_t1']['rate']:.4f}  diff {d1:+.4f} (SE {s1:.4f})"
    )
    print(f"   own-turn HT games   P {ga:.4f} P3 {gb:.4f}  diff {d2:+.4f} (SE {s2:.4f})")
    worse = []
    for k in ("ht_opp", "interrupt_opp"):
        d, s = rate_diff(p[k], p3[k])
        print(f"   {k:19s} P {p[k]['rate']:.4f} P3 {p3[k]['rate']:.4f}  diff {d:+.4f} (SE {s:.4f})")
        if d - 2 * s < -NI_REL * p[k]["rate"]:  # could be more than 10% below P's rate (95% bound)
            worse.append(k)
    for k in ("ht_own_t2p", "empty_draw_own", "empty_draw_opp"):
        d, s = rate_diff(p[k], p3[k])
        print(f"   (report) {k:10s} P {p[k]['rate']:.4f} P3 {p3[k]['rate']:.4f}  diff {d:+.4f} (SE {s:.4f})")
    play_ok = better and not worse
    print(
        f"   -> improved play: {'PASS' if play_ok else 'FAIL'} (waste lower: {better}; anti-passivity worse: {worse})"
    )

    # 2. non-inferiority
    _, m_all, se_all = pa["all"]
    lo = m_all - 1.645 * se_all
    seats = {s: pa[s][1] for s in ("first", "second") if s in pa}
    w, n = h2h([run / "p-hold", run / "p-new"])
    if n == 0:
        fail("0 head-to-head games in p-hold / p-new")
    wr = w / n
    h_lo = wr - 1.645 * math.sqrt(wr * (1 - wr) / n)
    ni_ok = lo >= -2.0 and all(v >= -3.0 for v in seats.values()) and h_lo >= 0.48
    print(f"2. results: P3 - P {m_all:+.1f} pp (SE {se_all:.2f}, lower 90% {lo:+.1f}); seats {seats}")
    print(f"   h2h P3 vs P (holdout + new-card) {wr:.3f} n={n}, lower 90% {h_lo:.3f}")
    print(f"   -> non-inferiority: {'PASS' if ni_ok else 'FAIL'}")
    big = m_all >= 5.0
    verdict = play_ok and ni_ok
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'}; big move (>= +5 pp): {'yes' if big else 'no'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
