"""Full-P3 card verdict (pre-registered NOTEBOOK 2026-10-21 "Full-P3 card"; user: "improved play at non inferiority").

Arms: stored Full-P = runs/value-p2-20261015-2044/hppo-p2; <run>/full_p3 (P3 + value_p2 + product search); <run>/full_p_rep
(Full-P on 248 of the deals, drift check). Inputs computed on the laptop first (engine replays, no model):

    R=runs/full-p3-<stamp>; S=runs/value-p2-20261015-2044/hppo-p2
    for x in "$R/full_p3 p3" "$S p"; do set -- $x
      taskset -c 0-3 uv run python scripts/audit/principles.py --name $2 --games $1 --seat s --out $R/principles-$2-s.json
      taskset -c 0-3 uv run python scripts/audit/maxx_habit.py --games $1 --seat s --name $2 --out $R/maxx-$2.json
    done
    uv run python scripts/audit/full_p3_gate.py $R

  1. drift: paired full_p_rep - stored, 95% CI contains 0 (else the stored pairing is invalid: verdict INVALID, exit 3);
  2. non-inferiority: paired full_p3 - stored lower 90% >= -2 pp, each seat >= -3 pp;
  3. improved play: own-turn Maxx games share or own-turn hand-trap games lower by > 2 SE; ht_opp / interrupt_opp not
     possibly > 10% below Full-P's rate (diff - 2 SE < -0.10 x rate).
Pass = 2 and 3. Exit 0 pass, 1 fail, 2 missing inputs, 3 drift.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_sm import games

ROOT = Path(__file__).resolve().parents[2]
STORED = ROOT / "runs/value-p2-20261015-2044/hppo-p2"
NI_REL = 0.10


def fail(msg: str) -> None:
    print(f"ALERT full_p3_gate: {msg}", flush=True)
    raise SystemExit(2)


def paired(a: dict, b: dict, seat: int | None = None) -> tuple[int, float, float]:
    """(n, mean B - A in pp, SE in pp) over deals present in both."""
    ks = [k for k in a if k in b and (seat is None or k[2] == seat)]
    if not ks:
        return 0, float("nan"), float("nan")
    d = [b[k] - a[k] for k in ks]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)
    return len(d), 100 * m, 100 * math.sqrt(var / len(d))


def prop_diff(pa: float, na: int, pb: float, nb: int) -> tuple[float, float]:
    return pb - pa, math.sqrt(pa * (1 - pa) / max(na, 1) + pb * (1 - pb) / max(nb, 1))


def main() -> int:
    run = Path(sys.argv[1])
    st, p3, rep = games(STORED), games(run / "full_p3"), games(run / "full_p_rep")
    if not st or not p3:
        fail(f"no games: stored {len(st)}, full_p3 {len(p3)}")
    try:
        pr = {k: json.loads((run / f"principles-{k}-s.json").read_text()) for k in ("p3", "p")}
        mx = {k: json.loads((run / f"maxx-{k}.json").read_text())["all"] for k in ("p3", "p")}
    except FileNotFoundError as e:
        fail(f"missing input: {e.filename} (see the docstring)")
    wr = {n: sum(g.values()) / len(g) for n, g in (("stored Full-P", st), ("full_p3", p3), ("full_p_rep", rep)) if g}
    print("win rates vs ygo-agent:", {k: f"{v:.3f} (n {len(g)})" for (k, v), g in zip(wr.items(), (st, p3, rep))})

    # 1. drift
    n, m, se = paired(st, rep)
    drift_ok = n > 0 and abs(m) <= 1.96 * se
    print(f"1. drift: full_p_rep - stored {m:+.1f} pp (SE {se:.2f}, n {n}) -> {'OK' if drift_ok else 'DRIFT'}")

    # 2. non-inferiority
    n, m, se = paired(st, p3)
    lo = m - 1.645 * se
    seats = {s: paired(st, p3, i)[1] for i, s in ((0, "first"), (1, "second"))}
    ni_ok = lo >= -2.0 and all(v >= -3.0 for v in seats.values())
    print(f"2. Full-P3 - Full-P {m:+.1f} pp (SE {se:.2f}, n {n}, lower 90% {lo:+.1f}); seats "
          f"{ {k: round(v, 1) for k, v in seats.items()} } -> {'PASS' if ni_ok else 'FAIL'}")
    if n < 0.9 * len(st):
        print(f"   note: {n} of {len(st)} stored deals paired (full_p3 capped?)")

    # 3. improved play
    d1, s1 = prop_diff(mx["p"]["share_games_fired"], mx["p"]["player_games"],
                       mx["p3"]["share_games_fired"], mx["p3"]["player_games"])
    ga, gb = pr["p"]["lines_with_own_turn_ht"] / len(st), pr["p3"]["lines_with_own_turn_ht"] / len(p3)
    d2, s2 = prop_diff(ga, len(st), gb, len(p3))
    better = d1 < -2 * s1 or d2 < -2 * s2
    print(f"3. own-turn Maxx games   P {mx['p']['share_games_fired']:.4f} P3 {mx['p3']['share_games_fired']:.4f}"
          f"  diff {d1:+.4f} (SE {s1:.4f})")
    print(f"   own-turn HT games     P {ga:.4f} P3 {gb:.4f}  diff {d2:+.4f} (SE {s2:.4f})")
    worse = []
    for k in ("ht_opp", "interrupt_opp", "ht_own_t1", "ht_own_t2p"):
        a, b = pr["p"][k], pr["p3"][k]
        d, s = prop_diff(a["rate"], a["opps"], b["rate"], b["opps"])
        print(f"   {k:20s} P {a['rate']:.4f} P3 {b['rate']:.4f}  diff {d:+.4f} (SE {s:.4f})")
        if k in ("ht_opp", "interrupt_opp") and d - 2 * s < -NI_REL * a["rate"]:
            worse.append(k)
    play_ok = better and not worse
    print(f"   -> improved play: {'PASS' if play_ok else 'FAIL'} (waste lower: {better}; anti-passivity worse: {worse})")
    if not drift_ok:
        print("VERDICT: INVALID (today's Full-P differs from the stored arm; rerun a fresh Full-P arm)")
        return 3
    verdict = ni_ok and play_ok
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
