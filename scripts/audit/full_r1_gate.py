"""Full-r1 product card verdict (pre-registered NOTEBOOK 2026-10-30 "Full-r1 product card"; user: "Can you test r1 to
become the product?"; standing rule for product changes: "improved play at non inferiority at least").

Arms (scripts/vast_full_r1.sh): <run>/full_r1, <run>/fz_r1, <run>/fz_p, <run>/full_p_rep; stored product arm
runs/full-p-ho-20261021-1204/full_p_ho (Full-P + ho, 0.623, n 1,488). Play inputs computed on the laptop first
(engine replays, no model):

    R=runs/full-r1-<stamp>; S=runs/full-p-ho-20261021-1204/full_p_ho
    for x in "$R/full_r1 r1" "$S p"; do set -- $x
      taskset -c 0-3 uv run python scripts/audit/principles.py --name $2 --games $1 --seat s --out $R/principles-$2-s.json
      taskset -c 0-3 uv run python scripts/audit/maxx_habit.py --games $1 --seat s --name $2 --out $R/maxx-$2.json
    done
    uv run python scripts/audit/full_r1_gate.py $R

  1. drift: paired full_p_rep - stored, 95% CI contains 0 (else INVALID, exit 3: rerun a fresh Full-P arm);
  2. non-inferiority (gauntlet): paired full_r1 - stored lower 90% >= -2 pp, each seat >= -3 pp;
  3. improvement (Forge archetypes): paired fz_r1 - fz_p, pooled over the 11 archetypes, lower 95% > 0;
  4. no new waste: own-turn Maxx games share and own-turn hand-trap games share not higher than Full-P's by > 2 SE.
Pass = 2, 3 and 4 (with 1 OK). Exit 0 pass, 1 fail, 2 missing inputs, 3 drift. Per-archetype rows are reported only.

Amendment (NOTEBOOK 2026-10-31 "fresh Full-P arm", before its data): the first run was INVALID by 1. Rerun with
`--stored runs/full-p-fresh-<stamp>/full_p` (today's Full-P on all 1,488 deals) and `principles-p-s.json` /
`maxx-p.json` recomputed from it; 1 then checks full_p_rep against the fresh arm (same code, both today).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_p3_gate import paired, prop_diff
from paired_sm import games

ROOT = Path(__file__).resolve().parents[2]
STORED = ROOT / "runs/full-p-ho-20261021-1204/full_p_ho"


def fail(msg: str) -> None:
    print(f"ALERT full_r1_gate: {msg}", flush=True)
    raise SystemExit(2)


def archetype_of(deck: str, names: list[str]) -> str:
    return next((a for a in names if deck.startswith(a + "-")), "?")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--stored", type=Path, default=STORED,
                    help="reference Full-P arm (amendment 2026-10-31: the fresh full-p-fresh-* arm)")
    a = ap.parse_args()
    run = a.run
    st, r1, rep = games(a.stored), games(run / "full_r1"), games(run / "full_p_rep")
    fr, fp = games(run / "fz_r1"), games(run / "fz_p")
    for label, g in (("stored", st), ("full_r1", r1), ("full_p_rep", rep), ("fz_r1", fr), ("fz_p", fp)):
        if not g:
            fail(f"{label} has no games")
    try:
        pr = {k: json.loads((run / f"principles-{k}-s.json").read_text()) for k in ("r1", "p")}
        mx = {k: json.loads((run / f"maxx-{k}.json").read_text())["all"] for k in ("r1", "p")}
    except FileNotFoundError as e:
        fail(f"missing input: {e.filename} (see the docstring)")
    res: dict = {}
    wr = {k: round(sum(g.values()) / len(g), 4) for k, g in
          (("stored Full-P", st), ("full_r1", r1), ("full_p_rep", rep), ("fz_r1", fr), ("fz_p", fp))}
    res["win_rates"] = wr
    print("win rates:", {k: f"{v:.3f} (n {n})" for (k, v), n in
                         zip(wr.items(), (len(st), len(r1), len(rep), len(fr), len(fp)))})

    n, m, se = paired(st, rep)
    drift_ok = n > 0 and abs(m) <= 1.96 * se
    res["drift"] = {"n": n, "pp": m, "se": se, "ok": drift_ok}
    print(f"1. drift: full_p_rep - stored {m:+.1f} pp (SE {se:.2f}, n {n}) -> {'OK' if drift_ok else 'DRIFT'}")

    n, m, se = paired(st, r1)
    lo = m - 1.645 * se
    seats = {s: paired(st, r1, i)[1] for i, s in ((0, "first"), (1, "second"))}
    ni_ok = lo >= -2.0 and all(v >= -3.0 for v in seats.values())
    res["ni"] = {"n": n, "pp": m, "se": se, "lower90": lo, "seats": seats, "ok": ni_ok}
    print(f"2. full_r1 - Full-P {m:+.1f} pp (SE {se:.2f}, n {n}, lower 90% {lo:+.1f}); seats "
          f"{ {k: round(v, 1) for k, v in seats.items()} } -> {'PASS' if ni_ok else 'FAIL'}")
    if n < 0.9 * len(st):
        print(f"   note: {n} of {len(st)} stored deals paired (full_r1 capped?)")

    n, m, se = paired(fp, fr)
    imp_ok = n > 0 and m - 1.96 * se > 0
    res["forge"] = {"n": n, "pp": m, "se": se, "ok": imp_ok, "per_archetype": {}}
    print(f"3. Forge archetypes fz_r1 - fz_p {m:+.1f} pp [{m - 1.96 * se:+.1f}, {m + 1.96 * se:+.1f}] (n {n}) -> "
          f"{'PASS' if imp_ok else 'FAIL'}")
    archs = sorted(p.stem for p in (ROOT / "data/forge/f3/pairs").glob("*.pairs"))
    for ar in archs:
        sub_r = {k: v for k, v in fr.items() if archetype_of(k[0][k[2]], archs) == ar}
        sub_p = {k: v for k, v in fp.items() if archetype_of(k[0][k[2]], archs) == ar}
        n2, m2, se2 = paired(sub_p, sub_r)
        res["forge"]["per_archetype"][ar] = {"n": n2, "pp": m2, "se": se2}
        print(f"   {ar:30s} {m2:+6.1f} pp (SE {se2:.1f}, n {n2})")

    d1, s1 = prop_diff(mx["p"]["share_games_fired"], mx["p"]["player_games"],
                       mx["r1"]["share_games_fired"], mx["r1"]["player_games"])
    ga, gb = pr["p"]["lines_with_own_turn_ht"] / len(st), pr["r1"]["lines_with_own_turn_ht"] / len(r1)
    d2, s2 = prop_diff(ga, len(st), gb, len(r1))
    waste_ok = not (d1 > 2 * s1 or d2 > 2 * s2)
    res["waste"] = {"maxx": [d1, s1], "own_ht": [d2, s2], "ok": waste_ok}
    print(f"4. own-turn Maxx games  P {mx['p']['share_games_fired']:.4f} r1 {mx['r1']['share_games_fired']:.4f}"
          f"  diff {d1:+.4f} (SE {s1:.4f})")
    print(f"   own-turn HT games    P {ga:.4f} r1 {gb:.4f}  diff {d2:+.4f} (SE {s2:.4f}) -> "
          f"{'PASS' if waste_ok else 'FAIL'}")
    for k in ("ht_opp", "interrupt_opp", "ht_own_t1", "ht_own_t2p"):
        x, y = pr["p"][k], pr["r1"][k]
        d, s = prop_diff(x["rate"], x["opps"], y["rate"], y["opps"])
        print(f"   {k:20s} P {x['rate']:.4f} r1 {y['rate']:.4f}  diff {d:+.4f} (SE {s:.4f})  (report only)")
    if not drift_ok:
        verdict, rc = "INVALID (today's Full-P differs from the stored arm; rerun a fresh Full-P arm)", 3
    else:
        ok = ni_ok and imp_ok and waste_ok
        verdict, rc = ("PASS" if ok else "FAIL"), (0 if ok else 1)
    res["verdict"] = verdict
    print(f"VERDICT: {verdict}")
    (a.json or run / "full_r1_gate.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
