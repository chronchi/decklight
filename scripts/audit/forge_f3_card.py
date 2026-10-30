"""Forge F3 card: cheaper distillation arms against r0 (R) and F2's r1 (N), paired by deal.

Pre-registered (NOTEBOOK 2026-10-30 "Forge F3"; do not change). For each arm X (T = ~560 teacher games per pack,
D = DAgger: r0 pilots the pack decks on the same deals, teacher labels):

- new: per new archetype X - R and X - N; pooled X - N over all new rows. GATE new PASS when the lower 95% bound of
  the pooled X - N > -2.0 pp.
- reserve: X-res - R-res. GATE reserve PASS when the lower bound > -1.5 pp.
- old: pooled X - R over the 5 old archetypes (the confirmation card's deals). GATE old PASS when the lower bound > -1.5 pp.
- ADEQUATE X yes|no: all three PASS.
- D vs T (both present): pooled new D - T. SHIFT matters when its lower bound > +1.0 pp.

R/N new and reserve arms come from F2's card dir, R/N old arms from the confirmation card dir, X arms from the F3
card dir. A missing arm -> ALERT and the gate FAILs. Exit 2 when an X arm has no new archetype at all. Engine-free.

    python scripts/audit/forge_f3_card.py --f2 runs/forge-round-20261029-2133/card \
        --old-card runs/forge-oldcard-20261030-0610/card --f3 runs/<f3>/card --arms T D --new <archs> --old <archs>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x4b_spec_card import games, paired, stats


def arm(d: Path, label: str) -> dict | None:
    if not d.is_dir():
        print(f"ALERT forge_f3_card: {label}: {d} missing")
        return None
    g = games(d)
    if not g:
        print(f"ALERT forge_f3_card: {label}: {d} has no games")
        return None
    return g


def diff(a: dict | None, b: dict | None, label: str) -> list[float] | None:
    """b - a on the shared deals; None on any problem (ALERT printed)."""
    if a is None or b is None:
        return None
    try:
        return paired(a, b, label)[1]
    except SystemExit as e:
        print(str(e))
        return None


def fmt(s: dict | None) -> str:
    if not s or s.get("diff_pp") is None:
        return "n/a"
    return f"{s['diff_pp']:+.2f} [{s['ci_pp'][0]:+.2f}, {s['ci_pp'][1]:+.2f}] (n={s['n']})"


def pp(s: dict | None) -> str:
    return f"{s['diff_pp']:+8.2f}" if s and s.get("diff_pp") is not None else "     n/a"


def gate(s: dict | None, floor: float) -> bool:
    return bool(s and s.get("ci_pp") and s["ci_pp"][0] > floor)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f2", type=Path, required=True)
    ap.add_argument("--old-card", type=Path, required=True)
    ap.add_argument("--f3", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["T", "D"])
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    R = {x: arm(a.f2 / f"R-{x}", f"R-{x}") for x in [*a.new, "res"]}
    N = {x: arm(a.f2 / f"N-{x}", f"N-{x}") for x in a.new}
    R_old = {x: arm(a.old_card / f"R-{x}", f"old R-{x}") for x in a.old}
    res: dict = {"arms": {}}
    X_new: dict[str, dict] = {}
    rc = 0
    for x in a.arms:
        r: dict = {"new": {}, "old": {}}
        xn = {s: arm(a.f3 / f"{x}-{s}", f"{x}-{s}") for s in a.new}
        X_new[x] = xn
        if all(v is None for v in xn.values()):
            print(f"ALERT forge_f3_card: arm {x} has no new archetype")
            rc = 2
        print(f"== arm {x}")
        print(f"{'archetype':30s} {'R wr':>6s} {'N-R':>8s} {'X-R':>8s} {'X-N':>32s}")
        pool_xn: list[float] = []
        pool_xr: list[float] = []
        pool_nr: list[float] = []
        for s in a.new:
            nr, xr, xnd = diff(R[s], N[s], f"{s} N-R"), diff(R[s], xn[s], f"{s} {x}-R"), diff(N[s], xn[s], f"{s} {x}-N")
            row = {"N-R": stats(nr) if nr else None, "X-R": stats(xr) if xr else None,
                   "X-N": stats(xnd) if xnd else None}
            r["new"][s] = row
            if nr and xr and xnd:
                pool_nr += nr; pool_xr += xr; pool_xn += xnd
            wr = sum(v[0] for v in R[s].values()) / len(R[s]) if R[s] else float("nan")
            print(f"{s:30s} {wr:6.3f} {pp(row['N-R'])} {pp(row['X-R'])} {fmt(row['X-N']):>32s}")
        sn, sx, sxn = stats(pool_nr), stats(pool_xr), stats(pool_xn)
        kept = (sx["diff_pp"] / sn["diff_pp"]) if sn.get("diff_pp") else None
        r["new_pooled"] = {"N-R": sn, "X-R": sx, "X-N": sxn, "kept_of_r1": round(kept, 3) if kept is not None else None}
        print(f"pooled new: N-R {fmt(sn)}  X-R {fmt(sx)}  X-N {fmt(sxn)}  kept of r1's gain "
              f"{kept if kept is None else round(kept, 3)}")
        missing_new = [s for s in a.new if r["new"][s]["X-N"] is None]
        g_new = gate(sxn, -2.0) and not missing_new
        print(f"GATE {x} new {'PASS' if g_new else 'FAIL'}: pooled {x} - r1 {fmt(sxn)} (need lower > -2.0)"
              + (f"; missing {missing_new}" if missing_new else ""))
        xres = arm(a.f3 / f"{x}-res", f"{x}-res")
        dres = diff(R["res"], xres, f"res {x}-R")
        sres = stats(dres) if dres else None
        g_res = gate(sres, -1.5)
        print(f"GATE {x} reserve {'PASS' if g_res else 'FAIL'}: {x}-res - R-res {fmt(sres)} (need lower > -1.5)")
        pool_old: list[float] = []
        miss_old = []
        for s in a.old:
            d = diff(R_old[s], arm(a.f3 / f"{x}-{s}", f"{x}-{s}"), f"old {s} {x}-R")
            r["old"][s] = stats(d) if d else None
            if d:
                pool_old += d
            else:
                miss_old.append(s)
            print(f"  old {s:30s} {fmt(r['old'][s])}")
        sold = stats(pool_old)
        g_old = gate(sold, -1.5) and not miss_old
        print(f"GATE {x} old {'PASS' if g_old else 'FAIL'}: pooled {x} - R {fmt(sold)} (need lower > -1.5)"
              + (f"; missing {miss_old}" if miss_old else ""))
        r.update({"reserve": sres, "old_pooled": sold,
                  "gates": {"new": g_new, "reserve": g_res, "old": g_old}, "adequate": g_new and g_res and g_old})
        print(f"ADEQUATE {x} {'yes' if r['adequate'] else 'no'}")
        res["arms"][x] = r
    if "T" in X_new and "D" in X_new:
        pool: list[float] = []
        for s in a.new:
            d = diff(X_new["T"][s], X_new["D"][s], f"{s} D-T")
            if d:
                pool += d
        sdt = stats(pool)
        res["D-T"] = sdt
        shift = gate(sdt, 1.0)
        print(f"SHIFT {'matters' if shift else 'not shown'}: pooled new D - T {fmt(sdt)} (matters when lower > +1.0)")
    out = a.json or a.f3.parent / "f3_card.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
