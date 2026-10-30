"""Forge round release card (docs/plans/2026-10-29-deck-forge.md, Pipeline step 5).

N (the distilled release candidate) minus R (the previous release) on the same deals
(raw, opponent R raw, both seats, paired by decks, seed, seat): each accepted pack's
verify pairs (R-<a> vs N-<a>), each old X4b archetype's pairs (R-<a> vs N-<a>), the
reserve set (R-res vs N-res), and the ygo-agent gauntlet (R-ga vs N-ga). Pre-registered
gates (do not change):

- GATE kept: per new archetype N - R (stats) and kept share = (N - R) / verify mean_pp
  (S - R); flag KEPT<50% below 0.5. PASS when >= 1 new archetype keeps >= 50% and the
  pooled (all new rows together) kept share >= 0.5 (pooled kept = pooled N - R mean /
  mean of the verify gains weighted by each archetype's card n).
- GATE old: pooled N - R over all old archetypes' rows; PASS when the lower 95% bound > -1.5 pp.
- GATE reserve: N-res - R-res; PASS when the lower bound > -1.5 pp.
- GATE gauntlet: N-ga - R-ga; PASS when the lower bound > -2.0 pp.
- RELEASE <name> yes|no: yes iff all four PASS.

Writes <card dir>/../release_card.json; prints a table and the GATE/RELEASE lines.
A missing arm -> ALERT, and the gate that needs it FAILs (never silently passes).
Exit 2 when R-res, R-ga or every new arm is missing. Engine-free.

    python scripts/audit/forge_release_card.py <card dir> --verify verify.json
        --new <archs> --old <archs> [--name r1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x4b_spec_card import games, paired, stats


def load_pair(card: Path, r_tag: str, n_tag: str, label: str) -> tuple[dict | None, list[float] | None]:
    """Paired N - R diffs for one arm pair; None pair on any problem (ALERT printed)."""
    r_dir, n_dir = card / r_tag, card / n_tag
    if not r_dir.is_dir() or not n_dir.is_dir():
        print(f"ALERT forge_release_card: {label}: arm {r_tag if not r_dir.is_dir() else n_tag} missing")
        return None, None
    r, n = games(r_dir), games(n_dir)
    if not r or not n:
        print(f"ALERT forge_release_card: {label}: arm {r_tag if not r else n_tag} has no games")
        return None, None
    try:
        keys, diffs = paired(r, n, label)
    except SystemExit as e:
        print(str(e))
        return None, None
    wr = round(sum(r[k][0] for k in keys) / len(keys), 4)
    return {"R_wr": wr, "keys": keys, "diffs": diffs, "stats": stats(diffs)}, diffs


def pooled_stats(all_diffs: list[list[float]]) -> dict | None:
    flat = [x for d in all_diffs for x in d]
    return stats(flat) if flat else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("card", type=Path)
    ap.add_argument("--verify", type=Path, required=True)
    ap.add_argument("--new", nargs="*", default=[])
    ap.add_argument("--old", nargs="*", default=[])
    ap.add_argument("--name", default="r1")
    a = ap.parse_args()
    verify = json.loads(a.verify.read_text())

    res: dict = {"name": a.name, "new": {}, "missing": []}
    new_diffs: list[list[float]] = []
    verify_weighted = 0.0
    verify_n = 0
    n_missing_new = 0
    for arch in a.new:
        pair, diffs = load_pair(a.card, f"R-{arch}", f"N-{arch}", arch)
        if pair is None:
            n_missing_new += 1
            res["missing"].append(arch)
            continue
        vg = verify.get(arch, {}).get("mean_pp")
        kept = None if not vg else round(pair["stats"]["diff_pp"] / vg, 3)
        pair["verify_S_minus_R_pp"] = vg
        pair["kept"] = kept
        pair["flag"] = "" if (kept is not None and kept >= 0.5) else "KEPT<50%"
        res["new"][arch] = {k: pair[k] for k in ("R_wr", "stats", "verify_S_minus_R_pp", "kept", "flag")}
        new_diffs.append(diffs or [])
        n = pair["stats"]["n"]
        if vg:
            verify_weighted += vg * n
            verify_n += n
    n_new_ok = sum(1 for v in res["new"].values() if v["kept"] is not None and v["kept"] >= 0.5)
    pooled_new = pooled_stats(new_diffs)
    pooled_kept = None
    if pooled_new and verify_n:
        pooled_kept = round(pooled_new["diff_pp"] / (verify_weighted / verify_n), 3)
    kept_pass = n_new_ok >= 1 and pooled_kept is not None and pooled_kept >= 0.5
    res["pooled_new"] = pooled_new
    res["pooled_kept"] = pooled_kept
    print("new archetype | R wr | N-R [95% CI] | verify S-R | kept | flag")
    for arch, v in res["new"].items():
        s = v["stats"]
        print(f"  {arch}: R {v['R_wr']:.3f} N-R {s['diff_pp']:+.2f} [{s['ci_pp'][0]:+.2f}, {s['ci_pp'][1]:+.2f}] "
              f"(n={s['n']}) verify {v['verify_S_minus_R_pp']:+.2f} kept {v['kept']} {v['flag']}")
    if pooled_new:
        print(f"  pooled N-R {pooled_new['diff_pp']:+.2f} (n={pooled_new['n']}) pooled kept {pooled_kept}")
    else:
        print("  pooled N-R missing (no new arms)")
    print(f"GATE kept {'PASS' if kept_pass else 'FAIL'}: {n_new_ok} of {len(res['new'])} new archetypes keep >= 50% "
          f"(need >= 1) and pooled kept {pooled_kept} (need >= 0.5)")
    res["GATE_kept"] = kept_pass

    old_diffs: list[list[float]] = []
    for arch in a.old:
        pair, diffs = load_pair(a.card, f"R-{arch}", f"N-{arch}", f"old {arch}")
        if pair is None:
            res["missing"].append(f"old-{arch}")
            continue
        res.setdefault("old", {})[arch] = {"R_wr": pair["R_wr"], "stats": pair["stats"]}
        old_diffs.append(diffs or [])
    pooled_old = pooled_stats(old_diffs)
    old_pass = pooled_old is not None and pooled_old["ci_pp"][0] > -1.5
    res["pooled_old"] = pooled_old
    if pooled_old:
        print(f"GATE old {'PASS' if old_pass else 'FAIL'}: pooled old N-R {pooled_old['diff_pp']:+.2f} "
              f"[{pooled_old['ci_pp'][0]:+.2f}, {pooled_old['ci_pp'][1]:+.2f}] (n={pooled_old['n']}, need lower > -1.5)")
    else:
        print("GATE old FAIL: no old archetype arms")
    res["GATE_old"] = old_pass

    res_pair, _ = load_pair(a.card, "R-res", "N-res", "reserve")
    res["reserve"] = None if res_pair is None else {"R_wr": res_pair["R_wr"], "stats": res_pair["stats"]}
    if res_pair is None:
        res["missing"].append("reserve")
    reserve_pass = res["reserve"] is not None and res["reserve"]["stats"]["ci_pp"][0] > -1.5
    if res["reserve"]:
        s = res["reserve"]["stats"]
        print(f"GATE reserve {'PASS' if reserve_pass else 'FAIL'}: N-res - R-res {s['diff_pp']:+.2f} "
              f"[{s['ci_pp'][0]:+.2f}, {s['ci_pp'][1]:+.2f}] (n={s['n']}, need lower > -1.5)")
    else:
        print("GATE reserve FAIL: reserve arms missing")
    res["GATE_reserve"] = reserve_pass

    ga_pair, _ = load_pair(a.card, "R-ga", "N-ga", "gauntlet")
    res["gauntlet"] = None if ga_pair is None else {"R_wr": ga_pair["R_wr"], "stats": ga_pair["stats"]}
    if ga_pair is None:
        res["missing"].append("gauntlet")
    ga_pass = res["gauntlet"] is not None and res["gauntlet"]["stats"]["ci_pp"][0] > -2.0
    if res["gauntlet"]:
        s = res["gauntlet"]["stats"]
        print(f"GATE gauntlet {'PASS' if ga_pass else 'FAIL'}: N-ga - R-ga {s['diff_pp']:+.2f} "
              f"[{s['ci_pp'][0]:+.2f}, {s['ci_pp'][1]:+.2f}] (n={s['n']}, need lower > -2.0)")
    else:
        print("GATE gauntlet FAIL: gauntlet arms missing")
    res["GATE_gauntlet"] = ga_pass

    release = kept_pass and old_pass and reserve_pass and ga_pass
    res["release"] = release
    print(f"RELEASE {a.name} {'yes' if release else 'no'}")
    (a.card.parent / "release_card.json").write_text(json.dumps(res, indent=1) + "\n")

    r_res_missing = (a.card / "R-res").is_dir() is False or not games(a.card / "R-res")
    r_ga_missing = (a.card / "R-ga").is_dir() is False or not games(a.card / "R-ga")
    if (a.new and n_missing_new == len(a.new)) or r_res_missing or r_ga_missing:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
