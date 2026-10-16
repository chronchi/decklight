"""Aggregate --prof 1 output into the cost table of plan 2026-10-16-speed-and-cheap-tests §1.2.

    python scripts/audit/prof_report.py DIR

Reads DIR/prof-*.json (per-proc forward counters from search_match --prof 1) and DIR/games-*.jsonl (search[].prof per
searched decision). Prints per-component seconds, mean ms per searched decision and shares of t_wall, plus searched
decisions per box-minute (box wall = the slowest proc's wall). H2D/D2H stay 0: BCPolicy.logits fuses the copies with
the forward, so they sit inside t_gpu. Exits 1 when no searched decision carries a prof dict.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

COMPS = ("t_det_init", "t_det", "t_setup", "t_env", "t_py", "t_wait", "t_value")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="search_match --out dir (prof-*.json + games-*.jsonl with search[].prof)")
    a = ap.parse_args(argv)
    profs = []
    for f in sorted(glob.glob(f"{a.dir}/prof-*.json")):
        with open(f) as fh:
            profs.append(json.load(fh))
    n_search, tot = 0, {c: 0.0 for c in COMPS}
    t_wall = 0.0
    n_prop = n_acc = rounds = obs_fwd = 0
    for f in sorted(glob.glob(f"{a.dir}/games-*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                for e in json.loads(line).get("search", []):
                    p = e.get("prof")
                    if not p:
                        continue
                    n_search += 1
                    for c in COMPS:
                        tot[c] += float(p.get(c, 0.0))
                    t_wall += float(p.get("t_wall", 0.0))
                    n_prop += int(p.get("n_prop", 0))
                    n_acc += int(p.get("n_acc", 0))
                    rounds += int(p.get("rounds", 0))
                    obs_fwd += int(p.get("obs_fwd", 0))
    if not n_search:
        print(f"ALERT prof_report: no search[].prof in {a.dir} (run search_match with --prof 1)")
        return 1
    known = sum(tot.values())
    unatt = t_wall - known
    print(f"prof over {n_search} searched decisions, {len(profs)} prof files:")
    print(f"  {'component':12} {'seconds':>10} {'ms/search':>10} {'share':>7}")
    for c in (*COMPS,):
        s = tot[c]
        print(f"  {c:12} {s:>10.1f} {1000 * s / n_search:>10.1f} {s / t_wall if t_wall else 0:>7.1%}")
    print(f"  {'unattributed':12} {unatt:>10.1f} {1000 * unatt / n_search:>10.1f} "
          f"{unatt / t_wall if t_wall else 0:>7.1%}")
    print(f"  {'t_wall':12} {t_wall:>10.1f} {1000 * t_wall / n_search:>10.1f} {'100.0%':>7}")
    print(f"  proposals {n_prop}, accepted {n_acc} ({n_acc / n_prop:.1%} of proposals), "
          f"{rounds} forward rounds, {obs_fwd} obs forwarded")
    box_wall = max((p.get("wall", 0.0) for p in profs), default=0.0)
    if box_wall > 0:
        print(f"  searched decisions per box-minute: {n_search / (box_wall / 60):.1f} "
              f"({n_search} searches / {box_wall:.0f} s slowest proc)")
    else:
        print("  searched decisions per box-minute: - (no prof wall clock)")
    if profs:
        calls = sum(p.get("calls", 0) for p in profs)
        bs = sum(sum(int(k) * v for k, v in p.get("hist", {}).items()) for p in profs)
        t_batch = sum(p.get("t_batch", 0.0) for p in profs)
        t_fwd = sum(p.get("t_fwd", 0.0) for p in profs)
        t_gpu = sum(p.get("t_gpu", 0.0) for p in profs)
        t_sync = sum(p.get("t_sync", 0.0) for p in profs)
        t_ygo = sum(p.get("t_ygo", 0.0) for p in profs)
        print(f"  forwards: {calls} calls, mean batch {bs / calls if calls else 0:.1f} obs; "
              f"batch {t_batch:.1f} s, fwd {t_fwd:.1f} s (gpu {t_gpu:.1f}, sync-wait {t_sync:.1f}), "
              f"ygo-agent {t_ygo:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
