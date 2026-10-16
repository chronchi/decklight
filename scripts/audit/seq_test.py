"""Sequential paired comparison of two search_match arms (Wald SPRT).

Pairs games exactly like ``paired_sm.py`` (key ``(tuple(decks), seed,
s_seat)``, score 1 win / 0 loss / 0.5 truncated of the scored side), then
walks a pre-registered SPRT over the paired diffs in a seeded random
schedule order, looking only after each complete block of deals. At
``max_deals`` the walk truncates and falls back to the fixed one-sided CI
rule. Always prints the fixed-sample paired diff with 95% CI overall and
per seat.

    python scripts/audit/seq_test.py A_DIR B_DIR --prereg docs/prereg/sprt-default.json
    python scripts/audit/seq_test.py --simulate --prereg docs/prereg/sprt-default.json

Exit codes: 0 decided (accept_H1 / accept_H0 / truncated_pass /
truncated_fail), 10 more deals needed (continue), 2 data error (< 90%
pairing, duplicate keys, empty).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ygo_zero.eval.seqtest import (
    ordered_keys,
    paired_ci,
    resolve_params,
    run_sequential,
    simulate,
)


def load_arm(d: Path) -> tuple[dict, int, int]:
    """Return (key -> score, n_games, n_duplicate_keys)."""
    out: dict = {}
    dupes = 0
    n = 0
    for f in sorted(d.glob("games-*.jsonl")):
        for raw in f.read_text().splitlines():
            if not raw.strip():
                continue
            g = json.loads(raw)
            n += 1
            key = (tuple(g["decks"]), g["seed"], g["s_seat"])
            if key in out:
                dupes += 1
            score = 0.5 if g.get("truncated") else float(g["winner"] == g["s_seat"])
            out[key] = score
    return out, n, dupes


def report(name: str, diffs: list[float]) -> str:
    n = len(diffs)
    if n < 2:
        return f"{name}: n={n}"
    m, lo, hi, se = paired_ci(diffs)
    up, down = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)
    return (
        f"{name}: n={n} B-A {m * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}] "
        f"(SE {se * 100:.2f}); B better {up}, A better {down}"
    )


def load_schedule(path: Path) -> list[tuple] | None:
    if path is None:
        return None
    sched = []
    for raw in path.read_text().splitlines():
        if raw.strip():
            decks, seed, seat = json.loads(raw)
            sched.append((tuple(decks), seed, seat))
    return sched


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a_dir", nargs="?", help="baseline arm dir (games-*.jsonl)")
    ap.add_argument("b_dir", nargs="?", help="variant arm dir (games-*.jsonl)")
    ap.add_argument("--prereg", required=True, help="pre-registered parameters JSON")
    ap.add_argument("--schedule", default=None, help="optional JSON-lines deal order")
    ap.add_argument("--baseline", action="store_true",
                    help="A_DIR is a cached baseline (may hold extra deals; pairing is unchanged)")
    ap.add_argument("--simulate", action="store_true",
                    help="Monte Carlo operating characteristics instead of a data run")
    ap.add_argument("--n-runs", type=int, default=20000)
    ap.add_argument("--sim-seed", type=int, default=0)
    args = ap.parse_args(argv)

    prereg = json.loads(Path(args.prereg).read_text())
    params = resolve_params(prereg)

    if args.simulate:
        table = simulate(
            params["p0"], params["p1"], params["alpha"], params["beta"],
            params["block"], params["max_deals"], d=params["d"],
            n_runs=args.n_runs, seed=args.sim_seed, margin=params["margin"],
        )
        print(f"mode={params['mode']} p0={params['p0']:.4f} p1={params['p1']:.4f} "
              f"alpha={params['alpha']} beta={params['beta']} block={params['block']} "
              f"max_deals={params['max_deals']} d={params['d']} margin_pp={params['margin_pp']} "
              f"runs={args.n_runs} seed={args.sim_seed}")
        print(f"{'delta_pp':>8} {'P(H1)':>7} {'ASN':>6} {'P(trunc)':>8}")
        for delta, row in table.items():
            print(f"{delta:+8.1f} {row['p_accept_h1']:7.3f} {row['asn']:6.0f} {row['p_trunc']:8.3f}")
        return 0

    if not args.a_dir or not args.b_dir:
        ap.error("A_DIR and B_DIR are required unless --simulate is given")
    a, na, dup_a = load_arm(Path(args.a_dir))
    b, nb, dup_b = load_arm(Path(args.b_dir))
    if na == 0 or nb == 0:
        print(f"ALERT seq_test: empty arm (A games={na}, B games={nb})")
        return 2
    if dup_a or dup_b:
        print(f"ALERT seq_test: duplicate pair keys (A dupes={dup_a}, B dupes={dup_b})")
        return 2
    keys = sorted(set(a) & set(b))
    if len(keys) < 0.9 * len(a):
        print(f"ALERT seq_test: only {len(keys)} of {len(a)} games pair up")
        return 2

    sched = load_schedule(Path(args.schedule)) if args.schedule else None
    if sched is not None:
        pos = {k: i for i, k in enumerate(sched)}
        head = sorted((k for k in keys if k in pos), key=lambda k: pos[k])
        tail = ordered_keys([k for k in keys if k not in pos], params["order_seed"])
        keys = head + tail
    else:
        keys = ordered_keys(keys, params["order_seed"])

    diffs_ordered = [b[k] - a[k] for k in keys]
    res = run_sequential(
        diffs_ordered, params["p0"], params["p1"], params["alpha"],
        params["beta"], params["block"], params["max_deals"], margin=params["margin"],
    )
    print(f"A={args.a_dir} ({sum(a.values()) / len(a):.3f}, n={len(a)})  "
          f"B={args.b_dir} ({sum(b.values()) / len(b):.3f}, n={len(b)})")
    print(f"prereg: mode={params['mode']} p0={params['p0']:.4f} p1={params['p1']:.4f} "
          f"alpha={params['alpha']} beta={params['beta']} block={params['block']} "
          f"max_deals={params['max_deals']} order_seed={params['order_seed']}")
    print(f"{'deals':>6} {'discord':>8} {'B-wins':>7} {'LLR':>7} "
          f"{'lower':>7} {'upper':>7} decision")
    for r in res["rows"]:
        print(f"{r['deals']:6d} {r['discord']:8.1f} {r['b_wins']:7.1f} {r['llr']:7.3f} "
              f"{r['lower']:7.3f} {r['upper']:7.3f} {r['decision']}")
    print(f"decision: {res['decision']} (deals_used={res['deals_used']}, "
          f"available={res['deals_available']})")
    print(report("all", diffs_ordered))
    for seat, label in ((0, "first"), (1, "second")):
        print(report(label, [b[k] - a[k] for k in keys if k[2] == seat]))
    return 0 if res["decision"] != "continue" else 10


if __name__ == "__main__":
    sys.exit(main())
