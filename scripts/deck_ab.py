"""Paired deck A/B harness: is deck variant B better than variant A under the same pilot?

Both variants play the same opponents on the same duel seeds in both seats, so each
(opponent deck, seed, seat) gives a paired result and the B-A difference has low variance.

    uv run python scripts/deck_ab.py --deck-a decks/pool/blue-eyes.ydk --deck-b decks/pool/other.ydk

Paired statistics (sign test, paired diff CI) reuse scripts/audit/paired_gate.py's functions
by import, with local copies as fallback. play_match is run in sync mode so two runs with
identical decks and same-seed greedy policies produce identical games (the noise check).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from decklight.decks import load_ydk  # noqa: E402
from decklight.engine.duel import Engine  # noqa: E402
from decklight.eval.match import play_match, wilson  # noqa: E402


def _paired_math():
    """signtest_p / diff_ci from scripts/audit/paired_gate.py (import; copy on failure)."""
    try:
        p = ROOT / "scripts" / "audit" / "paired_gate.py"
        spec = importlib.util.spec_from_file_location("deck_ab_paired_gate", p)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.signtest_p, mod.diff_ci
    except Exception:  # noqa: BLE001 - fall back to the local copies below
        pass

    def signtest_p(wl: int, lw: int) -> float:
        n = wl + lw
        if n == 0:
            return 1.0
        m = min(wl, lw)
        return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(m + 1)) / 2**n)

    def diff_ci(diffs: list[float]) -> tuple[float, float, float]:
        n = len(diffs)
        mean = sum(diffs) / n
        if n < 2:
            return mean, mean, mean
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        h = 1.96 * math.sqrt(var / n)
        return mean, mean - h, mean + h

    return signtest_p, diff_ci


signtest_p, diff_ci = _paired_math()

BUILTIN_POLICIES = ("random", "greedy", "ygoagent")


def power(n_pairs: int, sd: float, target: float) -> dict:
    """Power calc for a paired B-A comparison (win-rate fractions, not pp).

    Minimal detectable effect at alpha = 0.05 two-sided, power 0.8:
    MDE = (1.96 + 0.84) * sd / sqrt(n_pairs); n_needed inverts it for `target`.
    """
    z = 1.96 + 0.84
    mde = z * sd / math.sqrt(n_pairs) if n_pairs > 0 else float("inf")
    n_needed = math.ceil((z * sd / target) ** 2) if target > 0 else n_pairs
    return {"n_pairs": n_pairs, "sd": sd, "target": target, "mde": mde,
            "mde_pp": mde * 100, "n_needed": n_needed}


def power_lines(n_opp: int, seeds: int, pc: dict, target_pp: float) -> list[str]:
    seeds_needed = math.ceil(pc["n_needed"] / (n_opp * 2)) if n_opp > 0 else pc["n_needed"]
    lines = [
        f"power: n_pairs={pc['n_pairs']} ({n_opp} opp decks x {seeds} seeds x 2 seats), "
        f"assumed per-pair SD of B-A = {pc['sd']}",
        f"power: MDE at alpha=0.05 two-sided, power 0.8 = {pc['mde_pp']:.2f} pp",
        f"power: n_pairs needed for target {target_pp:g} pp = {pc['n_needed']} "
        f"(~{seeds_needed} seeds at {n_opp} opp decks)",
    ]
    if pc["mde_pp"] > target_pp:
        lines.append(f"WARNING deck_ab: MDE {pc['mde_pp']:.2f} pp > target {target_pp:g} pp: "
                     f"need ~{seeds_needed} seeds at {n_opp} opp decks for {target_pp:g} pp")
    return lines


def default_opp_files(opp_spec: str) -> list[Path]:
    if opp_spec == "ygoagent":
        return sorted((ROOT / "decks" / "gauntlet").glob("*.ydk"))
    hold = set((ROOT / "decks" / "pool" / "HOLDOUT").read_text().split()) | {"kewl-tune"}
    return sorted(p for p in (ROOT / "decks" / "pool").glob("*.ydk") if p.stem not in hold)


def make_side(spec: str, emb: Path, device: str, seed: int):
    """A fresh policy for one variant run (fresh so both runs draw identical sequences)."""
    if spec in BUILTIN_POLICIES:
        if spec == "ygoagent":
            from decklight.eval.ygoagent import YgoAgentPolicy

            return YgoAgentPolicy(seed=seed), spec
        from decklight.eval.policies import GreedyPolicy, RandomPolicy

        cls = GreedyPolicy if spec == "greedy" else RandomPolicy
        return cls(seed), spec
    from decklight.eval.bc_policy import BCPolicy

    pol = BCPolicy(spec, emb, device=device)
    return pol, getattr(pol, "name", spec)


def load_variant(label: str, path: Path):
    try:
        d = load_ydk(path)
    except Exception as e:  # noqa: BLE001 - missing/unreadable file
        return None, f"ALERT deck_ab: deck {label} {path} failed to load: {e}"
    if not 40 <= len(d.main) <= 60:
        return None, f"ALERT deck_ab: deck {label} {path} illegal main count {len(d.main)} (need 40-60)"
    if len(d.extra) > 15:
        return None, f"ALERT deck_ab: deck {label} {path} illegal extra count {len(d.extra)} (need <= 15)"
    return d, ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paired deck A/B: same pilot, same deals, both seats.")
    ap.add_argument("--deck-a", type=Path, required=True)
    ap.add_argument("--deck-b", type=Path, required=True)
    ap.add_argument("--pilot", default="greedy", help="greedy|random|ygoagent|<path to best.pt>")
    ap.add_argument("--opp", default="greedy", help="greedy|random|ygoagent|<path to best.pt>")
    ap.add_argument("--opp-decks", type=Path, default=None)
    ap.add_argument("--n-opp-decks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for --n-opp-decks")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None, help="report.json + summary.txt dir")
    ap.add_argument("--sd", type=float, default=0.75, help="assumed per-pair SD of B-A (fractions)")
    ap.add_argument("--target-pp", type=float, default=3, help="effect to detect, percentage points")
    ap.add_argument("--power-only", action="store_true", help="print the power calc and exit 0")
    return ap.parse_args(argv)


def _summarize(keys: list, a: dict, b: dict) -> dict:
    n = len(keys)
    diffs = [b[k] - a[k] for k in keys]
    mean, lo, hi = diff_ci(diffs)
    se = (hi - mean) / 1.96 if n >= 2 else 0.0
    wa, wb = sum(a[k] for k in keys), sum(b[k] for k in keys)
    loa, hia = wilson(wa, n)
    lob, hib = wilson(wb, n)
    ab = sum(1 for k in keys if a[k] == 0.0 and b[k] == 1.0)  # A lost -> B won
    ba = sum(1 for k in keys if a[k] == 1.0 and b[k] == 0.0)  # A won -> B lost
    ties = sum(1 for k in keys if a[k] == b[k])
    return {"n": n, "win_rate_a": wa / n, "ci_a": [loa, hia], "win_rate_b": wb / n,
            "ci_b": [lob, hib], "diff": mean, "se": se, "ci": [lo, hi],
            "a_lost_b_won": ab, "a_won_b_lost": ba, "ties": ties,
            "half": n - ab - ba - ties, "sign_p": signtest_p(ba, ab)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    opp_dir = args.opp_decks or ("decks/gauntlet" if args.opp == "ygoagent" else "decks/pool")
    opp_dir = Path(opp_dir)
    if args.opp_decks is not None:
        opp_files = sorted(p for p in opp_dir.glob("*.ydk"))
    else:
        opp_files = default_opp_files(args.opp) if args.opp != "ygoagent" else sorted(opp_dir.glob("*.ydk"))
    if args.n_opp_decks is not None:
        if args.n_opp_decks < 1:
            print(f"ALERT deck_ab: --n-opp-decks must be >= 1, got {args.n_opp_decks}", flush=True)
            return 2
        opp_files = random.Random(args.seed).sample(sorted(opp_files), min(args.n_opp_decks, len(opp_files)))
    if not opp_files or args.seeds < 1:
        print(f"ALERT deck_ab: nothing to play: {len(opp_files)} opp decks, {args.seeds} seeds", flush=True)
        return 2

    pc = power(len(opp_files) * args.seeds * 2, args.sd, args.target_pp / 100)
    for line in power_lines(len(opp_files), args.seeds, pc, args.target_pp):
        print(line, flush=True)
    if args.power_only:
        return 0

    for spec in (args.pilot, args.opp):
        if spec not in BUILTIN_POLICIES and not Path(spec).is_file():
            print(f"ALERT deck_ab: pilot/opp {spec!r} is not a builtin policy nor a checkpoint file", flush=True)
            return 2
    deck_a, err = load_variant("A", args.deck_a)
    if deck_a is None:
        print(err, flush=True)
        return 2
    deck_b, err = load_variant("B", args.deck_b)
    if deck_b is None:
        print(err, flush=True)
        return 2
    try:
        opp_decks = [load_ydk(p) for p in opp_files]
    except Exception as e:  # noqa: BLE001
        print(f"ALERT deck_ab: opponent deck failed to load: {e}", flush=True)
        return 2

    identical = (sorted(deck_a.main) == sorted(deck_b.main)
                 and sorted(deck_a.extra) == sorted(deck_b.extra))
    same_order = (tuple(deck_a.main) == tuple(deck_b.main)
                  and tuple(deck_a.extra) == tuple(deck_b.extra))
    if identical:
        print("WARNING identical decks (same card multiset): this run is a noise check, diff must be ~0",
              flush=True)

    engine = Engine()
    seeds = list(range(args.seeds))
    pairs = [(0, j) for j in range(1, len(opp_decks) + 1)]
    scores: dict[str, dict] = {}
    for tag, variant in (("A", deck_a), ("B", deck_b)):
        pilot, _ = make_side(args.pilot, args.emb, args.device, seed=1)
        opp, _ = make_side(args.opp, args.emb, args.device, seed=2)
        games = play_match(engine, [variant] + opp_decks, pilot, opp, pairs, seeds,
                           num_envs=args.num_envs, num_threads=args.threads, mode="sync")
        opp_names = [p.stem for p in opp_files]
        scores[tag] = {(opp_names[g.deck_b - 1], g.seed, g.a_first): g.winner_is_a for g in games}
    if not scores["A"] or not scores["B"]:
        print("ALERT deck_ab: 0 games played", flush=True)
        return 2
    keys = sorted(set(scores["A"]) & set(scores["B"]))
    cover = len(keys) / max(len(scores["A"]), len(scores["B"]))
    if cover < 0.95:
        print(f"ALERT deck_ab: only {len(keys)}/{max(len(scores['A']), len(scores['B']))} games pair up "
              f"({cover:.1%}, need >= 95%)", flush=True)
        return 2

    overall = _summarize(keys, scores["A"], scores["B"])
    by_seat = {name: _summarize([k for k in keys if k[2] == first], scores["A"], scores["B"])
               for name, first in (("a_first", True), ("a_second", False))}
    per_opp = {}
    for opp_name in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == opp_name]
        s = _summarize(ks, scores["A"], scores["B"])
        per_opp[opp_name] = {"n": s["n"], "diff": s["diff"], "se": s["se"], "ci": s["ci"],
                             "win_rate_a": s["win_rate_a"], "win_rate_b": s["win_rate_b"]}
    per_game = [{"opp": k[0], "seed": k[1], "a_first": k[2],
                 "a_score": scores["A"][k], "b_score": scores["B"][k]} for k in keys]

    # Noise check: fully identical decks (same multiset in the same order) on the same seeds deal
    # the same cards, so deterministic pilots must reproduce exactly. A resequenced multiset is NOT
    # held to this: card order changes the deal (engine shuffles the list as given), so its games
    # may legitimately differ; pairing is by seed, not by identical draws.
    if identical and same_order and args.pilot == "greedy" and args.opp == "greedy" \
            and overall["diff"] != 0.0:
        print(f"ALERT deck_ab: identical decks with deterministic pilots differ by "
              f"{overall['diff']:+.3f} (must be exactly 0)", flush=True)
        return 2

    seeds_needed = math.ceil(pc["n_needed"] / (len(opp_files) * 2))
    report = {"deck_a": str(args.deck_a), "deck_b": str(args.deck_b), "pilot": args.pilot,
              "opp": args.opp, "opp_decks": [p.name for p in opp_files], "seeds": args.seeds,
              "n_pairs": len(keys), "power": pc, "seeds_needed": seeds_needed,
              "win_rate_a": overall["win_rate_a"], "ci_a": overall["ci_a"],
              "win_rate_b": overall["win_rate_b"], "ci_b": overall["ci_b"],
              "diff": overall["diff"], "se": overall["se"], "ci": overall["ci"],
              "a_lost_b_won": overall["a_lost_b_won"], "a_won_b_lost": overall["a_won_b_lost"],
              "ties": overall["ties"], "half": overall["half"], "sign_p": overall["sign_p"],
              "by_seat": by_seat, "per_opp": per_opp, "identical": identical,
              "same_order": same_order, "per_game": per_game}

    summary = [
        f"deck A/B: {args.deck_a} vs {args.deck_b} (pilot {args.pilot}, opp {args.opp})",
        *power_lines(len(opp_files), args.seeds, pc, args.target_pp),
    ]
    if identical:
        summary.append("WARNING identical decks (same card multiset): noise check, diff must be ~0")
        if not same_order:
            all_equal = all(r["a_score"] == r["b_score"] for r in per_game)
            summary.append(f"identical multiset in a different card order; paired scores all equal: {all_equal}; "
                           "card order changes deals: pairing is by seed, not by identical draws")
    o = overall
    summary += [
        f"win rate A {o['win_rate_a']:.3f} [{o['ci_a'][0]:.3f}, {o['ci_a'][1]:.3f}]  "
        f"B {o['win_rate_b']:.3f} [{o['ci_b'][0]:.3f}, {o['ci_b'][1]:.3f}]  n={o['n']}",
        f"paired diff B-A {o['diff']:+.3f} (SE {o['se']:.3f}, 95% CI [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}])",
        f"strict flips A-lost->B-won {o['a_lost_b_won']}  A-won->B-lost {o['a_won_b_lost']}  "
        f"ties {o['ties']}  half-point {o['half']}  sign-test p={o['sign_p']:.5g}",
    ]
    for seat in ("a_first", "a_second"):
        s = by_seat[seat]
        summary.append(f"  {seat:<9} n={s['n']:4d}  A {s['win_rate_a']:.3f} vs B {s['win_rate_b']:.3f}  "
                       f"diff {s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  "
                       f"flips {s['a_lost_b_won']}/{s['a_won_b_lost']} p={s['sign_p']:.5g}")
    summary.append("per-opp diffs B-A:")
    for opp_name, s in sorted(per_opp.items()):
        summary.append(f"  {opp_name:<40} {s['diff']:+.3f} (SE {s['se']:.3f}) n={s['n']}")
    text = "\n".join(summary) + "\n"
    print(text, flush=True)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=1))
        (args.out / "summary.txt").write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
