"""Paired tech-card A/B: is variant B (slot-swapped cards) better than the original deck A?

Each question names a deck plus slot-for-slot swaps. A and B play the same
opponents on the same duel seeds in both seats (paired, like scripts/deck_ab.py),
so each (opponent, seed, seat) is one B-A pair. Because the swap preserves list
positions and the shuffle depends only on (seed, team, length), every deal is
identical except for the swapped card's identity.

Exposure: for each pair the variant's shuffled main deck is recomputed from
(seed, team), so pairs split into in-opening-hand / drawn-by-turn-3 /
unexposed. Unexposed pairs must play identically: the identical-choice-sequence
rate there is the noise floor.

Question file (JSON list):
    {"id": "...", "deck": "decks/gauntlet/Branded.ydk",
     "swaps": [[out_name_or_code, in_name_or_code, n]], "kind": "real"|"neg_control"|"pos_control"}
`n` (copies, default all) may be omitted per swap.

    uv run python scripts/tech_ab.py --questions docs/techcards/questions-2026-10-16.json \\
        --pilot greedy --opp greedy --n-opp-decks 2 --seeds 2 --out data/cache/tech-smoke

Splitting across processes (vast): --only <id> --seed0 <s> --seeds <b> writes one
unit dir per call; --collect <out> <unit...> merges units back into per-question
reports plus an overall table. A/A (deck vs itself, full invocation scale) runs
once per invocation as a built-in check and exits 2 unless the diff is exactly 0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import deck_ab as _ab

from decklight.decks import load_ydk
from decklight.engine.duel import Engine
from decklight.eval.match import play_match
from decklight.eval.techcards import (
    HAND_SIZE,
    apply_swaps,
    card_name,
    opening_exposure,
    resolve_code,
)

KINDS = ("real", "neg_control", "pos_control")
SEEN_TURNS = 3  # "seen by turn t" approximation: opening 5 + t draws


def load_questions(path: Path) -> list[dict]:
    try:
        items = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"ALERT tech_ab: question file {path} unreadable: {e}")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"ALERT tech_ab: question file {path} must be a non-empty JSON list")
    seen = set()
    for q in items:
        if not isinstance(q, dict) or not all(k in q for k in ("id", "deck", "swaps", "kind")):
            raise SystemExit(f"ALERT tech_ab: question needs id/deck/swaps/kind: {q!r}")
        if q["id"] in seen or q["kind"] not in KINDS:
            raise SystemExit(f"ALERT tech_ab: duplicate id or bad kind in {q!r}")
        seen.add(q["id"])
        if not isinstance(q["swaps"], list) or not q["swaps"]:
            raise SystemExit(f"ALERT tech_ab: {q['id']}: swaps must be a non-empty list")
        for s in q["swaps"]:
            if not isinstance(s, (list, tuple)) or len(s) not in (2, 3):
                raise SystemExit(f"ALERT tech_ab: {q['id']}: swap needs [out, in] or [out, in, n]: {s!r}")
    return items


def resolve_swaps(deck_path: Path, swaps: list) -> tuple[list[tuple[int, int, int | None]], list[str]]:
    """Swaps to codes; returns ((out, in, n) list, human-readable lines)."""
    deck = load_ydk(deck_path)
    codes, lines = [], []
    for s in swaps:
        out_c, in_c = resolve_code(s[0]), resolve_code(s[1])
        n = None if len(s) == 2 else s[2]
        if n is not None and (not isinstance(n, int) or n < 1):
            raise SystemExit(f"ALERT tech_ab: swap count must be a positive int: {s!r}")
        try:
            nm_out, nm_in = card_name(out_c), card_name(in_c)
        except KeyError as e:
            raise SystemExit(f"ALERT tech_ab: {e}")
        have_main = list(deck.main).count(out_c)
        if out_c not in deck.main and out_c not in deck.extra:
            raise SystemExit(f"ALERT tech_ab: {nm_out} ({out_c}) not in {deck_path}")
        codes.append((out_c, in_c, n))
        tag = "" if nm_out != nm_in else f" [{out_c}->{in_c}]"
        lines.append(f"{nm_out} x{have_main} -> {nm_in}{tag}"
                     + ("" if n is None else f" ({n} of {have_main} copies)"))
    return codes, lines


def opp_files(args: argparse.Namespace) -> list[Path]:
    d = Path(args.opp_decks) if args.opp_decks else ROOT / "decks" / "gauntlet"
    files = sorted(p for p in d.glob("*.ydk"))
    if args.n_opp_decks is not None:
        if args.n_opp_decks < 1:
            raise SystemExit(f"ALERT tech_ab: --n-opp-decks must be >= 1, got {args.n_opp_decks}")
        files = random.Random(args.seed).sample(files, min(args.n_opp_decks, len(files)))
    if not files or args.seeds < 1:
        raise SystemExit(f"ALERT tech_ab: nothing to play: {len(files)} opp decks, {args.seeds} seeds")
    return files


def run_side(engine, deck, opp_decks, pilot_spec, opp_spec, pairs, seeds, args):
    pilot, _ = _ab.make_side(pilot_spec, args.emb, args.device, seed=1)
    opp, _ = _ab.make_side(opp_spec, args.emb, args.device, seed=2)
    return play_match(engine, [deck] + opp_decks, pilot, opp, pairs, seeds,
                      num_envs=args.num_envs, num_threads=args.threads, mode="sync")


def pair_rows(games_a, games_b, names, main_b, in_codes_main) -> list[dict]:
    """Join two play_match runs on (opp, seed, seat); attach exposure + choice identity."""
    a = {(names[g.deck_b - 1], g.seed, g.a_first): g for g in games_a}
    b = {(names[g.deck_b - 1], g.seed, g.a_first): g for g in games_b}
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("ALERT tech_ab: 0 games pair up")
    rows = []
    for k in keys:
        ga, gb = a[k], b[k]
        opp, seed, a_first = k
        team = 0 if a_first else 1
        top5 = {c for _, c in opening_exposure(main_b, seed, team, HAND_SIZE)}
        top8 = {c for _, c in opening_exposure(main_b, seed, team, HAND_SIZE + SEEN_TURNS)}
        rows.append({"opp": opp, "seed": seed, "a_first": a_first,
                     "a_score": ga.winner_is_a, "b_score": gb.winner_is_a,
                     "turns_a": ga.turns, "turns_b": gb.turns,
                     "exposed_opening": bool(in_codes_main & top5),
                     "seen_t3": bool(in_codes_main & top8),
                     "same_choices": tuple(ga.choices) == tuple(gb.choices)})
    return rows


def _sub(rows, pred) -> dict:
    ks = [i for i, r in enumerate(rows) if pred(r)]
    if not ks:  # an exposure split can be empty (tiny runs); report zeros, never crash
        return {"n": 0, "win_rate_a": 0.0, "ci_a": [0.0, 1.0], "win_rate_b": 0.0,
                "ci_b": [0.0, 1.0], "diff": 0.0, "se": 0.0, "ci": [0.0, 0.0],
                "a_lost_b_won": 0, "a_won_b_lost": 0, "ties": 0, "half": 0, "sign_p": 1.0}
    a = {i: rows[i]["a_score"] for i in ks}
    b = {i: rows[i]["b_score"] for i in ks}
    return _ab._summarize(ks, a, b)


def summarize(rows: list[dict]) -> dict:
    overall = _sub(rows, lambda r: True)
    by_seat = {name: _sub(rows, lambda r, f=f: r["a_first"] == f)
               for name, f in (("a_first", True), ("a_second", False))}
    opening = _sub(rows, lambda r: r["exposed_opening"])
    drawn = _sub(rows, lambda r: r["seen_t3"] and not r["exposed_opening"])
    unexposed = [r for r in rows if not r["seen_t3"]]
    unexp = _sub(rows, lambda r: not r["seen_t3"])
    # NOTE: identical_choice_rate is exact only for fully identical runs (A/A,
    # neg_control). A real swap can desync choices without changing outcomes via
    # deck searches (option-list exposure, e.g. Melody showing a changed deck),
    # so the outcome_diff_rate is the noise floor that must be ~0.
    noise = {"n": len(unexposed),
             "identical_choice_rate": sum(r["same_choices"] for r in unexposed) / len(unexposed)
             if unexposed else 1.0,
             "outcome_diff_rate": sum(r["a_score"] != r["b_score"] for r in unexposed) / len(unexposed)
             if unexposed else 0.0}
    ta = sum(r["turns_a"] for r in rows) / len(rows)
    tb = sum(r["turns_b"] for r in rows) / len(rows)
    return {"n_pairs": len(rows), "overall": overall, "by_seat": by_seat,
            "exposure": {"opening": opening, "drawn_by_t3": drawn, "unexposed": unexp},
            "noise_floor": noise, "mean_turns_a": ta, "mean_turns_b": tb}


def fmt_diff(s: dict) -> str:
    return (f"{s['diff']:+.3f} (SE {s['se']:.3f}, 95% CI [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]) "
            f"n={s['n']} flips {s['a_lost_b_won']}/{s['a_won_b_lost']} "
            f"ties {s['ties']} half {s['half']} p={s['sign_p']:.5g}")


def question_markdown(qid, kind, deck, swap_lines, stats, pilot, opp, opp_names, seeds) -> str:
    o, lines = stats["overall"], []
    lines.append(f"# {qid} ({kind}): {deck} (pilot {pilot}, opp {opp})")
    lines.append(f"swaps: {'; '.join(swap_lines)}")
    lines.append(f"opponents: {len(opp_names)} decks x {seeds} seeds x 2 seats = {stats['n_pairs']} pairs")
    lines.append(f"overall B-A {fmt_diff(o)}")
    lines.append(f"A {o['win_rate_a']:.3f} [{o['ci_a'][0]:.3f}, {o['ci_a'][1]:.3f}]  "
                 f"B {o['win_rate_b']:.3f} [{o['ci_b'][0]:.3f}, {o['ci_b'][1]:.3f}]")
    for seat in ("a_first", "a_second"):
        s = stats["by_seat"][seat]
        lines.append(f"  {seat:<9} B-A {fmt_diff(s)}")
    lines.append("exposure (variant shuffled with the pair's seed/team):")
    for g in ("opening", "drawn_by_t3", "unexposed"):
        s = stats["exposure"][g]
        lines.append(f"  {g:<12} B-A {fmt_diff(s)}")
    nf = stats["noise_floor"]
    lines.append(f"noise floor (never seen by turn 3, n={nf['n']}): identical choice sequences "
                 f"{nf['identical_choice_rate']:.3f}, outcome differs {nf['outcome_diff_rate']:.3f}")
    lines.append(f"mean turns A {stats['mean_turns_a']:.1f} / B {stats['mean_turns_b']:.1f}")
    lines.append("confounds: won-when-drawn rates favour long games (never rank by them); "
                 "the pilot may misplay new cards; results depend on the opponent mix and seat.")
    return "\n".join(lines) + "\n"


def write_question(outdir: Path, qid, payload: dict, markdown: str) -> None:
    d = outdir / qid
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(payload, indent=1))
    (d / "summary.md").write_text(markdown)


def run_question(engine, q, files, opp_decks, args) -> tuple[dict, str]:
    deck_path = ROOT / q["deck"] if not Path(q["deck"]).is_absolute() else Path(q["deck"])
    codes, swap_lines = resolve_swaps(deck_path, q["swaps"])
    deck_a = load_ydk(deck_path)
    deck_b = apply_swaps(deck_a, codes)
    in_codes_main = {ic for (oc, ic, _) in codes if oc in deck_a.main}
    main_b = list(deck_b.main)
    opp_names = [p.stem for p in files]
    pairs = [(0, j) for j in range(1, len(opp_decks) + 1)]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    ga = run_side(engine, deck_a, opp_decks, args.pilot, args.opp, pairs, seeds, args)
    gb = run_side(engine, deck_b, opp_decks, args.pilot, args.opp, pairs, seeds, args)
    if not ga or not gb:
        raise SystemExit(f"ALERT tech_ab: {q['id']}: 0 games played")
    rows = pair_rows(ga, gb, opp_names, main_b, in_codes_main)
    stats = summarize(rows)
    pc = _ab.power(len(rows), args.sd, args.target_pp / 100)
    payload = {"id": q["id"], "kind": q["kind"], "deck": q["deck"], "swaps": q["swaps"],
               "swap_lines": swap_lines, "pilot": args.pilot, "opp": args.opp,
               "opp_decks": [p.name for p in files], "seed0": args.seed0, "seeds": args.seeds,
               "n_pairs": len(rows), "power": pc, "stats": stats, "rows": rows}
    md = question_markdown(q["id"], q["kind"], q["deck"], swap_lines, stats,
                           args.pilot, args.opp, opp_names, args.seeds)
    return payload, md


def run_aa(engine, deck, files, opp_decks, args) -> tuple[list[dict], dict]:
    """A/A: the deck vs itself; two independent runs must agree exactly (exit 2 if not)."""
    opp_names = [p.stem for p in files]
    pairs = [(0, j) for j in range(1, len(opp_decks) + 1)]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    g1 = run_side(engine, deck, opp_decks, args.pilot, args.opp, pairs, seeds, args)
    g2 = run_side(engine, deck, opp_decks, args.pilot, args.opp, pairs, seeds, args)
    rows = pair_rows(g1, g2, opp_names, list(deck.main), set())
    stats = summarize(rows)
    return rows, stats


def overall_table(results: dict[str, dict]) -> str:
    lines = ["# tech-card overall: paired B-A per question",
             "",
             "| question | kind | n | B-A | 95% CI | sign p | opening n / B-A | unexposed n / ident |",
             "|---|---|---|---|---|---|---|---|"]
    for qid, p in results.items():
        o = p["stats"]["overall"]
        e, u = p["stats"]["exposure"]["opening"], p["stats"]["exposure"]["unexposed"]
        nf = p["stats"]["noise_floor"]
        lines.append(f"| {qid} | {p['kind']} | {o['n']} | {o['diff']:+.3f} | "
                     f"[{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}] | {o['sign_p']:.4g} | "
                     f"{e['n']} / {e['diff']:+.3f} | {u['n']} / {nf['identical_choice_rate']:.2f} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paired tech-card A/B with exposure splits.")
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--only", default=None, help="run one question id (also selects the A/A deck)")
    ap.add_argument("--pilot", default="greedy")
    ap.add_argument("--opp", default="greedy")
    ap.add_argument("--opp-decks", type=Path, default=None)
    ap.add_argument("--n-opp-decks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for --n-opp-decks")
    ap.add_argument("--seed0", type=int, default=0, help="first duel seed (units split seed blocks)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--sd", type=float, default=0.75)
    ap.add_argument("--target-pp", type=float, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-aa", action="store_true",
                    help="skip the built-in A/A check (sharded units; the launcher gates on --aa-only)")
    ap.add_argument("--aa-only", type=Path, default=None,
                    help="run only the A/A check on this deck at the invocation's scale")
    ap.add_argument("--collect", nargs="+", type=Path, default=None,
                    help="merge unit dirs (each with <qid>/report.json) into --out")
    return ap.parse_args(argv)


def do_collect(args: argparse.Namespace) -> int:
    if args.out is None:
        print("ALERT tech_ab: --collect needs --out", flush=True)
        return 2
    per_q: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    aa_rows: list[dict] = []
    for u in args.collect:
        for rep in sorted(u.glob("*/report.json")):
            if rep.parent.name == "aa":
                aa_rows += json.loads(rep.read_text())["rows"]
                continue
            p = json.loads(rep.read_text())
            per_q.setdefault(rep.parent.name, []).extend(p["rows"])
            meta[rep.parent.name] = {k: p[k] for k in
                                     ("kind", "deck", "swaps", "swap_lines", "pilot", "opp")}
    if not per_q:
        print(f"ALERT tech_ab: --collect found no question reports in {args.collect}", flush=True)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for qid, rows in sorted(per_q.items()):
        stats = summarize(rows)
        payload = {"id": qid, **meta[qid], "n_pairs": len(rows), "stats": stats, "rows": rows}
        md = question_markdown(qid, meta[qid]["kind"], meta[qid]["deck"], meta[qid]["swap_lines"],
                               stats, meta[qid]["pilot"], meta[qid]["opp"], [], 0)
        write_question(args.out, qid, payload, md)
        results[qid] = payload
    if aa_rows:
        aa_stats = summarize(aa_rows)
        (args.out / "aa").mkdir(exist_ok=True)
        (args.out / "aa" / "report.json").write_text(json.dumps(
            {"n_pairs": len(aa_rows), "stats": aa_stats, "rows": aa_rows}, indent=1))
        print(f"A/A merged: n={len(aa_rows)} diff={aa_stats['overall']['diff']:+.3f}", flush=True)
        if aa_stats["overall"]["diff"] != 0.0:
            print("ALERT tech_ab: merged A/A differs (must be exactly 0)", flush=True)
            return 2
    (args.out / "overall.json").write_text(json.dumps(
        {qid: {"kind": p["kind"], "n_pairs": p["n_pairs"], "stats": p["stats"]} for qid, p in results.items()},
        indent=1))
    (args.out / "overall.md").write_text(overall_table(results))
    print((args.out / "overall.md").read_text(), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.collect is not None:
        return do_collect(args)
    for spec in (args.pilot, args.opp):
        if spec not in _ab.BUILTIN_POLICIES and not Path(spec).is_file():
            print(f"ALERT tech_ab: pilot/opp {spec!r} is not a builtin policy nor a checkpoint file",
                  flush=True)
            return 2
    files = opp_files(args)
    try:
        opp_decks = [load_ydk(p) for p in files]
    except Exception as e:  # noqa: BLE001
        print(f"ALERT tech_ab: opponent deck failed to load: {e}", flush=True)
        return 2
    engine = Engine()
    if args.out is None:
        print("ALERT tech_ab: need --out", flush=True)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    if args.aa_only is not None:
        aa_deck = load_ydk(args.aa_only)
        aa_rows, aa_stats = run_aa(engine, aa_deck, files, opp_decks, args)
        (args.out / "aa").mkdir(exist_ok=True)
        (args.out / "aa" / "report.json").write_text(json.dumps(
            {"deck": str(args.aa_only), "n_pairs": len(aa_rows),
             "stats": aa_stats, "rows": aa_rows}, indent=1))
        aa = aa_stats["overall"]
        print(f"A/A ({args.aa_only}): diff {aa['diff']:+.3f} n={aa['n']}", flush=True)
        if aa["diff"] != 0.0:
            print(f"ALERT tech_ab: A/A differs by {aa['diff']:+.3f} (must be exactly 0)", flush=True)
            return 2
        return 0
    if args.questions is None:
        print("ALERT tech_ab: need --questions (or --collect, or --aa-only)", flush=True)
        return 2
    questions = load_questions(args.questions)
    if args.only is not None:
        questions = [q for q in questions if q["id"] == args.only]
        if not questions:
            print(f"ALERT tech_ab: --only {args.only} matches no question", flush=True)
            return 2
    results = {}
    for q in questions:
        payload, md = run_question(engine, q, files, opp_decks, args)
        write_question(args.out, q["id"], payload, md)
        results[q["id"]] = payload
        o = payload["stats"]["overall"]
        print(f"{q['id']}: B-A {fmt_diff(o)}", flush=True)

    # Built-in A/A check on the first selected deck at the invocation's scale
    # (skipped for sharded units: the launcher gates on a full-scale --aa-only).
    if not args.skip_aa:
        aa_deck_path = questions[0]["deck"]
        aa_deck = load_ydk(ROOT / aa_deck_path if not Path(aa_deck_path).is_absolute() else aa_deck_path)
        aa_rows, aa_stats = run_aa(engine, aa_deck, files, opp_decks, args)
        (args.out / "aa").mkdir(exist_ok=True)
        (args.out / "aa" / "report.json").write_text(json.dumps(
            {"deck": questions[0]["deck"], "n_pairs": len(aa_rows),
             "stats": aa_stats, "rows": aa_rows}, indent=1))
        aa = aa_stats["overall"]
        print(f"A/A ({aa_deck_path}): diff {aa['diff']:+.3f} n={aa['n']}", flush=True)
        if aa["diff"] != 0.0:
            print(f"ALERT tech_ab: A/A differs by {aa['diff']:+.3f} (must be exactly 0)", flush=True)
            return 2

    (args.out / "overall.json").write_text(json.dumps(
        {qid: {"kind": p["kind"], "n_pairs": p["n_pairs"], "stats": p["stats"]} for qid, p in results.items()},
        indent=1))
    (args.out / "overall.md").write_text(overall_table(results))
    print((args.out / "overall.md").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
