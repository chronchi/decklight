"""Count proposed-behaviour prevalence per checkpoint (offline analysis, LLM judge).

    uv run python scripts/audit/behavior_judge.py --behaviors behaviors.json --card <card dir> \\
        --arms base,it0040,it0080,it0120,it0160,final --deals 120 --out prevalence.json \\
        [--model opencode-go/deepseek-v4.1-flash] [--dry-run]

For a seeded sample of deals (the SAME deals for every arm), render each arm's game once and
ask the judge, in one call per game, which behaviours occur (JSON verdicts). Output per
behaviour x arm: prevalence (Wilson 95% CI) plus our win rate with vs without the behaviour.
Prints a Figure-3-style table. Every call goes through decklight.demos.llm (disk cache by
model + prompt hash, throttled). --dry-run prints the call count and token estimate only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sm_common as SM

from decklight.demos import llm as _llm
from decklight.eval.match import wilson

MODEL = "opencode-go/deepseek-v4.1-flash"
MAX_RENDER_CHARS = 12000

JUDGE_PROMPT = """You are judging a Yu-Gi-Oh! duel log. Our bot's decisions are the lines prefixed `>>` (it pilots {us}); the opponent is scripted. Only our (`>>`) lines count as our behaviour.

{GAME}

Which of these behaviours occur in OUR play in this game? Reply JSON only, no prose: {{"<id>": true|false, ...}} for exactly these ids:
{BEHAVIOURS}
where each behaviour is "<id>: <title> — <detector_hint>".
true = the behaviour clearly occurs at least once in our (`>>`) lines; false otherwise. When unsure, answer false.
"""


def parse_verdicts(text: str, ids: list[str]) -> dict[str, bool] | None:
    """Parse a judge reply into {id: bool} for the known ids; None when unparseable."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return None
    out = {}
    for i in ids:
        v = obj.get(i)
        if isinstance(v, bool):
            out[i] = v
        elif isinstance(v, (int, float)) and v in (0, 1):
            out[i] = bool(v)
        elif isinstance(v, str) and v.lower() in ("true", "false", "yes", "no"):
            out[i] = v.lower() in ("true", "yes")
    return out if len(out) == len(ids) else None


def aggregate(behaviors: list[dict], verdicts: dict, wins: dict) -> dict:
    """Prevalence + win-with/without per behaviour x arm.

    verdicts: {(behaviour_id, arm, deal_key): bool}; wins: {(arm, deal_key): 0/0.5/1}.
    Returns {behaviour_id: {arm: {prev, lo, hi, n, n_with, win_with, win_without}}}.
    """
    out: dict = {}
    for b in behaviors:
        bid = b["id"]
        out[bid] = {"title": b.get("title", ""), "arms": {}}
        for arm in sorted({a for (_, a, _) in verdicts}):
            vv = [(k, v) for (i, a, k), v in verdicts.items() if i == bid and a == arm]
            n = len(vv)
            n_with = sum(1 for _, v in vv if v)
            lo, hi = wilson(n_with, n)
            with_w = [wins.get((arm, k), 0.5) for k, v in vv if v]
            wo_w = [wins.get((arm, k), 0.5) for k, v in vv if not v]
            out[bid]["arms"][arm] = {
                "prev": round(n_with / n, 3) if n else None,
                "lo": round(lo, 3), "hi": round(hi, 3), "n": n, "n_with": n_with,
                "win_with": round(sum(with_w) / len(with_w), 3) if with_w else None,
                "win_without": round(sum(wo_w) / len(wo_w), 3) if wo_w else None,
            }
    return out


def _deal_str(key: tuple) -> str:
    return f"{key[0][0]}-vs-{key[0][1]} seed={key[1]} P{key[2]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behaviors", type=Path, required=True)
    ap.add_argument("--card", type=Path, required=True, help="card dir holding one subdir per arm")
    ap.add_argument("--arms", required=True, help="comma-separated arm names, e.g. base,it0160,final")
    ap.add_argument("--deals", type=int, default=120, help="seeded sample of deals (same for every arm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gauntlet", type=Path, default=SM.GAUNTLET)
    ap.add_argument("--dry-run", action="store_true", help="print the call/token estimate only")
    args = ap.parse_args(argv)

    behaviors = json.loads(args.behaviors.read_text())
    if not behaviors:
        print(f"ALERT behavior_judge: no behaviours in {args.behaviors}", flush=True)
        return 1
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    by_arm = {a: {SM.deal_key(r): r for r in SM.iter_records(args.card / a)} for a in arms}
    for a, d in by_arm.items():
        if not d:
            print(f"ALERT behavior_judge: 0 games in {args.card / a}", flush=True)
            return 1
    common = sorted(set.intersection(*(set(d) for d in by_arm.values())))
    if not common:
        print("ALERT behavior_judge: no deal is present in every arm", flush=True)
        return 1
    rng = random.Random(args.seed)
    deals = rng.sample(common, min(args.deals, len(common)))
    print(f"{len(common)} common deals, judging {len(deals)} x {len(arms)} games", flush=True)

    engine, names = SM.Engine(), SM.CardNames()
    jobs: list[tuple[str, tuple]] = []  # (arm, deal_key)
    prompts: list[str] = []
    sizes: list[int] = []
    for a in arms:
        for k in deals:
            rec = by_arm[a][k]
            text, ok = SM.render_record(rec, engine, names, args.gauntlet)
            if not ok:
                print(f"ALERT behavior_judge: replay of {_deal_str(k)} ({a}) does not reproduce winner",
                      flush=True)
                return 2
            if len(text) > MAX_RENDER_CHARS:
                text = text[:MAX_RENDER_CHARS] + "\n... [truncated]"
            us = rec["decks"][rec["s_seat"]]
            blist = "\n".join(f"{b['id']}: {b['title']} — {b.get('detector_hint', '')}" for b in behaviors)
            prompts.append(JUDGE_PROMPT.format(us=us, GAME=text, BEHAVIOURS=blist))
            sizes.append(len(prompts[-1]))
            jobs.append((a, k))
    est_tokens = sum(sizes) // 4 + len(jobs) * 100
    print(f"calls: {len(jobs)} judge calls (1 per game), ~{est_tokens} input tokens total "
          f"(~{est_tokens // max(len(jobs), 1)}/call), model {args.model}", flush=True)
    if args.dry_run:
        return 0

    ids = [b["id"] for b in behaviors]
    resps = _llm.complete_many(prompts, model=args.model, max_workers=args.workers, return_exceptions=True)
    verdicts: dict = {}
    wins: dict = {}
    bad = 0
    for (a, k), r in zip(jobs, resps, strict=True):
        rec = by_arm[a][k]
        wins[(a, k)] = SM.our_score(rec)
        if isinstance(r, Exception):
            print(f"ALERT behavior_judge: {a} {_deal_str(k)} failed: {r}", flush=True)
            bad += 1
            continue
        v = parse_verdicts(r, ids)
        if v is None:
            print(f"ALERT behavior_judge: {a} {_deal_str(k)} unparseable verdict", flush=True)
            bad += 1
            continue
        for i, val in v.items():
            verdicts[(i, a, k)] = val
    if bad:
        print(f"ALERT behavior_judge: {bad}/{len(jobs)} judgements failed or unparseable", flush=True)
    prev = aggregate(behaviors, verdicts, wins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arms": arms, "deals": len(deals), "failed": bad,
                                    "prevalence": prev}, indent=1))
    print(f"wrote {args.out}")
    hdr = f"{'behaviour':40s}" + "".join(f"{a:>28s}" for a in arms)
    print(hdr)
    for b in behaviors:
        row = f"{b['id'][:40]:40s}"
        for a in arms:
            s = prev[b["id"]]["arms"][a]
            w = f" W{s['win_with']}/{s['win_without']}" if s["win_with"] is not None else ""
            row += f"{s['prev']:.0%} [{s['lo']:.0%},{s['hi']:.0%}]{w}".rjust(28)
        print(row)
    return 0 if not bad else 3


if __name__ == "__main__":
    raise SystemExit(main())
