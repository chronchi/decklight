"""Propose recurring behaviour changes from paired flip games (offline analysis, LLM).

    uv run python scripts/audit/behavior_proposer.py --pairs <flip_audit pairs dir> --out behaviors.json
        [--model opencode-go/muse-spark-1.3-contributor] [--batch 4] [--workers 2]

Sends the flip pairs in batches with PROMPT, merges/dedups candidates with a second LLM pass
(as in the reward-hacking paper: proposer names behaviours that rose during training, a judge
counts them later in behavior_judge.py). Every call goes through decklight.demos.llm (disk
cache by model + prompt hash, throttled). Prints the behaviour list.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decklight.demos import llm as _llm

MODEL = "opencode-go/muse-spark-1.3-contributor"

PROMPT = """You are analyzing how a fine-tuned Yu-Gi-Oh! game-playing policy changed relative to its base checkpoint.

Below are {n} paired duels played on IDENTICAL deals (same decks, same shuffle seed, same seat). Model A is the base policy; model B is its fine-tune on the pilot deck (marked "us", decisions prefixed `>>`; the opponent is the scripted ygo-agent, identical in both games of a pair). Each pair's header states the outcome-change direction (L->W = base lost, fine-tune won; W->L = the reverse) and the first decision where B's choice diverged from A's, then both full game logs.

{GAMES}

List recurring *behaviours* where B plays differently from A. A behaviour is a pattern visible in the logs (e.g. "B normal-summons the starter instead of setting it", "B holds Maxx C for the opponent's turn instead of firing it turn 1", "B attacks for lethal instead of passing with lethal on board"). Requirements:
- each behaviour must have evidence in at least 2 of the games above (cite game ids like "L->W-003");
- each behaviour gets one verdict on whether it plausibly caused the outcome change (caused / contributed / incidental / unclear) with a one-sentence reason;
- behaviours that appear in W->L pairs (regressions) are as important as L->W ones;
- ignore differences in the opponent's (non-`>>`) lines except as context for our decisions.

JSON only, no prose: a list of objects with exactly these keys:
[{{"id": "short-snake-case-name", "title": "...", "description": "1-3 sentences, concrete cards/lines when possible", "evidence": ["<pair-file-stem>", ...], "causal_verdict": "caused|contributed|incidental|unclear", "causal_why": "...", "detector_hint": "how to spot this in a log in one sentence"}}]
Return [] if no behaviour has 2+ games of evidence.
"""

MERGE_PROMPT = """You are deduplicating candidate behaviour descriptions of a fine-tuned Yu-Gi-Oh! policy (JSON below, from several batches of paired games; game-id evidence may overlap or use different pair-file names for the same kind of flip).

{CANDS}

Merge entries that describe the SAME behaviour change into one (union their evidence lists, keep the clearest title/description, keep the strongest causal verdict with a one-sentence reason). Drop entries with fewer than 2 distinct games of evidence after merging. Keep genuinely distinct behaviours separate even if they involve the same card.

JSON only, no prose: the merged list with exactly these keys per object:
[{{"id": "short-snake-case-name", "title": "...", "description": "...", "evidence": [...], "causal_verdict": "caused|contributed|incidental|unclear", "causal_why": "...", "detector_hint": "..."}}]
"""


def parse_behaviors(text: str) -> list[dict]:
    """Parse an LLM reply into a behaviour list; tolerant of code fences and prose around the JSON."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", t, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                obj = v
                break
    if not isinstance(obj, list):
        return []
    out = []
    for e in obj:
        if not isinstance(e, dict) or not e.get("title"):
            continue
        out.append({
            "id": str(e.get("id") or re.sub(r"\W+", "_", str(e["title"]).lower()).strip("_")[:60]),
            "title": str(e["title"]),
            "description": str(e.get("description", "")),
            "evidence": [str(x) for x in e.get("evidence", []) or []],
            "causal_verdict": str(e.get("causal_verdict", "unclear")),
            "causal_why": str(e.get("causal_why", "")),
            "detector_hint": str(e.get("detector_hint", "")),
        })
    return out


def _batches(files: list[Path], n: int) -> list[list[Path]]:
    return [files[i: i + n] for i in range(0, len(files), n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, required=True, help="flip_audit pairs/ dir with NNN.txt files")
    ap.add_argument("--out", type=Path, required=True, help="behaviors.json to write")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=4, help="pairs per proposer prompt")
    ap.add_argument("--workers", type=int, default=2, help="concurrent LLM calls")
    args = ap.parse_args(argv)

    files = sorted(args.pairs.glob("*.txt"))
    if not files:
        print(f"ALERT behavior_proposer: no pair files in {args.pairs}", flush=True)
        return 1
    batches = _batches(files, args.batch)
    print(f"{len(files)} pairs in {len(batches)} batches of <={args.batch}", flush=True)
    prompts = []
    for b in batches:
        games = "".join(f"\n--- pair file {f.stem} ---\n{f.read_text()[: 30000]}" for f in b)
        prompts.append(PROMPT.format(n=len(b), GAMES=games))
    resps = _llm.complete_many(prompts, model=args.model, max_workers=args.workers, return_exceptions=True)
    cands: list[dict] = []
    for b, r in zip(batches, resps, strict=True):
        if isinstance(r, Exception):
            print(f"ALERT behavior_proposer: batch {[f.stem for f in b]} failed: {r}", flush=True)
            continue
        got = parse_behaviors(r)
        print(f"batch {[f.stem for f in b]}: {len(got)} candidates", flush=True)
        cands.extend(got)
    if not cands:
        print("ALERT behavior_proposer: no candidates from any batch", flush=True)
        return 2
    merge = _llm.complete(MERGE_PROMPT.format(CANDS=json.dumps(cands)[: 60000]), model=args.model)
    merged = parse_behaviors(merge)
    final = merged or cands
    for i, b in enumerate(final):
        b["id"] = f"b{i + 1:02d}-{b['id']}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(final, indent=1) + "\n")
    print(f"wrote {len(final)} behaviours -> {args.out}")
    for b in final:
        print(f"  [{b['id']}] {b['title']} ({b['causal_verdict']}, {len(b['evidence'])} games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
