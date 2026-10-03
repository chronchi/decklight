"""Data-quality report over the demonstration records (no engine replay; fast).

    uv run python scripts/data_report.py [--top 8]

Prints: records and decisions by kind (line / improved / expanded / game plan / interruption) and label, deck
coverage (verified line records per deck: min / median / zero-coverage decks), game stats (winners, turns, interrupts,
judge scores when present), and the decks most in need of data (train vs holdout).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "data" / "demos"
TRAINABLE = {"stated", "inferred", "interrupt"}


def kind_of(path: Path) -> str:
    parts = path.relative_to(DEMOS).parts
    if any(p.startswith("expanded") for p in parts):
        return "line_expanded"
    if "improved" in parts:
        return "line_improved"
    if parts[0] == "games":
        return "game"
    return "line"


def load(path: Path) -> list[dict]:
    out = []
    for raw in path.read_text().splitlines():
        if raw.strip():
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk"))
    hold_file = ROOT / "decks" / "pool" / "HOLDOUT"
    holdout = set(hold_file.read_text().split()) | {"kewl-tune"} if hold_file.exists() else {"kewl-tune"}

    recs: Counter = Counter()
    decisions: dict[str, Counter] = defaultdict(Counter)
    line_cov: Counter = Counter()  # verified (non-expanded) line records per deck
    games = []
    for f in sorted(DEMOS.rglob("*.jsonl")):
        if "games_judged" in f.parts or ("lines" in f.parts and "verified" not in f.parts and "expanded" not in f.name):
            if f.parent.name in ("lines", "report"):
                continue
        k = kind_of(f)
        if f.parent.name in ("lines", "report") or (k != "game" and "verified" not in f.parts
                                                    and not any(p.startswith("expanded") for p in f.parts)):
            continue
        rows = load(f)
        recs[k] += len(rows)
        for r in rows:
            for lab in r.get("labels", []):
                kk = "game_interrupt" if (k == "game" and lab == "interrupt") else ("game_plan" if k == "game" else k)
                decisions[kk][lab] += 1
            if k in ("line", "line_improved"):
                line_cov[str(r.get("deck", "")).split("+")[0]] += 1
            if k == "game":
                games.append(r)

    print("== records by kind")
    for k, n in recs.most_common():
        print(f"  {k:15s} {n:7d}")
    print("== decisions by kind (trainable = stated+inferred+interrupt)")
    tot = 0
    for k in sorted(decisions):
        c = decisions[k]
        t = sum(v for lab, v in c.items() if lab in TRAINABLE)
        tot += t
        print(f"  {k:15s} trainable {t:7d}  " + " ".join(f"{lab}={v}" for lab, v in c.most_common()))
    print(f"  total trainable {tot}")

    print(f"== deck coverage ({len(pool)} pool decks, {len(holdout & set(pool))} holdout)")
    cov = [line_cov.get(d, 0) for d in pool]
    print(f"  verified line records per deck: min {min(cov)}, median {statistics.median(cov)}, max {max(cov)}; "
          f"zero: {sum(c == 0 for c in cov)}")
    need = sorted(pool, key=lambda d: line_cov.get(d, 0))[: args.top]
    print("  least covered: " + ", ".join(f"{d}={line_cov.get(d, 0)}{'(H)' if d in holdout else ''}" for d in need))

    print(f"== games ({len(games)})")
    if games:
        win = sum(g.get("winner") is not None for g in games)
        turns = [g.get("end_turn", 0) for g in games]
        ints = [x for g in games for x in g.get("interrupts", [])]
        used = sum(1 for x in ints if x.get("card") not in ("-", None))
        opened = sum(1 for g in games if g.get("opening"))
        in_games: Counter = Counter()
        for g in games:
            for d in g.get("decks", []):
                in_games[d] += 1
        print(f"  with winner {win} ({win / len(games):.0%}), mean end turn {statistics.mean(turns):.1f}, "
              f"going-second openings {opened}")
        print(f"  interruption asks {len(ints)} (activated {used}, passed {len(ints) - used})")
        print(f"  decks seen in games: {len(in_games)} / {len(pool) - len(holdout & set(pool))} train decks")
    judged = list((DEMOS / "games_judged").glob("*.jsonl")) if (DEMOS / "games_judged").exists() else []
    if judged:
        scores = [int(t["score"]) for f in judged for j in load(f) for t in j.get("turns", {}).values()
                  if str(t.get("score", "")).isdigit()]
        if scores:
            dist = Counter(scores)
            print(f"== judge: {len(scores)} turns, mean {statistics.mean(scores):.2f}, "
                  + " ".join(f"{s}:{dist[s]}" for s in range(1, 6)))


if __name__ == "__main__":
    main()
