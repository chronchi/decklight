"""Merge per-file puzzle shards (scripts/vast_mine_puzzles.sh) into the web's puzzle set.

Reads ``<shards>/*.json`` (lists of verified puzzles), drops duplicate ids, orders them with
:func:`ygo_zero.analysis.puzzles.select` (variety first: round-robin over deck pair and turn, at most 2 each), keeps
``--target``, and writes ``data/puzzles/puzzles.json`` plus the committed copy without ``solution_steps``
(``src/ygo_zero/web/puzzles.json``, at most 300 KB). Fails loud: exit 2 on 0 puzzles or an oversized copy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT_COPY = ROOT / "src" / "ygo_zero" / "web" / "puzzles.json"
COMMIT_LIMIT = 300 * 1024


# Difficulty mix to aim for (hard puzzles are the most common verified kind: 41 of 71 in the first run).
MIX = {"easy": 0.3, "medium": 0.35, "hard": 0.35}


def pick(puzzles: list[dict], target: int) -> list[dict]:
    """Greedy variety: each step takes the puzzle whose decks are used least so far, then whose difficulty is furthest
    below its MIX share, then the lowest naive win rate. (``puzzles.select`` round-robins over (deck pair, turn) groups in
    alphabetical order, so with more groups than the target it only reached early-alphabet decks: 120 picks covered
    30 decks.)"""
    left = sorted(puzzles, key=lambda p: p["id"])
    used: Counter = Counter()
    diff: Counter = Counter()
    out: list[dict] = []
    while left and len(out) < target:
        n = len(out) + 1

        def key(p: dict, n: int = n) -> tuple:
            d = p.get("difficulty", "hard")
            return (
                sum(used[x] for x in set(p["decks"])),
                diff[d] - MIX.get(d, 0.0) * n,
                p.get("naive_win_rate") or 0.0,
                p["id"],
            )

        best = min(left, key=key)
        left.remove(best)
        out.append(best)
        used.update(set(best["decks"]))
        diff[best.get("difficulty", "hard")] += 1
    return out


def main(argv=None) -> int:

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", type=Path, nargs="+", help="shard dirs (several runs merge, deduped by id)")
    ap.add_argument("--target", type=int, default=80)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "puzzles" / "puzzles.json")
    args = ap.parse_args(argv)
    seen: dict[str, dict] = {}
    for f in sorted(f for d in args.shards for f in d.glob("*.json")):
        for p in json.loads(f.read_text(encoding="utf-8")):
            seen.setdefault(p["id"], p)
    if not seen:
        print(f"ALERT merge_puzzles: 0 puzzles in {args.shards}", flush=True)
        return 2
    chosen = pick(list(seen.values()), args.target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(chosen) + "\n", encoding="utf-8")
    payload = json.dumps([{k: v for k, v in p.items() if k != "solution_steps"} for p in chosen])
    if len(payload.encode()) > COMMIT_LIMIT:
        print(f"ALERT merge_puzzles: committed copy {len(payload.encode())} B > 300 KB", flush=True)
        return 2
    COMMIT_COPY.write_text(payload + "\n", encoding="utf-8")
    diff = Counter(p.get("difficulty") for p in chosen)
    decks = Counter(d for p in chosen for d in p["decks"])
    naive = [p.get("naive_win_rate") for p in chosen if p.get("naive_win_rate") is not None]
    print(
        f"puzzles: {len(seen)} verified in shards -> {len(chosen)} chosen; difficulty {dict(diff)}; "
        f"{len(decks)} decks; naive win rate mean {sum(naive) / max(1, len(naive)):.2f}; "
        f"copy {len(payload.encode())} B -> {COMMIT_COPY.relative_to(ROOT)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
