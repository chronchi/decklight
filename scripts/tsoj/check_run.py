#!/usr/bin/env python3
"""Post-run invariants for a tsoj-arms run; append problems to ALERT and exit non-zero."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--arm", default="p")
    ap.add_argument("--alert", required=True)
    args = ap.parse_args()
    recs = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line.strip()]
    problems: list[str] = []
    if not recs:
        problems.append("0 records")
    finished = 0
    retried = 0
    bad = 0
    for i, r in enumerate(recs):
        res = r.get("result") or {}
        reason = res.get("reason")
        if res.get("turns") is not None:
            finished += 1
        if isinstance(reason, str) and ("bot_error" in reason or "illegal_response" in reason):
            bad += 1
            problems.append(f"game {i}: bad result {reason!r}")
        if int(res.get("retries") or 0) > 0:
            retried += 1
    if recs and finished == 0:
        problems.append("0 finished games")
    if recs and retried / len(recs) > 0.02:
        problems.append(f"retries in {retried}/{len(recs)} games (> 2%)")
    if args.arms and args.arm != "p":
        rows = [json.loads(x) for x in Path(args.arms).read_text().splitlines() if x.strip()]
        kind, _, kraw = args.arm.partition("@")
        k = int(kraw) if kraw.isdigit() else None
        rows = [x for x in rows if x.get("arm") == kind and (k is None or x.get("k") == k)]
        forced = sum(bool(x.get("forced")) for x in rows)
        share = forced / len(rows) if rows else 0.0
        print(f"forced share: {forced}/{len(rows)} = {share:.3f}")
        if share < 0.5:
            problems.append(f"forced share {share:.3f} < 0.50")
    alert = Path(args.alert)
    if problems:
        with alert.open("a", encoding="utf-8") as f:
            for p in problems:
                f.write(f"ALERT tsoj-arms {args.arm}: {p}\n")
        print("\n".join(problems), file=sys.stderr)
        return 3
    print(f"ok: {len(recs)} games, {finished} finished, {retried} retried, {bad} bad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
