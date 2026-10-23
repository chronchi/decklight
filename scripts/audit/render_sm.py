"""Render search_match game records as readable text (engine replays, no model).

    uv run python scripts/audit/render_sm.py <games.jsonl|arm dir> [--out DIR] [--limit N]

Library: render_record / replay_trace / header_of live in sm_common. The CLI replays every
record and writes one text file per game (<out>/game-<game_idx>.txt); exit non-zero if any
replay does not reproduce the recorded winner (AGENTS.md: fail fast and loud).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sm_common as SM


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="a games-*.jsonl file or an arm dir holding them")
    ap.add_argument("--out", type=Path, default=None, help="write game-<idx>.txt files here (default: stdout)")
    ap.add_argument("--limit", type=int, default=0, help="render at most N games (0 = all)")
    ap.add_argument("--gauntlet", type=Path, default=SM.GAUNTLET)
    args = ap.parse_args(argv)

    recs = SM.iter_records(args.src) if args.src.is_dir() else [
        json.loads(line) for line in args.src.read_text().splitlines() if line.strip()]
    if args.limit:
        recs = recs[: args.limit]
    if not recs:
        print(f"ALERT render_sm: no games in {args.src}", flush=True)
        return 1
    engine, names = SM.Engine(), SM.CardNames()
    bad = 0
    for rec in recs:
        try:
            text, ok = SM.render_record(rec, engine, names, args.gauntlet)
        except Exception as e:  # noqa: BLE001 - report the game, keep going, fail at the end
            print(f"ALERT render_sm: game {rec.get('game_idx')} replay raised {e!r}", flush=True)
            bad += 1
            continue
        if not ok:
            print(f"ALERT render_sm: game {rec.get('game_idx')} replay does not reproduce "
                  f"winner {rec.get('winner')}", flush=True)
            bad += 1
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"game-{rec.get('game_idx')}.txt").write_text(text)
        else:
            print(text)
    if args.out is not None:
        print(f"rendered {len(recs)} games -> {args.out} ({bad} replay mismatches)")
    if bad:
        print(f"ALERT render_sm: {bad}/{len(recs)} replays do not reproduce the recorded winner", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
