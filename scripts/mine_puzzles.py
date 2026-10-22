"""Mine "Win this turn" puzzles from game records and check them by search.

Mining is local (replay only); verification runs the puzzle as a practice
fast-mode DuelSession with the product bot answering live. Laptop limit:
verify at most 15 candidates locally (``--max-verify 15``); the full run is a
vast job (``--shard i/n`` splits candidates across shards, each writing its own
``--out`` file). Fails loud: exit 2 on 0 candidates or 0 verified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    ROOT / "runs" / "value-p2-20261015-2044" / "hppo-p",
    ROOT / "runs" / "value-p2-20261015-2044" / "hppo-p2",
]
COMMIT_COPY = ROOT / "src" / "decklight" / "web" / "puzzles.json"
COMMIT_LIMIT = 300 * 1024


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", type=Path, default=None)
    ap.add_argument("--pattern", default="games-*.jsonl")
    ap.add_argument("--max-games", type=int, default=0, help="cap records scanned (0 = all)")
    ap.add_argument("--max-verify", type=int, default=15, help="cap verifications (laptop limit)")
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--shard", default=None, help="i/n: verify only every n-th candidate from i")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "puzzles" / "puzzles-smoke.json")
    ap.add_argument("--commit-copy", action="store_true")
    ap.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="verify with ProductLauncher(fast=True) (default: on)",
    )
    ap.add_argument("--no-fast", dest="fast", action="store_false")
    return ap.parse_args(argv)


def read_records(dirs, pattern, max_games):
    recs: list[dict] = []
    files: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            print(f"ALERT mine_puzzles: games dir missing: {d}", flush=True)
            return None
        files += sorted(d.glob(pattern))
    if not files:
        print(f"ALERT mine_puzzles: no {pattern} in {[str(d) for d in dirs]}", flush=True)
        return None
    for f in files:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            if max_games and len(recs) >= max_games:
                break
            recs.append(json.loads(line))
        if max_games and len(recs) >= max_games:
            break
    return recs


def main(argv=None) -> int:
    from decklight.analysis import puzzles as P
    from decklight.engine.duel import Engine

    args = parse_args(argv)
    recs = read_records(args.games or DEFAULT_DIRS, args.pattern, args.max_games)
    if recs is None or not recs:
        print("ALERT mine_puzzles: 0 games read", flush=True)
        return 2

    engine = Engine()
    try:
        candidates, stats = P.mine_candidates(engine, recs)
    except Exception as e:  # noqa: BLE001 -- fail loud with the record id
        print(f"ALERT mine_puzzles: mining failed: {type(e).__name__}: {e}", flush=True)
        return 2
    print(f"mined: {len(candidates)} candidates from {len(recs)} games {stats}", flush=True)
    if not candidates:
        print("ALERT mine_puzzles: 0 candidates", flush=True)
        return 2

    # Variety order over ALL candidates; verification stops at --target verified. (Trimming to --target before
    # verifying tried only 4 of 50-90 candidates per file in mine-puzzles-20261018-0321.)
    ordered = P.select(candidates, target=len(candidates))
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            print(f"ALERT mine_puzzles: bad --shard {args.shard!r} (want i/n)", flush=True)
            return 2
        ordered = ordered[i::n]
        print(f"shard {i}/{n}: {len(ordered)} candidates", flush=True)

    from decklight.edopro.product_bot import ProductLauncher

    launcher = ProductLauncher(device="cpu", fast=args.fast, replay_threads=1)
    verified: list[dict] = []
    reasons: Counter = Counter()
    secs: list[float] = []
    naive_rates: list[float] = []
    budget = args.max_verify if args.max_verify > 0 else len(ordered)
    for c in ordered[:budget]:
        t0 = time.monotonic()
        try:
            ok, reason = P.verify_puzzle(engine, c, launcher)
        except Exception as e:  # noqa: BLE001 -- verify never raises, but belt and braces
            ok, reason = False, f"verify crashed: {type(e).__name__}: {e}"
        dt = time.monotonic() - t0
        secs.append(dt)
        head = reason.split(":")[0]
        if ok:
            try:
                rate = float(P.naive_win_rate(engine, c, launcher))
            except Exception as e:  # noqa: BLE001 -- a crashed naive check rejects
                ok, reason = False, f"naive check crashed: {type(e).__name__}: {e}"
                head = reason.split(":")[0]
                rate = 1.0
            else:
                c["naive_win_rate"] = round(rate, 3)
                naive_rates.append(rate)
                if rate > P.NAIVE_KEEP_MAX:
                    ok, reason = False, f"too easy: naive_win_rate {rate:.3f} > {P.NAIVE_KEEP_MAX}"
                    head = reason.split(":")[0]
        if ok:
            c["verified"] = True
            c["game_seed"] = c["seed"]
            c["ws_seed"] = c["seed"]
            verified.append(c)
        reasons[("verified" if ok else "rejected") + ":" + head] += 1
        print(f"verify {c['id'][:60]}: {'OK' if ok else 'NO'} ({dt:.1f}s) {reason}", flush=True)
        if len(verified) >= args.target:
            break
    avg = sum(secs) / len(secs) if secs else 0.0
    if naive_rates:
        from collections import Counter as _C

        buckets = _C(f"{r:.3f}" for r in sorted(naive_rates))
        print(f"naive_win_rate dist over {len(naive_rates)} verified: {dict(buckets)}", flush=True)
    print(
        f"puzzles: {len(candidates)} candidates, {len(verified)} verified "
        f"({avg:.1f}s avg per candidate), reasons {dict(reasons)}",
        flush=True,
    )
    if not verified:
        print("ALERT mine_puzzles: 0 verified", flush=True)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verified) + "\n", encoding="utf-8")
    if args.commit_copy:
        stripped = [{k: v for k, v in p.items() if k != "solution_steps"} for p in verified]
        payload = json.dumps(stripped)
        if len(payload.encode()) <= COMMIT_LIMIT:
            COMMIT_COPY.write_text(payload + "\n", encoding="utf-8")
            print(f"puzzles: committed copy -> {COMMIT_COPY}", flush=True)
        else:
            print(
                f"ALERT mine_puzzles: payload {len(payload.encode())} B > 300 KB, committed copy skipped",
                flush=True,
            )
    if verified:
        print("--- sample ---", flush=True)
        print(P.render_text(engine, verified[0]), flush=True)
    print(
        "full-run command: taskset -c 0-3 uv run python scripts/mine_puzzles.py "
        "--max-verify 0 --target 60 --out data/puzzles/puzzles.json "
        "(vast: one job per --shard i/8, then concat the outs)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
