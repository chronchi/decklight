"""Build opponent deck primers from bot-vs-bot game records.

Reads ``games-*.jsonl`` files, aggregates per-deck primers (see
:mod:`ygo_zero.analysis.primer`) and writes ``primers.json``. Fails loud:
exit 2 when 0 games were read or a record fails to replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    ROOT / "runs" / "value-p2-20261015-2044" / "hppo-p",
    ROOT / "runs" / "value-p2-20261015-2044" / "hppo-p2",
]
COMMIT_COPY = ROOT / "src" / "ygo_zero" / "web" / "primers.json"
COMMIT_LIMIT = 300 * 1024


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--games",
        nargs="*",
        type=Path,
        default=None,
        help="game-record dirs (default: the value-p2 hppo-p/hppo-p2 dirs)",
    )
    ap.add_argument("--pattern", default="games-*.jsonl")
    ap.add_argument("--max-games", type=int, default=0, help="cap games read (0 = all)")
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument(
        "--archetype", action="append", default=[], help="manual archetype override HASH:NAME (repeatable)"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "primers" / "primers.json")
    ap.add_argument("--commit-copy", action="store_true", help=f"also write {COMMIT_COPY} when <= 300 KB")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    import ygo_zero.analysis.primer as P
    from ygo_zero.analysis.primer import (  # noqa: F401  # noqa: F401
        MIN_GAMES,
        _summarize_game,
        aggregate,
        load_store,
        names_of,
    )
    from ygo_zero.engine.duel import Engine
    from ygo_zero.engine.resources import load_cards
    from ygo_zero.web.view import DB

    args = parse_args(argv)
    overrides: dict[str, str] = {}
    for item in args.archetype:
        h, _, name = item.partition(":")
        if not h or not name:
            print(f"ALERT build_primers: bad --archetype {item!r} (want HASH:NAME)", flush=True)
            return 2
        overrides[h] = name
    P.MIN_GAMES = args.min_games

    dirs = args.games or DEFAULT_DIRS
    files: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            print(f"ALERT build_primers: games dir missing: {d}", flush=True)
            return 2
        files += sorted(d.glob(args.pattern))
    if not files:
        print(f"ALERT build_primers: no {args.pattern} in {[str(d) for d in dirs]}", flush=True)
        return 2

    engine = Engine()
    cards = load_cards()
    names = DB.names()
    summaries: list[dict] = []
    problems: list[str] = []
    n_read = 0
    for f in files:
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            if args.max_games and n_read >= args.max_games:
                break
            n_read += 1
            try:
                rec = json.loads(line)
                summaries.append(_summarize_game(engine, rec, cards, names))
            except Exception as e:  # noqa: BLE001 -- fail loud below with the record id
                problems.append(f"{f.name}: {type(e).__name__}: {e}")
        if args.max_games and n_read >= args.max_games:
            break
    for p in problems[:10]:
        print(f"ALERT build_primers: {p}", flush=True)
    if problems:
        print(f"ALERT build_primers: {len(problems)} records failed to replay", flush=True)
        return 2
    if n_read == 0:
        print("ALERT build_primers: 0 games read", flush=True)
        return 2

    store = aggregate(summaries, cards, names, overrides)
    payload = json.dumps(store)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload + "\n", encoding="utf-8")

    # Coverage over the lobby decks (pool + newcards).
    lobby = sorted((ROOT / "decks" / "pool").glob("*.ydk")) + sorted(
        (ROOT / "decks" / "newcards").glob("*.ydk")
    )
    from ygo_zero.decks import load_ydk

    covered = uncovered = 0
    for p in lobby:
        h = P.hash_of(load_ydk(p))
        d = store["decks"].get(h)
        if d is not None and d["games"] >= args.min_games:
            covered += 1
        else:
            uncovered += 1
    print(
        f"primers: {n_read} games -> {len(store['decks'])} decks, "
        f"lobby coverage {covered}/{covered + uncovered} "
        f"(pool+newcards, >= {args.min_games} games)",
        flush=True,
    )
    if uncovered:
        print(
            "full-run command: "
            "taskset -c 0-3 uv run python scripts/build_primers.py "
            "--out data/primers/primers.json --commit-copy",
            flush=True,
        )
    if args.commit_copy:
        if len(payload.encode()) <= COMMIT_LIMIT:
            COMMIT_COPY.write_text(payload + "\n", encoding="utf-8")
            print(f"primers: committed copy -> {COMMIT_COPY}", flush=True)
        else:
            print(
                f"ALERT build_primers: payload {len(payload.encode())} B > 300 KB, committed copy skipped",
                flush=True,
            )
    _ = names_of
    return 0


if __name__ == "__main__":
    sys.exit(main())
