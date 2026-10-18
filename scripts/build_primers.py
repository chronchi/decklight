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
# Committed copy: compact gzip without the per-deck file lists (the full store is ~840 KB of JSON for 220 decks;
# gzip brings it to ~110 KB). The web server reads this file.
COMMIT_COPY = ROOT / "src" / "ygo_zero" / "web" / "primers.json.gz"
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
        "--summaries",
        type=Path,
        default=ROOT / "data" / "primers" / "summaries.json.gz",
        help="per-game summaries cache: written after replaying, read instead of replaying with --from-summaries",
    )
    ap.add_argument(
        "--from-summaries",
        action="store_true",
        help="aggregate the --summaries cache (no replay, ~1 min vs ~50)",
    )
    ap.add_argument(
        "--archetype", action="append", default=[], help="manual archetype override HASH:NAME (repeatable)"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "primers" / "primers.json")
    ap.add_argument(
        "--commit-copy", action="store_true", help=f"also write {COMMIT_COPY} (gzip) when <= 300 KB"
    )
    return ap.parse_args(argv)


_W: dict = {}


def _summarize_file(job: tuple[str, int]) -> tuple[str, list[dict], list[str], int]:
    """Worker: summarize every record of one games file (engine and card tables built once per process)."""
    from ygo_zero.analysis.primer import _summarize_game

    if not _W:
        from ygo_zero.engine.duel import Engine
        from ygo_zero.engine.resources import load_cards
        from ygo_zero.web.view import DB

        _W.update(engine=Engine(), cards=load_cards(), names=DB.names())
    fname, cap = job
    out: list[dict] = []
    probs: list[str] = []
    n = 0
    for line in Path(fname).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if cap and n >= cap:
            break
        n += 1
        try:
            rec = json.loads(line)
            out.append(_summarize_game(_W["engine"], rec, _W["cards"], _W["names"]))
        except Exception as e:  # noqa: BLE001 -- reported by the parent with the file name
            probs.append(f"game {n - 1}: {type(e).__name__}: {e}")
    return fname, out, probs, n


def main(argv=None) -> int:
    import ygo_zero.analysis.primer as P
    from ygo_zero.analysis.primer import (  # noqa: F401  # noqa: F401
        MIN_GAMES,
        _summarize_game,
        aggregate,
        load_store,
        names_of,
    )
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

    import gzip

    if args.from_summaries:
        if not args.summaries.is_file():
            print(f"ALERT build_primers: no summaries cache {args.summaries}", flush=True)
            return 2
        summaries = json.loads(gzip.decompress(args.summaries.read_bytes()))
        n_read = len(summaries)
        if n_read == 0:
            print("ALERT build_primers: 0 games in the summaries cache", flush=True)
            return 2
    else:
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

        # One process per games file (4 at most: the laptop's core cap); problems are printed as each file finishes, so a
        # replay bug shows in minutes, not at the end (the first full build ran 112 min serially, then failed on 7 games).
        from concurrent.futures import ProcessPoolExecutor

        # --max-games (smoke): the first 4 files, each capped, then trimmed to the budget in order
        jobs = [(str(f), args.max_games) for f in (files[:4] if args.max_games else files)]
        summaries: list[dict] = []
        problems: list[str] = []
        n_read = 0
        with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as ex:
            for fname, s_list, probs, n in ex.map(_summarize_file, jobs):
                if args.max_games and n_read >= args.max_games:
                    continue
                take = s_list[: args.max_games - n_read] if args.max_games else s_list
                summaries.extend(take)
                n_read += n if not args.max_games else min(n, args.max_games - n_read)
                for pr in probs:
                    print(f"ALERT build_primers: {Path(fname).name}: {pr}", flush=True)
                problems.extend(f"{Path(fname).name}: {pr}" for pr in probs)
        if problems:
            print(f"ALERT build_primers: {len(problems)} records failed to replay", flush=True)
            return 2
        if n_read == 0:
            print("ALERT build_primers: 0 games read", flush=True)
            return 2
        if not args.max_games:
            args.summaries.parent.mkdir(parents=True, exist_ok=True)
            args.summaries.write_bytes(
                gzip.compress(json.dumps(summaries, separators=(",", ":")).encode(), 6)
            )
            print(f"primers: {n_read} summaries -> {args.summaries}", flush=True)

    cards = load_cards()
    names = DB.names()
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
        import gzip

        slim = {
            **store,
            "decks": {h: {k: v for k, v in e.items() if k != "files"} for h, e in store["decks"].items()},
        }
        blob = gzip.compress(json.dumps(slim, separators=(",", ":")).encode(), 9, mtime=0)
        if len(blob) <= COMMIT_LIMIT:
            COMMIT_COPY.write_bytes(blob)
            print(f"primers: committed copy {len(blob)} B -> {COMMIT_COPY}", flush=True)
        else:
            print(
                f"ALERT build_primers: gzip copy {len(blob)} B > 300 KB, committed copy skipped", flush=True
            )
            return 2
    _ = names_of
    return 0


if __name__ == "__main__":
    sys.exit(main())
