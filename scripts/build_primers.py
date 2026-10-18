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
        "--score",
        action="store_true",
        help="score every game with the product value net (win chance per step -> impact-ranked primers)",
    )
    ap.add_argument("--procs", type=int, default=4, help="replay workers (default 4 = laptop cap)")
    ap.add_argument(
        "--device",
        default=None,
        help="score device: cpu|cuda (default: cuda if available, else cpu)",
    )
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


def _summarize_file(job: tuple) -> tuple[str, list[dict], list[str], int, int]:
    """Worker: summarize every record of one games file (engine and card tables built once per process)."""
    from ygo_zero.analysis.primer import _summarize_game

    if len(job) == 2:
        fname, cap = job
        score, device = False, "cpu"
    else:
        fname, cap, score, device = job
    if not _W:
        from ygo_zero.engine.duel import Engine
        from ygo_zero.engine.resources import load_cards
        from ygo_zero.web.view import DB

        _W.update(engine=Engine(), cards=load_cards(), names=DB.names())
    value_fn = calib = None
    if score and "value_fn" not in _W:
        from ygo_zero.eval.winchance import load_calib
        from ygo_zero.search.product import load_value, resolve_product_paths

        _, value_p, emb_p = resolve_product_paths()
        _W["value_fn"] = load_value(value_p, emb_p, device=device)
        _W["calib"] = load_calib()
    if score:
        value_fn, calib = _W["value_fn"], _W["calib"]
    fname = str(fname)
    out: list[dict] = []
    probs: list[str] = []
    bad_w = 0
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
            s = _summarize_game(_W["engine"], rec, _W["cards"], _W["names"], value_fn, calib)
            if score and (
                sum(1 for st in s["steps"] if "w" in st) < len(s["steps"])
                or "w_end" not in s
                or any(not 0.0 <= float(st["w"]) <= 1.0 or st["w"] != st["w"] for st in s["steps"])
            ):
                bad_w += 1
            else:
                out.append(s)
        except Exception as e:  # noqa: BLE001 -- reported by the parent with the file name
            probs.append(f"game {n - 1}: {type(e).__name__}: {e}")
    return fname, out, probs, n, bad_w


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

        # One process per games file (the laptop's core cap by default); problems are printed as each file finishes, so a
        # replay bug shows in minutes, not at the end (the first full build ran 112 min serially, then failed on 7 games).
        from concurrent.futures import ProcessPoolExecutor

        device = args.device
        if args.score and device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        # --max-games (smoke): the first 4 files, each capped, then trimmed to the budget in order
        jobs = [(str(f), args.max_games) for f in (files[:4] if args.max_games else files)]
        if args.score:
            jobs = [(f, c, True, device) for f, c in jobs]
            print(f"primers: scoring with the product value net on {device}", flush=True)
        summaries: list[dict] = []
        problems: list[str] = []
        n_read = 0
        n_bad_w = 0
        with ProcessPoolExecutor(max_workers=min(args.procs, len(jobs))) as ex:
            for fname, s_list, probs, n, bad_w in ex.map(_summarize_file, jobs):
                print(f"PROGRESS {Path(fname).name} {len(s_list)}", flush=True)
                if args.max_games and n_read >= args.max_games:
                    continue
                take = s_list[: args.max_games - n_read] if args.max_games else s_list
                summaries.extend(take)
                n_read += n if not args.max_games else min(n, args.max_games - n_read)
                n_bad_w += bad_w
                for pr in probs:
                    tag = "ALERT" if not args.score else "note"
                    print(f"{tag} build_primers: {Path(fname).name}: {pr}", flush=True)
                problems.extend(f"{Path(fname).name}: {pr}" for pr in probs)
        if args.score and n_bad_w:
            print(
                f"ALERT build_primers: {n_bad_w} scored games have bad/missing w (NaN or outside [0, 1])",
                flush=True,
            )
            return 2
        if problems and (not args.score or len(problems) / max(1, n_read) > 0.01):
            print(f"ALERT build_primers: {len(problems)} records failed to replay", flush=True)
            return 2
        if problems:
            print(
                f"build_primers: {len(problems)} records failed (within the 1% budget, dropped)",
                flush=True,
            )
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
    if args.score:
        _print_impact_sanity(summaries, store, names)
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


def _print_impact_sanity(summaries: list[dict], store: dict, names: dict) -> None:
    """Sanity numbers for a scored run: impact coverage, top cards, credit scale."""
    import math

    decks = store.get("decks", {})
    n_impact = sum(
        1 for d in decks.values() if d.get("key_cards") and any("impact" in c for c in d["key_cards"])
    )
    top: list[tuple[float, str, str]] = []
    for h, d in decks.items():
        for c in d.get("key_cards", []):
            if "impact" in c:
                top.append((c["impact"], c.get("name", f"#{c['code']}"), d.get("archetype", h[:8])))
    top.sort(key=lambda t: -t[0])
    print(f"impact: {n_impact} decks with impact key cards", flush=True)
    for v, nm, arch in top[:10]:
        print(f"impact: {v:+.1f} pts/game {nm} ({arch})", flush=True)
    play_abs = play_n = var_play = var_all = 0.0
    for s in summaries:
        steps = s.get("steps", [])
        if not steps or not all("w" in st for st in steps) or "w_end" not in s:
            continue
        W = [float(st["w"]) for st in steps] + [float(s["w_end"])]
        if any(math.isnan(v) or not 0.0 <= v <= 1.0 for v in W):
            continue
        from ygo_zero.analysis.primer import _play_credits

        for c in _play_credits(steps, float(s["w_end"])).values():
            play_abs += abs(c)
            play_n += 1
            var_play += abs(c)
        var_all += sum(abs(W[i + 1] - W[i]) for i in range(len(steps)))
    mean_abs = play_abs / play_n * 100 if play_n else math.nan
    share = var_play / var_all if var_all else math.nan
    print(f"impact: mean |credit| per play {mean_abs:.2f} pts over {int(play_n)} plays", flush=True)
    print(f"impact: play steps carry {share:.3f} of total |ΔW|", flush=True)


if __name__ == "__main__":
    sys.exit(main())
