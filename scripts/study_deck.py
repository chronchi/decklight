"""Study a deck list on demand: play bot games, score them, store its primer.

Usage: study_deck.py --ydk <file> --out-dir data/primers/studied [--procs 3] [--games 32] [--no-score]

The deck becomes ``study-<first 10 hex of its hash>``; its games are raw
product-policy games (scripts/search_match.py, ``--raw 1``) against 8 fixed
pool opponents, both seats. Phase A writes a frequency primer for the exact
list; phase B (unless ``--no-score``) re-scores the same games with the
product value net into an impact primer. Progress goes to stdout (one line
per milestone) and to ``<out-dir>/<name>.log``. Fails loud: exit non-zero on
0 games, replay problems, or scoring errors.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

STUDY_OPPONENTS = (
    "mitsurugi-meta1",
    "snake-eye",
    "tearlaments",
    "blue-eyes",
    "kashtira",
    "labrynth",
    "yubel",
    "sky-striker-meta1",
)
SEEDS_PER_PAIR = 2


def fail(msg: str) -> int:
    print(f"ALERT study_deck: {msg}", flush=True)
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ydk", type=Path, required=True, help="deck list to study")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "primers" / "studied")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--no-score", action="store_true", help="stop after the frequency primer")
    args = ap.parse_args(argv)

    import decklight.analysis.primer as P
    from decklight.analysis.primer import aggregate
    from decklight.decks import parse_ydk

    P.MIN_GAMES = 1  # studied lists are served from few games; lookup still uses MIN_GAMES

    try:
        text = args.ydk.read_text(encoding="utf-8")
    except OSError as e:
        return fail(f"cannot read {args.ydk}: {e}")
    try:
        deck = parse_ydk(text, "study")
    except ValueError as e:
        return fail(f"bad deck {args.ydk}: {e}")
    h = P.hash_of(deck)
    name = f"study-{h[:10]}"

    decks_dir = args.out_dir / "decks"
    decks_dir.mkdir(parents=True, exist_ok=True)
    # Replay workers resolve the studied deck through deck_path (events.py):
    # point them at this out-dir's decks (inherited across fork).
    os.environ["YGO_STUDIED_DECKS"] = str(decks_dir)
    (decks_dir / f"{name}.ydk").write_text(text, encoding="utf-8")
    study_dir = args.out_dir / name
    games_out = study_dir / "games"
    games_out.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"{name}.log"

    def say(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    from decklight.search.product import resolve_product_paths

    try:
        ckpt_p, _, _ = resolve_product_paths()
    except (OSError, ValueError) as e:
        return fail(str(e))

    pairs = ",".join(f"{name}:{o},{o}:{name}" for o in STUDY_OPPONENTS)
    deal_seed0 = 30000 + int(h[:6], 16) * 64
    total = min(args.games, len(STUDY_OPPONENTS) * 2 * SEEDS_PER_PAIR)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "search_match.py"),
        "--ckpt",
        str(ckpt_p),
        "--out",
        str(games_out),
        "--pairs",
        pairs,
        "--deck-dir",
        str(decks_dir),
        "--raw",
        "1",
        "--opp",
        "self",
        "--s-seat-only",
        "0",
        "--seeds",
        str(SEEDS_PER_PAIR),
        "--seed-per-pair",
        "1",
        "--deal-seed0",
        str(deal_seed0),
        "--games-limit",
        str(args.games),
        "--nproc",
        str(args.procs),
    ]
    procs = []
    for i in range(args.procs):
        procs.append(subprocess.Popen([*cmd, "--proc", str(i)], cwd=ROOT))
    shown = -1
    while any(p.poll() is None for p in procs):
        n = _count_games(games_out)
        if n != shown:
            shown = n
            say(f"STUDY games {n}/{total}")
        time.sleep(2)
    bad = [p.returncode for p in procs if p.returncode]
    if bad:
        say(f"STUDY failed: search_match exited {bad}")
        return fail(f"search_match exited {bad}")
    n = _count_games(games_out)
    if n != shown:
        say(f"STUDY games {n}/{total}")
    if n == 0:
        say("STUDY failed: 0 games played")
        return fail("0 games played")

    from decklight.engine.resources import load_cards
    from decklight.web.view import DB

    cards, names = load_cards(), DB.names()
    files = sorted(games_out.glob("games-*.jsonl"))
    recs = [json.loads(x) for f in files for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not recs:
        say("STUDY failed: 0 games played")
        return fail("0 games played")

    # Phase A: frequency primer (no scoring).
    summaries = _summarize(files, procs=args.procs, score=False, device="cpu", say=say)
    if summaries is None:
        say("STUDY failed: frequency replay had errors")
        return 2
    store = aggregate(summaries, cards, names)
    entry = store["decks"].get(h)
    if entry is None:
        say("STUDY failed: studied deck has no games")
        return fail("studied deck has no games")
    entry["study"] = {"phase": "games", "games": len(recs)}
    _merge(args.out_dir / "primers.json", {h: entry}, store.get("negated_by", {}))
    (study_dir / "summaries.json.gz").write_bytes(
        gzip.compress(json.dumps(summaries, separators=(",", ":")).encode(), 6)
    )
    say("STUDY phase games done")

    if not args.no_score:
        # Phase B: the build_primers --score pass over this study's games only.
        scored = _summarize(
            files, procs=args.procs, score=True, device="cpu", say=say, scored_total=len(recs)
        )
        if scored is None:
            say("STUDY failed: scoring had errors")
            return 2
        store2 = aggregate(scored, cards, names)
        entry2 = store2["decks"].get(h)
        if entry2 is None:
            say("STUDY failed: studied deck has no scored games")
            return fail("studied deck has no scored games")
        entry2["study"] = {"phase": "scored", "games": len(recs)}
        _merge(args.out_dir / "primers.json", {h: entry2}, store2.get("negated_by", {}))
        (study_dir / "summaries-scored.json.gz").write_bytes(
            gzip.compress(json.dumps(scored, separators=(",", ":")).encode(), 6)
        )
        for f in games_out.glob("games-*.jsonl"):
            f.unlink()
    say("STUDY done")
    return 0


def _count_games(games_out: Path) -> int:
    n = 0
    for f in sorted(games_out.glob("games-*.jsonl")):
        try:
            with f.open("rb") as fh:
                n += sum(1 for line in fh if line.strip())
        except OSError:
            continue
    return n


def _summarize(
    files: list[Path], *, procs: int, score: bool, device: str, say, scored_total: int = 0
) -> list[dict] | None:
    """Replay every games file with build_primers._summarize_file (reused, not copied).

    Returns the summaries, or None (after an ALERT) when a record failed to
    replay or a scored game has a bad/missing win chance. Scoring progress is
    reported per finished file as cumulative games.
    """
    from build_primers import _summarize_file

    jobs = [(str(f), 0, True, device) for f in files] if score else [(str(f), 0) for f in files]
    summaries: list[dict] = []
    problems: list[str] = []
    n_read = 0
    n_bad_w = 0
    done = 0
    with ProcessPoolExecutor(max_workers=min(procs, len(jobs))) as ex:
        for fname, s_list, probs, n, bad_w in ex.map(_summarize_file, jobs):
            summaries.extend(s_list)
            n_read += n
            n_bad_w += bad_w
            for pr in probs:
                tag = "ALERT" if not score else "note"
                print(f"{tag} build_primers: {Path(fname).name}: {pr}", flush=True)
            problems.extend(f"{Path(fname).name}: {pr}" for pr in probs)
            if score:
                done += len(s_list)
                say(f"STUDY scored {done}/{scored_total}")
    if score and n_bad_w:
        print(
            f"ALERT study_deck: {n_bad_w} scored games have bad/missing w (NaN or outside [0, 1])",
            flush=True,
        )
        return None
    if problems and (not score or len(problems) / max(1, n_read) > 0.01):
        print(f"ALERT study_deck: {len(problems)} records failed to replay", flush=True)
        return None
    if n_read == 0:
        print("ALERT study_deck: 0 games read", flush=True)
        return None
    return summaries


def _merge(path: Path, decks: dict, negated_by: dict) -> None:
    """Merge studied entries into primers.json (atomic temp + os.replace)."""
    try:
        cur = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cur = {}
    cur.setdefault("decks", {}).update(decks)
    neg = cur.setdefault("negated_by", {})
    for k, victims in negated_by.items():
        slot = neg.setdefault(k, {})
        for v, n in victims.items():
            slot[v] = slot.get(v, 0) + n
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur) + "\n", encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
