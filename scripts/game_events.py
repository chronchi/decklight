"""Per-turn event summaries for stored search_match games (engine replay only, no model).

    taskset -c 0-3 uv run python scripts/game_events.py --games runs/<job>/<variant> \\
        --out data/cache/events.jsonl [--max-games N] [--workers 4]

Reads every ``games-*.jsonl`` in --games (or a single file), replays each record in the
Python reference env and writes one JSON line per game (``game_idx``, ``seed``,
``decks`` plus ``src/decklight/analysis/events.py:events``). Truncated records are
skipped (no full trace). Exit 2 when 0 games were processed or any invariant fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_ENGINE = None


def _worker_init() -> None:
    global _ENGINE
    from decklight.engine.duel import Engine

    _ENGINE = Engine()


def _one(payload: str) -> str:
    from decklight.analysis.events import events

    rec = json.loads(payload)
    if rec.get("truncated"):
        return json.dumps({"skipped": rec.get("game_idx")})
    try:
        out = events(rec, _ENGINE)
    except Exception as e:  # noqa: BLE001 - relayed to the parent, which exits 2
        return json.dumps({"error": f"game {rec.get('game_idx')}: {type(e).__name__}: {e}"})
    out["decks"] = rec["decks"]
    return json.dumps(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, required=True, help="a variant dir or a games-*.jsonl file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = [a.games] if a.games.is_file() else sorted(a.games.glob("games-*.jsonl"))
    payloads = [ln for f in files if f.is_file() for ln in f.read_text().splitlines() if ln.strip()]
    if a.max_games is not None:
        payloads = payloads[: a.max_games]
    if not payloads:
        print("ALERT game_events: no game records found", file=sys.stderr)
        sys.exit(2)
    n_act = n_int = n_neg = n_games = n_skip = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_worker_init) as pool, a.out.open("w") as fh:
        for line in pool.map(_one, payloads, chunksize=1):
            rec = json.loads(line)
            if "skipped" in rec:
                n_skip += 1
                continue
            if "error" in rec:
                print(f"ALERT game_events: {rec['error']}", file=sys.stderr)
                pool.shutdown(cancel_futures=True)
                sys.exit(2)
            fh.write(line + "\n")
            n_games += 1
            n_act += sum(p["activations"] for t in rec["turns"] for p in t["players"])
            n_int += sum(p["activations_opp_turn"] for t in rec["turns"] for p in t["players"])
            n_neg += sum(p["negated_by_p_suffered"] for t in rec["turns"] for p in t["players"])
    if n_games == 0:
        print("ALERT game_events: 0 games processed (all skipped or failed)", file=sys.stderr)
        sys.exit(2)
    print(
        f"{n_games} games ({n_skip} truncated skipped) -> {a.out}\n"
        f"activations per game per seat: {n_act / (2 * n_games):.2f}\n"
        f"interruptions shown per game: {n_int / n_games:.2f}\n"
        f"negations suffered per game: {n_neg / n_games:.2f}"
    )


if __name__ == "__main__":
    main()
