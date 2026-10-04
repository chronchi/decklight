"""Export hand-played gold games to one JSONL row per gold decision.

Reads data/demos/gold/*.jsonl, replays each record (seed + decks + response list), and writes
data/demos/gold_decisions.jsonl: one row per decision whose label is `gold` (real choices; `auto`
zone answers and `pass` window passes are skipped).

Row: {game_id, index, actor, deck, turn, msg, state, chain, options, choice, why}; `why` is the note
whose start index is the latest one <= index within the same turn and seat (notes are
[start_index, text] pairs in the record).

    uv run python scripts/export_gold.py [--out data/demos/gold_decisions.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hand_play

from ygo_zero.demos.dataset import record_config
from ygo_zero.demos.names import CardNames
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import Engine

GOLD = ROOT / "data" / "demos" / "gold"


def _why(notes: list, turns: list[int], actors: list[int], k: int) -> str:
    """The note for decision `k`: latest start <= k within the same turn and seat, else ""."""
    best, text = -1, ""
    for s, t in notes:
        if int(s) <= k and int(s) > best and turns[int(s)] == turns[k] and actors[int(s)] == actors[k]:
            best, text = int(s), str(t)
    return text


def export_game(engine: Engine, names: CardNames, rec: dict, out_f) -> int:
    env = CppEnv(engine, record_config(rec), rec.get("auto_single", True))
    obs = env.reset()
    seen: set[int] = set()
    n = 0
    choices, labels = rec["choices"], rec["labels"]
    turns, actors, decks, notes = rec["turns"], rec["actors"], rec["decks"], rec.get("notes", [])
    for k, (choice, label) in enumerate(zip(choices, labels, strict=True)):
        if obs is None:
            print(f"  !! {rec['game_id']}: duel ended at choice {k}/{len(choices)}")
            break
        if label == "gold":
            d = hand_play.decision_json(env, obs, k, decks[actors[k]], names, seen)
            row = {
                "game_id": rec["game_id"],
                "index": d["index"],
                "actor": d["actor"],
                "deck": d["deck"],
                "turn": d["turn"],
                "msg": d["msg"],
                "hint": d["hint"],
                "state": d["state"],
                "chain": d["chain"],
                "options": d["options"],
                "choice": choice,
                "why": _why(notes, turns, actors, k),
            }
            out_f.write(json.dumps(row) + "\n")
            n += 1
        obs = env.step(int(choice))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "demos" / "gold_decisions.jsonl"))
    a = ap.parse_args()
    engine, names = Engine(), CardNames()
    total = 0
    with open(a.out, "w") as f:
        for path in sorted(GOLD.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n = export_game(engine, names, rec, f)
                print(f"{rec['game_id']} ({path.name}): {n} gold decisions")
                total += n
    print(f"total: {total} rows -> {a.out}")


if __name__ == "__main__":
    main()
