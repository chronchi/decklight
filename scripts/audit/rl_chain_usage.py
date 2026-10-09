"""Our chain-window behaviour per checkpoint: at our CHAIN prompts (≥ 1 activatable card), how often we activate,
and which cards, split by whose turn it is (respond on the opponent's turn = hand traps).

    uv run python scripts/audit/rl_chain_usage.py <run dir> --tags it0 it200 [--opp ga|vb]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from render_games import decisions_lines

from ygo_zero.demos.names import CardNames
from ygo_zero.engine.duel import Engine

LINE = re.compile(r"\[(\d+)\] T(\d+) P(\d) CHAIN (\S+) (.*?) '.*/(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--tags", nargs="+", default=["it0", "it200"])
    args = ap.parse_args()
    names, engine = CardNames(), Engine()
    for tag in args.tags:
        n_games = 0
        stats = {"own": Counter(), "opp": Counter()}
        cards = {"own": Counter(), "opp": Counter()}
        for raw in (args.run / tag / "games.jsonl").read_text().splitlines():
            g = json.loads(raw)
            st = json.loads((args.run / tag / "games" / g["game"] / "state.json").read_text())
            seat = int(st["a_seat"])
            n_games += 1
            for ln in decisions_lines(st, names, engine):
                m = LINE.match(ln)
                if not m or int(m.group(3)) != seat:
                    continue
                turn = int(m.group(2))
                whose = "own" if (turn % 2 == 1) == (seat == 0) else "opp"
                stats[whose]["prompts"] += 1
                if m.group(4) == "ACTIVATE":
                    stats[whose]["activate"] += 1
                    cards[whose][m.group(5).strip()] += 1
        for whose in ("opp", "own"):
            s = stats[whose]
            print(f"{tag} chain windows on {whose} turn: {s['prompts']} prompts, activate {s['activate']} "
                  f"({s['activate'] / max(s['prompts'], 1):.3f}), per game {s['activate'] / n_games:.2f} | "
                  + ", ".join(f"{c} {n}" for c, n in cards[whose].most_common(8)))


if __name__ == "__main__":
    main()
