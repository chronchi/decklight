"""Write the tsoj-arms opening book (ygo-arbiter format) and print YGO_CARDS_ONLY.

Our decks x their pool decks x N seeds. The arbiter always plays each opening twice with the
bots swapping seats and each seat keeping its deck, so a line's two decks are the two seats:
P holds `ours` in one game of the pair and `opp` in the other. `arms_summary.py` keeps the
games where P held `ours` for the split-by-deck analysis. `--mirror` also writes each line with the
decks reversed (same seed), so P holds every deck in both seats (NOTEBOOK 2026-11-07 gauntlet).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECKS = HERE / "decks"
OURS = ["skydeck-tcg2026-10", "light-magistus-f10-tcg2026-10"]
OPP = [
    "fire-king", "galaxy-eyes", "raise-moon-ygopd", "cyberdark-ygopd",
    "firewall-cyberse-ygopd", "kashtira", "sacred-beasts-meta1",
]  # rocket-warrior-ygopd is out: its 101403017/101403036 are not in ygo-cards' card db


def ydk(path: Path) -> dict:
    deck, part = {"name": path.stem, "main": [], "extra": []}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line in ("#main", "#extra"):
            part = line[1:]
        elif line.startswith(("#", "!")):
            part = None
        elif part and line.isdigit():
            deck[part].append(int(line))
    if not 40 <= len(deck["main"]) <= 60:
        raise SystemExit(f"{path}: {len(deck['main'])} main-deck cards")
    return deck


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "books" / "tsoj.jsonl"))
    ap.add_argument("--seeds", type=int, default=1, help="book lines per (ours, opp) pair")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--mirror", action="store_true", help="also write every line with its decks reversed")
    args = ap.parse_args()
    ours = [ydk(DECKS / f"{n}.ydk") for n in OURS]
    opp = [ydk(DECKS / f"{n}.ydk") for n in OPP]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        # seed-major order: a run cut short (`-n` / a time cap) still covers every deck pair evenly
        for k in range(args.seeds):
            for o in ours:
                for p in opp:
                    seed = args.seed_base + n
                    for pair in ([o, p], [p, o]) if args.mirror else ([o, p],):
                        f.write(json.dumps({"seed": seed, "decks": pair}) + "\n")
                    n += 1
    codes = sorted({c for d in (*ours, *opp) for c in (*d["main"], *d["extra"])})
    print(f"{out}: {n * (2 if args.mirror else 1)} openings, {len(codes)} cards")
    print("YGO_CARDS_ONLY=" + " ".join(map(str, codes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
