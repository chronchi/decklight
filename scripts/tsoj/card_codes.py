#!/usr/bin/env python3
"""The YGO_CARDS_ONLY list for the tsoj-arms book decks (space separated, no newline)."""
from pathlib import Path

here = Path(__file__).resolve().parent
codes: set[int] = set()
for p in sorted((here / "decks").glob("*.ydk")):
    part = None
    for line in p.read_text().splitlines():
        line = line.strip()
        if line in ("#main", "#extra"):
            part = line[1:]
        elif line.startswith(("#", "!")):
            part = None
        elif part and line.isdigit():
            codes.add(int(line))
print(" ".join(str(c) for c in sorted(codes)), end="")
