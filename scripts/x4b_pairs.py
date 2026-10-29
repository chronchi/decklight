"""X4b: search_match --pairs files from decks/x4b/split.json (scripts/x4b_lists.py).

Every pair is `<S deck>:<opponent>` over the 8 fixed opponents (O side = P raw). Writes decks/x4b/pairs/:
  arch-<archetype>.pairs   its 3 train lists + its held-out list (the specialist card, H2)
  arch-all.pairs           the 5 archetypes' lists together (P's arm and the student's, one run)
  related.pairs, unrelated.pairs
  ga-arch-<archetype>.pairs, ga-related.pairs, ga-unrelated.pairs: the same S lists vs each of
    ygo-agent's gauntlet lists (search_match --opp ygoagent: ygo-agent pilots the gauntlet list)
search_match resolves names through --deck-dir (the job flattens decks/x4b/**/*.ydk into one directory) and then
decks/pool.

    python scripts/x4b_pairs.py
"""

from __future__ import annotations

import json
from pathlib import Path

X4B = Path(__file__).resolve().parents[1] / "decks" / "x4b"


def main() -> None:
    split = json.loads((X4B / "split.json").read_text())
    opps = split["opponents"]
    out = X4B / "pairs"
    out.mkdir(exist_ok=True)

    def write(name: str, decks: list[str]) -> None:
        (out / f"{name}.pairs").write_text("".join(f"{d}:{o}\n" for d in decks for o in opps))
        print(f"{name}: {len(decks)} decks x {len(opps)} opponents = {len(decks) * len(opps)} pairs")

    every = []
    for arch, train in split["train"].items():
        decks = [*train, split["heldout"][arch]]
        write(f"arch-{arch}", decks)
        every += decks
    write("arch-all", every)
    write("related", split["related"])
    write("unrelated", split["unrelated_fresh"] + split["unrelated_holdout"])
    opps = sorted(p.stem for p in (X4B.parent / "gauntlet").glob("*.ydk"))
    for arch, train in split["train"].items():  # per archetype: P, D and its specialist pair on one file's seeds
        write(f"ga-arch-{arch}", [*train, split["heldout"][arch]])
    write("ga-related", split["related"])
    write("ga-unrelated", split["unrelated_fresh"] + split["unrelated_holdout"])


if __name__ == "__main__":
    main()
