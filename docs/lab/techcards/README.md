# Tech-card questions, 2026-10-16 (engine plan N4)

Harness: `scripts/tech_ab.py` (paired B-A, exposure splits, built-in A/A check).
Question file: `questions-2026-10-16.json` (25 questions: 5 decks x 5).
Vast run: `scripts/vast_tech_ab.sh` + `scripts/launch/launch-tech-ab.sh`
(pilot = opponent = `data/cache/P/best.pt` raw, no search).

## Deck choice (measured 2026-10-16, `load_ydk` main-deck counts)

Picked 5 gauntlet decks that each run >= 2 copies of a hand trap, with
different engines so a tech verdict is not one-deck luck. (Blackwing runs no
hand trap from the list and was skipped.)

| deck | hand traps in main |
|---|---|
| Branded (46) | Ash x3, Maxx "C" x3, Imperm x3, Called x2 |
| CenturIon (40) | Ash x3, Maxx "C" x3, Veiler x3, Imperm x3, Droll x2, Called x2, Nibiru x1 |
| SnakeEyeFire (40) | Ash x3, Maxx "C" x3, Imperm x3, Veiler x2, Called x2, Droll x1, Nibiru x1 |
| Labrynth (40) | Ash x3, Maxx "C" x3, Imperm x3, Called x2, Veiler x1, Belle x1 |
| Chimera (44) | Ash x3, Maxx "C" x3, Nibiru x3, Veiler x2, Droll x2, Called x2 |

All five run Ash x3 + Maxx "C" x3 + Called x2, so every question type applies
to every deck, and the `real` targets are always cards the deck does not run.

## Question types per deck

- **pos_control** (`<deck>-pos-ht-vanilla`): all 6 copies of Ash + Maxx "C" ->
  Mystical Elf (15025844, vanilla Normal Monster, ATK 800 / DEF 2000, in the
  card DB). Expected clearly worse in both seats.
- **neg_control** (`<deck>-neg-altash`): 1 Ash (14558127) -> alternate-art
  14558128 (alias = 14558127, same text). Verified 2026-10-16: 8 greedy games
  Branded vs Hero, seeds 0-3, identical outcomes AND identical choice sequences
  (the core resolves the alt art to the same script). Expected diff 0.
- **real** x2: 1 copy of a most-played hand trap -> 1 copy of a hand trap the
  deck does not run. Targets differ per deck so we learn about several cards:
  Branded Ash->Droll / Maxx "C"->Veiler; CenturIon Ash->Ogre / Maxx "C"->Crow;
  SnakeEyeFire Ash->Ogre / Maxx "C"->Crow; Labrynth Ash->Droll / Maxx "C"->Ogre;
  Chimera Ash->Imperm / Maxx "C"->Ogre.
- **real Called** (`<deck>-called-*`): 1 Called by the Grave -> 1 copy of an
  unrun hand trap (Crow, or Belle where Crow is taken). All five decks run
  Called x2, so this is the "out" direction everywhere.

## Reading a report (`<out>/<id>/summary.md`)

- `overall B-A`: paired win-rate difference with 95% CI and sign test.
- `by_seat`: going-first / going-second splits (seat confound).
- `exposure`: pairs where a swapped-in copy is in the opening hand (top 5 of
  the seeded shuffle) / drawn by turn 3 (top 8) / never seen. Effect
  conditional on exposure is the causal read; the unexposed split must be ~0.
- `noise floor`: among never-seen-by-turn-3 pairs, the identical-choice-sequence
  rate (want ~1 with deterministic pilots) and the outcome-difference rate.
- Pre-registered reads: A/A exactly 0 (else exit 2, job stops); neg_control not
  significant; pos_control significantly negative in both seats.

## Known behavior: option-list exposure (measured 2026-10-16)

Draw-based exposure misses one channel from plan §3c ("appear in an option
list"): deck searches. Verified on fixture BlueEyes (3x The Melody of
Awakening Dragon, a deck search): swapping 1 Blue-Eyes -> Mystical Elf leaves
3 pairs unexposed by draws, yet their choice sequences differ (option lists at
the Melody search change) while outcomes are identical (B-A +0.000). So:
- the identical-choice rate is exact for A/A and neg_control (fully identical
  runs: measured 1.0 on both);
- for real swaps it is a lower bound on "untouched": deck searches legitimately
  desync choices without changing outcomes. The outcome-difference rate among
  unexposed pairs is the noise floor that must be ~0.
