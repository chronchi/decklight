# Search plan (ADR 0006 M2/M3 → M4/M5), after the PPO plateau

_2026-10-06. User: "can we do 1 and 3 in parallel?" (1 = search, 3 = a long PPO run as a scale test, ≈ $X)._
Why now: PPO from base_r3 plateaus at ≈ 0.51 vs base_r3 with three recipes (NOTEBOOK 19:30). Round-2 search labels
(value v2, h1, argmax playouts) also saturated (NOTEBOOK 11:10): the value net predates the current policy and the
search was shallow.

## Principle (AlphaZero / expert iteration)
A search over the policy is only a useful teacher if it plays clearly better than the policy alone. Measure that
first and cheaply; distil only if the gap exists.

## Steps (each gated by a number)
- **S0 Gate: test-time search match** (`scripts/search_match.py`): base_r3 + search at its uncertain decisions vs raw
  base_r3, both seats, the report card's 8 pool decks.
  - Search: windows plus own-turn decisions with ≥ 2 options where the policy's top probability is < 0.9. It tries
    the k = 4 policy candidates with 4 sampled playouts each to the end of the turn, scored by a value.
  - Variants: value v2 vs the board score; horizon 1 vs 2.
  - Gate: S wins **≥ 58%** (≈ significant at n ≈ 450).
  - Also measures seconds per searched decision, which sizes S2.
- **S1 Value v3**: a value for the current policy. The cheapest source is the PPO value head. The next PPO run must
  keep it (today `vast_rl.sh` drops `last.pt` and `policy_state` strips `value_head.`).
  - Then compare v2 vs v3 inside the S0 harness: the better search value wins.
- **S2 Distil (ExIt round)**: search labels on base_r3's own games at the decisions S0 shows matter (window
  search with the S0 settings), fine-tune base_r3, report card vs base_r3.
- **S3 RL from the distilled policy** (M4), then **S4 test-time search** at play (M5) if S0's gap holds at play
  speed.

## Costs
S0 is CPU-heavy: clone-by-replay plus playouts; a many-core GPU box, ≈ $X. S2 is similar plus a fine-tune.
