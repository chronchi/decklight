# Line executor / verifier (ADR 0002 stage 3) — design

Goal: turn an extracted `ComboLine` (`src/ygo_zero/demos/schema.py`) into a **verified env trajectory**, or reject it
with a reason. This is the only gate between internet text and training data, so it errs on the side of rejecting.

## Setup (turn-1 solitaire)
- Puzzle start (`DuelConfig.setup`, `hand=0`): my hand = `hand_codes` placed in HAND; my deck = the deck list minus
  those cards (shuffled by the seed); extra = deck list extra. The line's `requires_codes` must be in deck/extra,
  else reject (`missing_card`).
- Opponent: any legal deck, **empty hand** (no hand traps). Its decisions are trivial and auto-taken. Lines that
  need opponent interaction (`going == second`, attacks) are out of scope for v1 (reject `out_of_scope`).
- I go first; stop at my End Phase (or when the line ends + `TO_END` is legal).

## Matching steps to options
Each step names an **anchor**: the option that starts it.

| step.action | anchor option |
|---|---|
| normal_summon / set_monster | SUMMON / MSET with code = card |
| activate | ACTIVATE with code = card (any effect; `effect` text only breaks ties via the Lua record's description) |
| special_summon | SPSUMMON with code = card, else ACTIVATE of a card whose results include it |
| link/synchro/xyz/fusion/ritual_summon | SPSUMMON of the extra-deck card (procedure); fusion/ritual via the spell: ACTIVATE |
| set_spelltrap | SSET with code = card |

After the anchor, every decision until the next IDLECMD/BATTLECMD (or until the chain resolves) belongs to the step:
costs, targets, materials, searches, zones, positions, yes/no, trigger order.
- An option is **stated** if its card code is in the step's materials/targets/results codes (or the YES/NO the text implies).
- Otherwise it is **inferred**: chosen by search.

## Search
Depth-first over decisions with backtracking through replay-clone (C++ `Env` + replayed choices; creating a duel costs
~15 ms). Order candidates: stated options first, then the rest in a fixed order. Prune a branch when:
- a step's results are not where the text says once the step ends (e.g. "search Maiden" → Maiden in hand),
- the next step's anchor is no longer legal (e.g. a once-per-turn already used),
- per-line node budget exhausted (default 2,000 decisions) → reject `budget`.
Bonus checks that also prune: count limits from `cards/lua_effects.py` (HOPT) to skip activations that would block a later step.

## Acceptance of a line
Executed every step in order **and** at the End of the line `end_board_codes` ⊆ my field (monsters + spells/traps,
by code; Extra Monster Zone included). Unresolved names: steps whose card is unresolved make the line
`unresolved` (rejected) in v1.

## Output record (`data/demos/verified/<deck>.jsonl`)
`{line_id, source_url, deck, config (decks, seed, setup), auto_single, choices: [int], labels: ["stated"|"inferred"|"auto"],
end_board, seeds_ok: [...]}`. Expansion (stage 4): re-run the verified choice *policy* (anchors + stated picks) under
other seeds, same hand. Keep the seeds where it still verifies.

## Metrics (per deck and on the gold set)
lines in → verified %, reasons for rejection (missing_card, unresolved, no_anchor@step k, budget, end_board_mismatch),
% inferred choices per verified line, search nodes per line.
