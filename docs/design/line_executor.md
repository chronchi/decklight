# Line executor / verifier (ADR 0002 stage 3) — design

Goal: turn an extracted `ComboLine` (`src/decklight/demos/schema.py`) into a **verified env trajectory**, or reject it
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

## Quality guards (rejected even when the board matches)
Executed lines are still rejected — kept out of the verified set but counted by reason — when:
- `vacuous`: all steps were skipped, or the line used 0 agent (non-`auto`) choices. A verify that did
  nothing proves nothing.
- `too_skipped`: more than 1/3 of the steps were skipped (`len(skipped) * 3 > n`). Skips are only for
  conditional steps (no results/materials/targets, anchor never legal); a line that mostly skips is a fragment.
- `trivial`: fewer than 2 steps actually executed (`n - len(skipped) < 2`). One-step "combos" are fragments.
Priority: `vacuous` > `too_skipped` > `trivial`. Every record carries `executed_steps` (= n − skipped) and
`inferred_frac` (= inferred / agent choices) so later stages can filter/sort by quality.

## De-bundling multi-result steps
Extraction sometimes attributes to one step results that only arrive a step later (bundled results), which the
old strict check pruned. When a step ends and its results check fails **but at least one expected result is
already out**, the missing codes are carried forward once: they must all be out of the deck by the end of the
*next* step, together with that step's own results. If they are still missing there — or if nothing had arrived
at all — the branch fails as before. Carries are per search path (one step of grace, chainable step by step)
and are recorded in the verdict `detail` as `carried=[(from_step, [codes]), ...]`. The last step cannot carry
(past the line end there is no next step).

## Hand recovery (MDM lazy-load loss)
Some fetched guides (MasterDuelMeta) arrive with an empty `hand` list although the combo starts from hand cards.
When a line has no known hand cards, the hand is inferred as the anchor cards of the leading steps that are
in-hand actions (`normal_summon` / `set_monster` / `set_spelltrap` / `activate`), stopping **after** the first
search/special effect (a step whose results name any card other than its own anchor, or any
special/link/synchro/xyz/fusion/ritual summon — later anchors may have been searched, so they are not hand
cards). Only cards already named in the steps are used; nothing is invented (one copy each, capped by what the
deck holds via the usual `missing_card` check). Inferred hands are marked `hand_inferred=True` in the line
`notes` and in the verified record. Lines that open with a fusion (no in-hand anchor before the special) infer
nothing and still fail with `no_anchor@0` — they are genuine fragments.

## Output record (`data/demos/verified/<deck>.jsonl`)
`{line_id, source_url, deck, config (decks, seed, setup), auto_single, choices: [int], labels: ["stated"|"inferred"|"auto"],
end_board, seeds_ok: [...], skipped_steps, executed_steps, inferred_frac, hand_inferred, deck_source}`. Expansion (stage 4): re-run the verified choice *policy* (anchors + stated picks) under
other seeds, same hand. Keep the seeds where it still verifies.

## Metrics (per deck and on the gold set)
lines in → verified %, reasons for rejection (missing_card, unresolved, no_anchor@step k, budget, end_board_mismatch,
vacuous, too_skipped, trivial), % inferred choices per verified line, search nodes per line.

## Rules added 2026-10-02 (Claude, from traced failures)
- **Step transitions at any decision.** Inside step k, an option that anchors step k+1 (e.g. a summon trigger
  offered as yes/no mid-chain) may close step k; k's missing results are carried to k+1's end.
- **Implicit steps.** A step whose results already came out on their own (forced trigger of a card involved in
  the previous step) counts as executed; listed as `implicit=[...]` in the detail.
- **Where results must be.** Summon/set steps: on my field. Other steps: hand, field, GY, banished or overlay.
  Never the Extra Deck (an Extra Deck card has not arrived anywhere). Carried results count if on the field or
  newly arrived elsewhere since the carry (used as material); a card that stayed in the GY does not.
- **Set via activation.** "X sets itself from the GY" is an activation: set steps also anchor on ACTIVATE/YES of
  the same card. Special-summon steps anchor on activations only of cards the step names.
- **Bounded unstated actions.** Between steps, at most `MAX_EXTRA_ACTIONS` (1) main-phase action the guide did not
  state, and only with a card the line mentions later. Everything else is anchors, skips, implicit steps and phase moves.
- **Generic names** (`demos/generic.py`). "a Branded Spell/Trap", "Bystial", "a FIRE monster" resolve to the
  deck's cards matching archetype (CardScripts setcodes) + type/attribute words. Hand placeholders become the
  first match; step cards accept any match as anchor; materials/targets add all matches to the stated set;
  generic results and requires are dropped from the checks. Descriptions with no archetype/attribute
  constraint ("lock monster") stay unresolved, so the line is rejected.
- **Diagnosis.** Failed verdicts carry the path to the deepest step reached; `scripts/explain_line.py` replays it
  and prints the zones and options at the dead end.
