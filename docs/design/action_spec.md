# Action spec (v1)

Status: v1, 2026-10-01. Reference implementation: `src/decklight/engine/actions.py`. The C++ env must produce
the same option lists in the same order, and the same core responses.

Every decision is a **choice of one option** from a list of legal options (≤ MAX_OPTIONS = 64). Core
messages that take a compound answer (pick k cards, pick a sum, order cards, place counters) are split into
a sequence of single choices; the env builds the core response when the sequence is complete. The
policy therefore always outputs one index, as in ygo-agent.

## Option features — int32 [64, 10]

| col | name | meaning |
|---|---|---|
| 0 | kind | see table below (0 = padding) |
| 1 | card | row in `cards` (−1 = none / no row) |
| 2 | code | card code involved (for DECLARE: the declared code); 0 = none, hidden row, or a row-less card I do not control |
| 3 | effect_code | `desc >> 20` of the effect/string involved (0 = system string) |
| 4 | effect_idx | `desc & 0xfffff` (string index 0–15 for card strings; system string id otherwise) |
| 5 | arg | kind-specific: zone, position, number, bit, option index (below) |
| 6 | msg | MSG_* type of the core message being answered |
| 7 | location | location id (obs_spec col 1) of the card, 0 = none |
| 8 | controller | 0 me, 1 opp, −1 none |
| 9 | step | how many picks are already made in this multi-step selection (0 for single-step) |

## Kinds

| id | kind | from message | arg |
|---|---|---|---|
| 1 | SUMMON | IDLECMD summonable | |
| 2 | SPSUMMON | IDLECMD spsummonable | |
| 3 | REPOS | IDLECMD repositionable | |
| 4 | MSET | IDLECMD msetable | |
| 5 | SSET | IDLECMD ssetable | |
| 6 | ACTIVATE | IDLECMD / BATTLECMD activatable, CHAIN chains, EFFECTYN "yes" is **not** this | client_mode |
| 7 | ATTACK | BATTLECMD attackable | 1 if it can attack directly |
| 8 | TO_BATTLE | IDLECMD | |
| 9 | TO_MAIN2 | BATTLECMD | |
| 10 | TO_END | IDLECMD / BATTLECMD | |
| 11 | PASS | CHAIN (don't chain; absent when forced) | |
| 12 | YES | YESNO / EFFECTYN (card + desc set for EFFECTYN) | |
| 13 | NO | YESNO / EFFECTYN | |
| 14 | OPTION | SELECT_OPTION (desc in effect cols) | option index |
| 15 | SELECT | pick a card: SELECT_CARD, TRIBUTE, SUM, UNSELECT_CARD (selectable part) | release/sum param |
| 16 | UNSELECT | UNSELECT_CARD (unselectable part) | |
| 17 | FINISH | stop picking (min reached / finishable) | |
| 18 | CANCEL | cancel the whole selection (cancelable) | |
| 19 | PLACE | SELECT_PLACE / DISFIELD | zone: 0–6 my MZ, 8–15 my SZ, 16–22 opp MZ, 24–31 opp SZ |
| 20 | POSITION | SELECT_POSITION (code col = the card) | POS_* bit |
| 21 | NUMBER | ANNOUNCE_NUMBER | the number |
| 22 | ATTRIBUTE | ANNOUNCE_ATTRIB | ATTRIBUTE_* bit |
| 23 | RACE | ANNOUNCE_RACE | bit index |
| 24 | DECLARE | ANNOUNCE_CARD | |
| 25 | SORT | SORT_CARD / SORT_CHAIN: the card that goes next | |
| 26 | COUNTER | SELECT_COUNTER: remove one counter from this card | counters left on it |
| 27 | RPS | ROCK_PAPER_SCISSORS | 1 rock, 2 paper, 3 scissors |

Option order within a message = the order of the lists in the core message (IDLECMD: summon, spsummon,
repos, mset, sset, activate, then TO_BATTLE, TO_END; BATTLECMD: activate, attack, TO_MAIN2, TO_END). IDLECMD
"shuffle hand" is never offered (it does nothing for play).

## Multi-step decompositions

- **SELECT_CARD** (min..max, max clamped to the list size): SELECT any unpicked card; FINISH once
  min ≤ picks < max; CANCEL if cancelable and nothing is picked yet. The response is sent automatically when
  picks = max. Picks are a set, so the response lists indices in ascending order.
- **SELECT_TRIBUTE**: the core accepts ≤ max cards whose release params (a card may count as 2) sum to
  ≥ min. FINISH once the sum ≥ min; a card is offered only if a valid completion still exists after picking
  it; the response is sent automatically when picks = max.
- **SELECT_SUM**: valid subsets are enumerated once (must-cards always included); a card is offered only if
  some valid subset contains the current picks plus it. FINISH appears when the current picks are a valid
  subset. Exact mode: min..max counts only the non-must cards (as the core checks it). At-least mode:
  valid = the core's rule (no redundant card). Enumeration is a pruned DFS capped at 200k nodes.
- **SELECT_UNSELECT_CARD** is already one-pick-per-message in the core: SELECT / UNSELECT one card, then
  FINISH if finishable, else CANCEL if cancelable (never both: both send −1).
- **SELECT_PLACE / DISFIELD** with count > 1: PLACE one zone at a time (picked zones removed).
- **SELECT_COUNTER**: `count` steps of COUNTER (one counter at a time, from cards that still have one).
- **SORT_CARD / SORT_CHAIN**: SORT picks the next card in the order until one is left (it goes last). The
  response gives each card its rank in that order.
- **ANNOUNCE_RACE / ATTRIB** with count > 1: one bit per step.
- **ANNOUNCE_CARD**: options are the codes satisfying the core's filter (`is_declarable`) among my deck
  list (main + extra) ∪ codes of the rows visible to me now ∪ codes in my history (public entries for the
  opponent's), sorted by code. If none qualify, the
  first declarable code in the whole card DB (by code) is the single option.

The `selected` column of `cards` marks cards already picked in the current multi-step selection.

## Automatic steps

If a decision has exactly **one** legal option, the env takes it without asking the agent and does not
record it in the history (it is still part of the core response log). This covers the 57% of
SELECT_CHAIN messages with no chainable effect (PASS only). Configurable (`auto_single`), on by default.

## Truncation

If more than 64 options exist, the first 64 in the order above are offered and `info["truncated"]` counts
it. Measured rate: see NOTEBOOK (target: never in the deck pool).

## Replays

An env game is reproduced by (DuelConfig, env spec version, `auto_single`, list of chosen option indices;
automatic steps are not in the list). Reference: `PyEnv.replay` in `src/decklight/engine/env.py`. Option lists
are deterministic given the core state, so replaying the indices rebuilds every observation.
