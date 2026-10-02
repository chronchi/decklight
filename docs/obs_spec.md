# Observation spec (v1)

Status: v1, 2026-10-01. Reference implementation: `src/ygo_zero/engine/obs.py`; the C++ env must produce
identical arrays (the contract test compares them). Bump `OBS_VERSION` on any change and log it in NOTEBOOK.

Adapted from ygo-agent's `docs/feature_engineering.md`, with these differences:
- **No card-ID table.** Cards carry their raw `code` (int32, 0 = hidden). The model looks up card
  knowledge (effect tokens, Phase 2) by code, so an unseen card works out of the box.
- **Effects are referenced by (code, string index)**, as edo encodes them: `desc = code << 20 | idx`.
- **No deck-order leak.** ygoenv listed your own deck in deck order, which tells the policy what it will
  draw. Here the own deck is a multiset (sorted by code, no sequence).
- Everything is int32 (simple, and the model embeds or normalises each column itself).
- Perspective: always the player to act (`me`). "Controller 0" means me.

All arrays are fixed size and zero padded; `*_mask` marks the valid rows.

## `cards` — int32 [MAX_CARDS=160, 21], `card_mask` bool [160]

Row order (stable, so truncation drops the least important rows first):
me: hand, mzone, szone, grave, removed, extra, overlay → opp: hand, mzone, szone, grave, removed,
extra (face-up only), overlay → me: deck. Within a location: by sequence (own deck: by code).
Overlay units follow their own zone order (holder zone, then material index).

| col | name | value |
|---|---|---|
| 0 | code | card code; 0 if hidden from me |
| 1 | location | 1 deck, 2 hand, 3 mzone, 4 szone, 5 grave, 6 removed, 7 extra, 8 overlay |
| 2 | sequence | zone/position index + 1 (overlay: holder zone + 1); 0 for own deck |
| 3 | controller | 0 me, 1 opp |
| 4 | owner | 0 me, 1 opp (0 if hidden) |
| 5 | position | mzone: 1 FU-ATK, 2 FD-ATK, 3 FU-DEF, 4 FD-DEF; szone/grave/removed/extra: 5 face-up, 6 face-down; hand/deck/overlay: 0 |
| 6 | public | 1 if the card is public (face-up, revealed, or in a chain) |
| 7 | type | current TYPE_* bitmask (0 if hidden) |
| 8 | attribute | current ATTRIBUTE_* bitmask |
| 9 | race | current RACE_* bitmask, low 32 bits (bit 62 YOKAI folded to bit 31… see note) |
| 10 | level | current level (0 if none) |
| 11 | rank | current rank |
| 12 | link | link rating |
| 13 | link_marker | LINK_MARKER_* bitmask |
| 14 | attack | current ATK (−2 = "?") |
| 15 | defense | current DEF |
| 16 | base_attack | |
| 17 | base_defense | |
| 18 | lscale \| rscale << 8 | pendulum scales |
| 19 | status | `status & STATUS_OBS_MASK`: DISABLED, SET_TURN, SUMMON_TURN, FLIP_SUMMON_TURN, SPSUMMON_TURN, FORM_CHANGED, ATTACK_CANCELED, CHAINING, FORBIDDEN |
| 20 | selected | 1 if already picked in the multi-step selection in progress (action spec §Multi-step) |

Hidden rows (code 0) keep location, sequence, controller, position and public; every other column is 0.
Overlay rows carry only code, location, sequence, controller, owner (= holder's controller) and public = 1;
the card stats come from the card-knowledge side by code. Counters are not in v1.

Race note: RACE_YOKAI is bit 62 in the core. v1 maps it to bit 31 (RACE_GALAXY is unused by any TCG card).

Visibility rules (what is "hidden from me"):
- Own cards: everything is visible, except deck order.
- Opponent: hand, face-down field cards and face-down banished cards are hidden unless the core
  reports them as public. Deck and face-down extra deck get no rows, only counts in `global`.
- Overlay materials are public.
- Cards revealed earlier and since hidden again (a searched card shown, then held) are **not** remembered
  in v1. The action history and the model's recurrence carry that information.

## `global` — int32 [24]

| idx | name |
|---|---|
| 0 | my LP |
| 1 | opp LP |
| 2 | turn number (1 = first turn of the duel) |
| 3 | phase (PHASE_* value) |
| 4 | I went first (1/0) |
| 5 | it is my turn (1/0) |
| 6–12 | my counts: deck, hand, mzone, szone, grave, removed, extra |
| 13–19 | opp counts, same order |
| 20 | chain length |
| 21 | decision message type (MSG_* of the pending core message) |
| 22 | rows truncated (count of card rows dropped because of MAX_CARDS) |
| 23 | spec version (1) |

## `chain` — int32 [MAX_CHAIN=16, 5], `chain_mask` bool [16]

Current chain from link 1 up: card row (−1 if no row), code of the handler, effect `desc >> 20`,
`desc & 0xfffff`, controller (0 me, 1 opp).

## `actions` — see `docs/action_spec.md`
`actions` int32 [MAX_OPTIONS=64, 10] + `action_mask` bool [64].

## `history` — int32 [HISTORY=32, 13], `history_mask` bool [32]

The last 32 decisions taken in this duel by either player, most recent first. Columns 0–9 are the action's
columns from `action_spec.md` with `card` set to −1 (rows move; `code`, `location` and `controller` identify
the card as it was), followed by
10 actor (0 me, 1 opp), 11 turns ago (0 = this turn), 12 phase when taken.
Opponent entries keep `code` and the effect columns only if the card was public to me: the kind reveals it
(ACTIVATE, SUMMON, SPSUMMON, REPOS, ATTACK), I control the card, or its row was public at decision time;
no-card options are public except a face-down POSITION. Otherwise columns 2–4 are 0.
Auto-taken single-option steps (see action spec) are not recorded.
