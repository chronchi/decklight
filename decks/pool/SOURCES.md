# Pool deck sources (tracked, current TCG lists)

Date: 2026-10-02. Each `decks/pool/<deck>.ydk` is the most recent TCG list
(2025–2026) from a source whose combos are in `data/guides/<deck>/`,
preferring the list the most extracted guide lines were written for.
`tests/fixtures/decks/*.ydk` are untouched (tests depend on them).

Validation: every code exists in our card DB (`load_cards()`, 14982 cards);
main 40–60, extra ≤ 15, side ≤ 15, ≤ 3 copies per card with alt arts grouped
by alias. Ban check against the `!2026.2 TCG` section of
`../ygorl/refs/ygopro/lflist.conf` (no TCG `lflist.conf` in this repo; only
sibling refs + a rush template). 0 = Forbidden, 1 = Limited, 2 = Semi-Limited.

## blue-eyes → `decks/pool/blue-eyes.ydk` (40 + 15 + 0 side)

- Source URL: https://gatheringgames.co.uk/community/blog/yu-gi-oh-blue-eyes-white-destiny-structure-deck-beginners-guide
- Site: Gathering Games (UK TCG shop blog). List date: 2025-04-19. Format: TCG (3x Structure Deck: Blue-Eyes White Destiny).
- Why: 8/14 extracted Blue-Eyes lines come from this source (the most of any; Game8 MD has 2, note.com OCG has 4, ygoprodeck MD primer has 0). Also the most recent TCG list in the guides (ygoprodeck-faithsc 2025-03-07 is MD).
- Validation: all 55 codes in DB; no copy > 3; no `!2026.2 TCG` hits. Side: guide has no side deck → `!side` empty.

## salamangreat → `decks/pool/salamangreat.ydk` (43 + 15 + 15 side)

- Source URL: https://duelingnexus.com/blog/salamangreat-combo-2025-deck/
- Site: Dueling Nexus blog (user EinfachnurOk / Shiloh). List date: 2025-04-14 (TCG April 2025 banlist). Format: TCG.
- Why: the only TCG list among the guide sources (Game8 2025-11 and ygoprodeck-beginners 2024-06 are MD). 5/8 extracted lines are RotK 2019 foundational combos; the TCG list grounds them via the augmented fallback where cards were cut.
- Validation: all 73 codes in DB; no single card > 3 in any one section. `!2026.2 TCG` hits: Called by the Grave 1/1, Triple Tactics Talent 1/1, Crossout Designator 1/1 — all within limits. FLAGGED: Droll & Lock Bird 1 (main) + 2 (side) = 3 combined > semi-limit 2 under the current list (the April-2025 list predates that hit). Kept faithful, not silently changed.

## sky-striker → `decks/pool/sky-striker.ydk` (40 + 15 + 15 side)

- Source URL: https://ygoprodeck.com/deck/sky-striker-6-5-2025-609865
- Site: YGOPRODeck (Kn1ghtmare3, "modified Ryan Yu list"). List date: 2025-06-05. Format: TCG (40-card main).
- Why: the only TCG list among the guide sources (Game8 2025-12 and MDM secant0 2025-11 are MD; Konami LD26 2026 guide has lines but no decklist on the page). Konami:1 (Area Zero line) is missing_card against it — known, augmented fallback covers.
- Validation: all 70 codes in DB; `!2026.2 TCG` hits all within limits (Feather Duster 1/1, Called by 1/1, Talent 1/1, RoTA 1/1, Hornet Drones 1/1). FLAGGED: Lightning Storm 2 (main) + 2 (side) = 4 combined > 3-copy rule. Kept faithful, not silently changed.

## branded → `decks/pool/branded.ydk` (40 + 15 + 15 side)

- Source URL: https://ygoprodeck.com/deck/branded-dracotail-687091
- Site: YGOPRODeck (Fidgety42). List date: 2026-02-01 (uploaded ~8 months before fetch). Format: TCG (Meta Decks category, TCGplayer-priced).
- Why: Branded had no extracted decklist JSON; this page (already in `data/guides/branded/04_ygoprodeck-dracotail.txt` as primer-only) is the current-TCG Branded Dracotail shell. Chosen over pure/Despia because it matches the guide combos best on both ends: the RotK line's Fallen of the White Dragon + Ecclesia and the Dark Dragon package (2x Fallen WD + 1x Albaz + Ecclesia Dark all in the list) and the DeckScope/MDM-Vector classic engine (Branded Fusion, Cartesia, Albion, Mirrorjade, Titaniklad). 11/13 extracted lines are classic MDM-Vector lines; their key cards (Aluber, Lubellion Searing, Granguignol, Quem) are NOT in this Dracotail list → expect missing_card → augmented fallback at verify time. Side deck (15) from the same page grid.
- Fetches used: 1 (this page re-fetch; card-id grid recovered, all ids resolve vs our DB). Raw page also saved into the guide file's new decklist section.
- Validation: all 70 codes in DB; no copy > 3 with Fallen of the White Dragon (alias 68468459) grouped with Fallen of Albaz: 2 + 1 = 3, exactly at the rule. `!2026.2 TCG` hits all within limits: Branded Fusion 2/2 (semi), Mululu 1/1, Ketu 1/1, Magnamhut 1/1, Druiswurm 1/1. No violations.
- Note: page ban icons (limited_1 on Branded Fusion etc.) are Master Duel rarities display, not TCG — TCG check above governs.
