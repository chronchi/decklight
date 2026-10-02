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

## mitsurugi → `decks/pool/mitsurugi.ydk` (44 + 15 + 15 side)

- Source URL: https://duelingnexus.com/blog/yummy-mitsurugi-deck-2025/
- Site: Dueling Nexus blog (user Daudodz_VN / Shiloh). List date: 2025-10-27, updated 2026-01-18 (October 2025 TCG banlist). Format: TCG.
- Why: the only TCG list among the guide sources (Game8 pure/Ryzeal are MD; RotK/Flipside carry lines but no decklist). Yummy-Mitsurugi shell holds the full Mitsurugi core the Game8/RotK lines need (Habakiri 3, Prayers 3, Ritual, Mirror, Saji/Aramasa/Kusanagi, Murakumo, Pre-Prep 3, Great Purification); Night Sword Serpent, Futsu no Mitama, Magatama and Wousu are absent → augmented fallback covers those lines.
- Validation: all 74 codes in DB; no copy > 3. `!2026.2 TCG` hits: Talent 1/1, Called by 1/1, Crossout 1/1, Snatchy 1/1 (all within limits). FLAGGED, kept faithful: Ame no Habakiri no Mitsurugi 3 (main) > limited 1; Droll & Lock Bird 3 (main) > semi-limit 2; Herald of the Arc Light 1 (extra) is Forbidden (0) under `!2026.2 TCG`. The Oct-2025 list predates all three hits.

## kewl-tune → `decks/pool/kewl-tune.ydk` (40 + 15 + 15 side)

- Source URL: https://duelingnexus.com/blog/kewl-tune-deck-2026-2/
- Site: Dueling Nexus blog (user QuickSwipe / Lorelei). List date: 2026-09-24 (September 2026 TCG banlist). Format: TCG.
- Why: the most current TCG list among the guide sources, and it matches both combo sources on the end-board core (Track Maker 2, Remix 2, RS 2, Crackle, B2B, JJ, Synchro 3, Zalen, Clip, Cue/Reco/Mix 3). Chosen over the MDM-counter 40 (no banlist tag) and the YGOPRODeck spreadsheet 42 (Branded package: Albaz/Lubellion/Mirrorjade/Titaniklad). Kewl Tune Rotary (needed by both combo sources' Cue lines) is NOT in this list → augmented fallback (the MDM-counter guide decklist does carry Rotary 3).
- Validation: all 70 codes in DB; no copy > 3. `!2026.2 TCG` hits all within limits (Droll & Lock Bird 2/2). No violations.

## elfnote → `decks/pool/elfnote.ydk` (40 + 15 + 15 side)

- Source URL: https://kotacards.com/deck.php?d=Elfnote
- Site: KotaCards (tournament profiles). List: 1st Place, Edoardo Fanton, Rome WCQ Regional 2026-09-27. Format: TCG.
- Why: tournament-winning, most current TCG list; holds the YGOPRODeck 9-step combo's core (Medius 2, Power Patron 3, Tinia, Welcome Home 2, Crystal Wing) via Accel Synchron? — NO: Accel Synchron, Jet Synchron, Ultimaya Tzolkin and Link Spider are absent → augmented fallback for that line. Chosen over the DN Aug-2026 list (runs 3x Ars Magna + Mediclius, none in our DB).
- Validation: all 70 codes in DB; no copy > 3. `!2026.2 TCG` hits all within limits (Magnamhut 1/1, T.G. Hyper Librarian 1/1, Droll 2/2). No violations.

## mikanko → `decks/pool/mikanko.ydk` (40 + 14 + 0 side)

- Source URL: https://ygoprodeck.com/deck/tcg-mikanko-april-2026-704304
- Site: YGOPRODeck (huh, clever). List date: 2026-04-06, edited 2026-04-13. Format: TCG, built to play first (40-card main).
- Why: the combo source's own list and the only going-first TCG Mikanko list: grounds the Rivalry/Spiritwalk/Ni-Ni interruption lines (Hu-Li 2, Ohime 3, Rivalry, Spiritwalk, Promise, Arabesque 3, Fire Dance, Mayowashidori, Ceremony, Uzuhime 2, Ni-Ni).
- Fetches used: 1 (page re-parse; full list recovered from the revision history, sections re-split by card type, totals 40+14). The page's extra grid is JS, so a 15th extra card (if the author runs one) is unknown; the list as published sums to 40+14, both legal sizes. Side grid is also JS → `!side` empty (blue-eyes precedent).
- Validation: all 54 codes in DB; no copy > 3. `!2026.2 TCG` hits all within limits (There Can Be Only One 1/1, Called by 1/1). No violations.
