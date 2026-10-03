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

## ygoprodeck meta fetch — 2026-10-03 (82 decks)

- Endpoint: https://ygoprodeck.com/api/decks/getDecks.php?format=Tournament%20Meta%20Decks (pages 10 x 50). Raw cache: `data/cache/meta_decks/`.
- Selection: newest <=2 valid lists per archetype slug, preferring better placements. Validation: official script + engine card DB, alias-grouped <=3, main 40-60, extra <=15, side <=15, engine smoke (2 self-play games).
- Decks:
  - `decks/pool/angelechy-meta1.ydk` (40 + 15 + 15 side): Angelechy — Aalst WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/angelechy-735186 [train]
  - `decks/pool/angelechy-meta2.ydk` (40 + 15 + 15 side): Angelechy — Aalst WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/angelechy-735188 [train]
  - `decks/pool/angelechy-elfnote-meta1.ydk` (40 + 15 + 15 side): Angelechy Elfnote — Rende WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/angelechy-elfnote-735371 [train]
  - `decks/pool/angelechy-elfnote-meta2.ydk` (41 + 15 + 15 side): Angelechy Elfnote — The Side Deck Remote Duel WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/angelechy-elfnote-735109 [train]
  - `decks/pool/atlantean-mermail-meta1.ydk` (44 + 15 + 15 side): Atlantean Mermail — Montevideo WCQ Regional, Runner-Up, 2026-09-12 — https://ygoprodeck.com/deck/atlantean-mermail-734360 [train]
  - `decks/pool/atlantean-mermail-meta2.ydk` (60 + 15 + 15 side): Atlantean Mermail — Kissimmee WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/atlantean-mermail-735165 [train]
  - `decks/pool/azamina-light-and-darkness-ritual-meta1.ydk` (42 + 15 + 15 side): Azamina Light and Darkness Ritual — Santiago WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/azamina-light-and-darkness-ritual-735065 [train]
  - `decks/pool/azamina-light-and-darkness-ritual-meta2.ydk` (40 + 15 + 15 side): Azamina Light and Darkness Ritual — Le Mans WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/azamina-light-and-darkness-ritual-735779 [train]
  - `decks/pool/azamina-mitsurugi-orcust-meta1.ydk` (47 + 15 + 15 side): Azamina Mitsurugi Orcust — Rende WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/azamina-mitsurugi-orcust-735951 [train]
  - `decks/pool/azamina-shining-sarcophagus-mitsurugi-orcust-meta1.ydk` (60 + 15 + 15 side): Azamina Shining Sarcophagus Mitsurugi Orcust — Poznań WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/azamina-shining-sarcophagus-mitsurugi-orcust-734799 [train]
  - `decks/pool/azamina-white-forest-meta1.ydk` (40 + 15 + 15 side): Azamina White Forest — Ballarat WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/azamina-white-forest-734647 [train]
  - `decks/pool/blitzclique-meta1.ydk` (40 + 15 + 15 side): Blitzclique — Le Mans WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/blitzclique-735151 [train]
  - `decks/pool/blitzclique-meta2.ydk` (40 + 15 + 15 side): Blitzclique — Castleford WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/blitzclique-735085 [train]
  - `decks/pool/branded-angelechy-light-and-darkness-ritual-meta1.ydk` (45 + 15 + 15 side): Branded Angelechy Light and Darkness Ritual — Bilbao WCQ Regional, Top 8, 2026-09-19 — https://ygoprodeck.com/deck/branded-angelechy-light-and-darkness-ritual-734306 [train]
  - `decks/pool/branded-blitzclique-meta1.ydk` (41 + 15 + 15 side): Branded Blitzclique — Groningen WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/branded-blitzclique-735096 [train]
  - `decks/pool/branded-dracotail-meta1.ydk` (40 + 15 + 15 side): Branded Dracotail — Castleford WCQ Regional, Top 4, 2026-09-27 — https://ygoprodeck.com/deck/branded-dracotail-735101 [train]
  - `decks/pool/branded-dracotail-meta2.ydk` (43 + 15 + 15 side): Branded Dracotail — Riyadh WCQ Regional, Top 8, 2026-09-25 — https://ygoprodeck.com/deck/branded-dracotail-734812 [train]
  - `decks/pool/branded-light-and-darkness-ritual-meta1.ydk` (44 + 15 + 15 side): Branded Light and Darkness Ritual — Foggia WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/branded-light-and-darkness-ritual-735946 [train]
  - `decks/pool/branded-light-and-darkness-ritual-meta2.ydk` (40 + 15 + 15 side): Branded Light and Darkness Ritual — Ballarat WCQ Regional, Runner-Up, 2026-09-26 — https://ygoprodeck.com/deck/branded-light-and-darkness-ritual-734767 [train]
  - `decks/pool/crystron-trains-meta1.ydk` (42 + 15 + 15 side): Crystron Trains — Montevideo WCQ Regional, Top 8, 2026-09-12 — https://ygoprodeck.com/deck/crystron-trains-734334 [train]
  - `decks/pool/dark-magician-azamina-snake-eye-meta1.ydk` (40 + 15 + 15 side): Dark Magician Azamina Snake-Eye — Poznań WCQ Regional, Runner-Up, 2026-09-26 — https://ygoprodeck.com/deck/dark-magician-azamina-snake-eye-734878 [train]
  - `decks/pool/dark-magician-mitsurugi-yummy-meta1.ydk` (60 + 15 + 15 side): Dark Magician Mitsurugi Yummy — Caracas WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/dark-magician-mitsurugi-yummy-735499 [train]
  - `decks/pool/despia-branded-meta1.ydk` (42 + 15 + 15 side): Despia Branded — Newry WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/despia-branded-734803 [train]
  - `decks/pool/despia-branded-meta2.ydk` (40 + 15 + 15 side): Despia Branded — Le Mans WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/despia-branded-735953 [train]
  - `decks/pool/dogmatika-branded-light-and-darkness-ritual-meta1.ydk` (60 + 15 + 15 side): Dogmatika Branded Light and Darkness Ritual — Buenos Aires WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/dogmatika-branded-light-and-darkness-ritual-736023 [holdout]
  - `decks/pool/doomz-meta1.ydk` (41 + 15 + 15 side): DoomZ — Thessaloniki WCQ Regional, Top 4, 2026-09-27 — https://ygoprodeck.com/deck/doomz-735284 [holdout]
  - `decks/pool/doomz-meta2.ydk` (40 + 15 + 15 side): DoomZ — Poznań WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/doomz-734788 [holdout]
  - `decks/pool/dracotail-despia-branded-meta1.ydk` (42 + 15 + 15 side): Dracotail Despia Branded — Rome WCQ Regional, Top 4, 2026-09-27 — https://ygoprodeck.com/deck/dracotail-despia-branded-735074 [train]
  - `decks/pool/elfnote-meta1.ydk` (40 + 15 + 15 side): Elfnote — Rome WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/elfnote-735631 [train]
  - `decks/pool/elfnote-meta2.ydk` (40 + 15 + 15 side): Elfnote — Bloomington WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/elfnote-735142 [train]
  - `decks/pool/elfnote-mitsurugi-synchron-meta1.ydk` (60 + 15 + 15 side): Elfnote Mitsurugi Synchron — Groningen WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/elfnote-mitsurugi-synchron-735183 [train]
  - `decks/pool/exosister-meta1.ydk` (40 + 15 + 15 side): Exosister — Arapiraca WCQ Regional, Top 4, 2026-09-27 — https://ygoprodeck.com/deck/exosister-735261 [train]
  - `decks/pool/fiendsmith-elfnote-meta1.ydk` (40 + 15 + 15 side): Fiendsmith Elfnote — Thessaloniki WCQ Regional, Runner-Up, 2026-09-27 — https://ygoprodeck.com/deck/fiendsmith-elfnote-735214 [train]
  - `decks/pool/fiendsmith-elfnote-meta2.ydk` (41 + 15 + 15 side): Fiendsmith Elfnote — Avrillé WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/fiendsmith-elfnote-734828 [train]
  - `decks/pool/fiendsmith-light-and-darkness-ritual-meta1.ydk` (41 + 15 + 15 side): Fiendsmith Light and Darkness Ritual — Rende WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/fiendsmith-light-and-darkness-ritual-735399 [holdout]
  - `decks/pool/fiendsmith-sacred-beasts-meta1.ydk` (40 + 15 + 15 side): Fiendsmith Sacred Beasts — Caracas WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/fiendsmith-sacred-beasts-735500 [holdout]
  - `decks/pool/ignister-maliss-meta1.ydk` (40 + 15 + 15 side): @Ignister Maliss — Cannington WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/ignister-maliss-736110 [holdout]
  - `decks/pool/ignister-maliss-meta2.ydk` (40 + 15 + 15 side): @Ignister Maliss — Arapiraca WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/ignister-maliss-735119 [holdout]
  - `decks/pool/invoked-meta1.ydk` (40 + 15 + 15 side): Invoked — Thessaloniki WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/invoked-735213 [train]
  - `decks/pool/invoked-meta2.ydk` (40 + 15 + 15 side): Invoked — Catskill WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/invoked-735651 [train]
  - `decks/pool/invoked-dracotail-meta1.ydk` (40 + 15 + 15 side): Invoked Dracotail — Santiago WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/invoked-dracotail-735062 [train]
  - `decks/pool/kewl-tune-meta1.ydk` (40 + 15 + 15 side): Kewl Tune — YCS Paris, Top 8, 2026-09-19 — https://ygoprodeck.com/deck/kewl-tune-734627 [train]
  - `decks/pool/light-and-darkness-ritual-sacred-beasts-meta1.ydk` (47 + 15 + 15 side): Light and Darkness Ritual Sacred Beasts — Duluth WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/light-and-darkness-ritual-sacred-beasts-735126 [train]
  - `decks/pool/lunalight-meta1.ydk` (41 + 15 + 15 side): Lunalight — Footscray WCQ Regional, Winner, 2026-09-27 — https://ygoprodeck.com/deck/lunalight-734787 [train]
  - `decks/pool/lunalight-meta2.ydk` (42 + 15 + 15 side): Lunalight — Chelmsford WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/lunalight-735367 [train]
  - `decks/pool/magistus-fairy-tail-meta1.ydk` (43 + 15 + 15 side): Magistus Fairy Tail — Arapiraca WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/magistus-fairy-tail-735366 [holdout]
  - `decks/pool/magistus-invoked-fairy-tail-meta1.ydk` (43 + 15 + 15 side): Magistus Invoked Fairy Tail — Cosne-Cours-sur-Loire WCQ Regional, Runner-Up, 2026-09-26 — https://ygoprodeck.com/deck/magistus-invoked-fairy-tail-735984 [train]
  - `decks/pool/magistus-invoked-fairy-tail-meta2.ydk` (43 + 15 + 15 side): Magistus Invoked Fairy Tail — Avrillé WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/magistus-invoked-fairy-tail-734806 [train]
  - `decks/pool/megalith-ryzeal-mitsurugi-meta1.ydk` (50 + 15 + 15 side): Megalith Ryzeal Mitsurugi — Campinas WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/megalith-ryzeal-mitsurugi-735560 [train]
  - `decks/pool/melodious-meta1.ydk` (40 + 15 + 15 side): Melodious — Ballarat WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/melodious-734651 [train]
  - `decks/pool/memento-meta1.ydk` (41 + 15 + 15 side): Memento — Footscray WCQ Regional, Top 4, 2026-09-27 — https://ygoprodeck.com/deck/memento-734786 [train]
  - `decks/pool/memento-meta2.ydk` (44 + 15 + 15 side): Memento — Folkestone WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/memento-735584 [train]
  - `decks/pool/mitsurugi-meta1.ydk` (40 + 15 + 15 side): Mitsurugi — Kissimmee WCQ Regional, Runner-Up, 2026-09-26 — https://ygoprodeck.com/deck/mitsurugi-735391 [train]
  - `decks/pool/mitsurugi-meta2.ydk` (41 + 15 + 15 side): Mitsurugi — Santiago WCQ Regional, Runner-Up, 2026-09-26 — https://ygoprodeck.com/deck/mitsurugi-734774 [train]
  - `decks/pool/mitsurugi-clown-crew-meta1.ydk` (44 + 15 + 15 side): Mitsurugi Clown Crew — Cannington WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/mitsurugi-clown-crew-735999 [train]
  - `decks/pool/mitsurugi-light-and-darkness-ritual-orcust-meta1.ydk` (46 + 15 + 15 side): Mitsurugi Light and Darkness Ritual Orcust — YCS Paris, Top 64, 2026-09-19 — https://ygoprodeck.com/deck/mitsurugi-light-and-darkness-ritual-orcust-734625 [train]
  - `decks/pool/mitsurugi-magnet-warrior-meta1.ydk` (46 + 15 + 15 side): Mitsurugi Magnet Warrior — Saskatoon WCQ Regional, Top 4, 2026-09-20 — https://ygoprodeck.com/deck/mitsurugi-magnet-warrior-735121 [train]
  - `decks/pool/mitsurugi-orcust-phantom-knights-meta1.ydk` (60 + 15 + 15 side): Mitsurugi Orcust Phantom Knights — Auckland WCQ Regional, Runner-Up, 2026-09-19 — https://ygoprodeck.com/deck/mitsurugi-orcust-phantom-knights-735410 [train]
  - `decks/pool/mitsurugi-yummy-light-and-darkness-ritual-meta1.ydk` (55 + 15 + 15 side): Mitsurugi Yummy Light and Darkness Ritual — Buenos Aires WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/mitsurugi-yummy-light-and-darkness-ritual-736022 [train]
  - `decks/pool/odion-meta1.ydk` (40 + 15 + 15 side): Odion — Rochester WCQ Regional, Runner-Up, 2026-09-20 — https://ygoprodeck.com/deck/odion-734308 [train]
  - `decks/pool/onomat-constellar-tellarknights-meta1.ydk` (41 + 15 + 15 side): Onomat Constellar Tellarknights — Thessaloniki WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/onomat-constellar-tellarknights-735208 [train]
  - `decks/pool/onomat-ryzeal-meta1.ydk` (41 + 15 + 15 side): Onomat Ryzeal — Newry WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/onomat-ryzeal-735177 [train]
  - `decks/pool/orcust-meta1.ydk` (40 + 15 + 15 side): Orcust — Rende WCQ Regional, Top 4, 2026-09-27 — https://ygoprodeck.com/deck/orcust-735396 [train]
  - `decks/pool/orcust-meta2.ydk` (40 + 15 + 15 side): Orcust — Newry WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/orcust-735104 [train]
  - `decks/pool/power-patron-meta1.ydk` (47 + 15 + 15 side): Power Patron — Folkestone WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/power-patron-734801 [train]
  - `decks/pool/power-patron-artmage-meta1.ydk` (41 + 15 + 15 side): Power Patron Artmage — Campinas WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/power-patron-artmage-735393 [train]
  - `decks/pool/power-patron-artmage-meta2.ydk` (42 + 15 + 15 side): Power Patron Artmage — Ioannina WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/power-patron-artmage-735221 [train]
  - `decks/pool/r-b-meta1.ydk` (42 + 15 + 15 side): R.B. — Riyadh WCQ Regional, Winner, 2026-09-25 — https://ygoprodeck.com/deck/r-b-734796 [train]
  - `decks/pool/radiant-typhoon-meta1.ydk` (42 + 15 + 15 side): Radiant Typhoon — Campinas WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/radiant-typhoon-736002 [train]
  - `decks/pool/radiant-typhoon-meta2.ydk` (40 + 15 + 15 side): Radiant Typhoon — Olathe WCQ Regional, Top 4, 2026-09-19 — https://ygoprodeck.com/deck/radiant-typhoon-734838 [train]
  - `decks/pool/regenesis-meta1.ydk` (40 + 15 + 15 side): Regenesis — Cannington WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/regenesis-736001 [train]
  - `decks/pool/ryzeal-mitsurugi-meta1.ydk` (42 + 15 + 15 side): Ryzeal Mitsurugi — Caracas WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/ryzeal-mitsurugi-735497 [train]
  - `decks/pool/sacred-beasts-meta1.ydk` (42 + 15 + 15 side): Sacred Beasts — Arapiraca WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/sacred-beasts-735378 [train]
  - `decks/pool/sacred-beasts-meta2.ydk` (40 + 15 + 15 side): Sacred Beasts — Donnezac WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/sacred-beasts-735147 [train]
  - `decks/pool/sky-striker-meta1.ydk` (40 + 15 + 15 side): Sky Striker — Caracas WCQ Regional, Top 4, 2026-09-26 — https://ygoprodeck.com/deck/sky-striker-735502 [train]
  - `decks/pool/sky-striker-meta2.ydk` (40 + 15 + 15 side): Sky Striker — Thessaloniki WCQ Regional, Top 8, 2026-09-27 — https://ygoprodeck.com/deck/sky-striker-735215 [train]
  - `decks/pool/sky-striker-r-b-meta1.ydk` (42 + 15 + 15 side): Sky Striker R.B. — Buenos Aires WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/sky-striker-r-b-735641 [train]
  - `decks/pool/synchron-elfnote-meta1.ydk` (40 + 15 + 15 side): Synchron Elfnote — Poznań WCQ Regional, Winner, 2026-09-26 — https://ygoprodeck.com/deck/synchron-elfnote-734793 [train]
  - `decks/pool/synchron-elfnote-meta2.ydk` (42 + 15 + 15 side): Synchron Elfnote — Cannington WCQ Regional, Top 4, 2026-09-27 — https://ygoprodeck.com/deck/synchron-elfnote-736112 [train]
  - `decks/pool/synchron-kewl-tune-meta1.ydk` (40 + 15 + 15 side): Synchron Kewl Tune — YCS Paris, Top 32, 2026-09-19 — https://ygoprodeck.com/deck/synchron-kewl-tune-734629 [train]
  - `decks/pool/trains-meta1.ydk` (40 + 15 + 15 side): Trains — Newry WCQ Regional, Top 8, 2026-09-26 — https://ygoprodeck.com/deck/trains-734807 [train]
  - `decks/pool/trains-meta2.ydk` (40 + 15 + 15 side): Trains — YCS Paris, Top 16, 2026-09-19 — https://ygoprodeck.com/deck/trains-735141 [train]
