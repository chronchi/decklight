# New-cards test decks (ADR 0007)

Evaluation-only decks built from archetypes with **zero** cards in any training
data. Never train on these; the report card's "new cards" test uses them to
measure generalization to unseen cards. All non-staple cards verified unseen
via `training_cards()` (`scripts/card_holdout.py newcards`).

- `infernity.ydk` — Infernity, combo. Empty the hand, then loop Launcher/Necromancer/Mirage into Doom Dragon synchros behind Barrier. List from knowledge of classic Infernity combo builds (DuelingNexus/DB Grinder era). 21 non-staple cards, all unseen.
- `dragunity.ydk` — Dragunity, combo. Senatus/Remus tutor small dragons as equips, climb through Gae Dearg/Ascalon synchros with Divine Lance/Whirlwind extension. List from knowledge of modern Dragunity combo builds (Master Duel meta sites). 24 non-staple cards, all unseen.
- `dinomorphia.ydk` — Dinomorphia, control. Pay half LP on Therizia/Diplos/Frenzy to fuse Kentregina/Rexterm, back them with Solemns and hand traps. List from knowledge of Dinomorphia trap-control builds (YGOPRODeck/DB). 13 non-staple cards, all unseen.
- `ragnaraika.ydk` — Ragnaraika, midrange (2024, Legacy of Destruction). Evil Seed/Samurai Beetle swarm plants/insects into Stag Sovereign/Mantis Monk xyz, Bloom/Hunting Dance grind. List from knowledge of post-LEDE Ragnaraika builds (YGOPRODeck). 12 non-staple cards, all unseen.
- `goldpride.ydk` — Gold Pride, midrange (2023, Cyberstorm Access). Take damage to trigger Leon/Captain Carrie, synchro climb into Chariot Carrie/Star Leon with Start Your Engines/Pedal to the Metal. List from knowledge of Gold Pride builds (Master Duel Meta). 16 non-staple cards, all unseen.
- `sforce.ydk` — S-Force, control. Gravitino/Pla-Tina/Rappa Chiyomaru column-lock with Bridgehead/Chase/Showdown disruption into Justify/Nightchaser links. List from knowledge of S-Force control builds (YGOPRODeck). 18 non-staple cards, all unseen.
