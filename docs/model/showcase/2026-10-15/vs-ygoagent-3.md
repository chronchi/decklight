# Our AI vs ygo-agent, game 3: ygo-agent (CenturIon) vs Our AI (Branded)

- Goes first: ygo-agent (CenturIon)
- Result: **ygo-agent (CenturIon) wins** on turn 7
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 6001; 201 decisions in total, 18 searched, 10 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: ygo-agent (CenturIon)'s turn  (LP ygo-agent 8000 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (5): Ash Blossom & Joyous Spring, Infinite Impermanence, Stand Up Centur-Ion!, Centur-Ion True Awakening, Stand Up Centur-Ion!
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 15
- **Our AI**
  - hand (5): Ash Blossom & Joyous Spring, Foolish Burial, Bystial Saronir, Maxx "C", Called by the Grave
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 41, extra deck 15

Plays:

- ygo-agent sets Stand Up Centur-Ion!
- ygo-agent activates Stand Up Centur-Ion!
- ygo-agent activates Stand Up Centur-Ion! ("Place 1 "Centur-Ion" monster in your Spell & Trap Zone")
- ygo-agent selects Stand Up Centur-Ion!  _[Select the card(s) to send to the GY]_
- ygo-agent selects Centur-Ion Trudea  _[Select the card(s) to place on the field]_
- ygo-agent sets Infinite Impermanence
- ygo-agent sets Centur-Ion True Awakening
- ygo-agent ends the turn

## Turn 2: Our AI (Branded)'s turn  (LP ygo-agent 8000 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (1): Ash Blossom & Joyous Spring
  - monsters: -
  - spells/traps: Centur-Ion Trudea, Infinite Impermanence (set), Centur-Ion True Awakening (set), Stand Up Centur-Ion!
  - GY (1): Stand Up Centur-Ion!
  - deck 34, extra deck 15
- **Our AI**
  - hand (6): Ash Blossom & Joyous Spring, Foolish Burial, Bystial Saronir, Maxx "C", Called by the Grave, Keeper of Dragon Magic
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 40, extra deck 15

Plays:

- Our AI activates Foolish Burial
- ygo-agent activates Ash Blossom & Joyous Spring ("Negate that effect")
- Our AI activates Maxx "C" ("Draw 1 card for eatch Special Summoned card on opponent's Field")  **(search: chose this over the network's first choice: activates Called by the Grave)**
- Our AI Normal Summons Keeper of Dragon Magic
- Our AI uses the effect of Keeper of Dragon Magic
- Our AI selects Bystial Saronir  _[Select the card(s) to discard]_
- ygo-agent activates Infinite Impermanence
- Our AI uses the effect of Bystial Saronir
- Our AI selects Branded Fusion  _[Select the card(s) to send to the GY]_
- Our AI sets Called by the Grave
- Our AI enters the Battle Phase
- Our AI attacks with Keeper of Dragon Magic
- Our AI goes to Main Phase 2

## Turn 3: ygo-agent (CenturIon)'s turn  (LP ygo-agent 6200 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (1): Droll & Lock Bird
  - monsters: -
  - spells/traps: Centur-Ion Trudea, Centur-Ion True Awakening (set), Stand Up Centur-Ion!
  - GY (3): Stand Up Centur-Ion!, Ash Blossom & Joyous Spring, Infinite Impermanence
  - deck 33, extra deck 15
- **Our AI**
  - hand (1): Ash Blossom & Joyous Spring
  - monsters: Keeper of Dragon Magic (ATK)
  - spells/traps: Called by the Grave (set)
  - GY (4): Maxx "C", Foolish Burial, Bystial Saronir, Branded Fusion
  - deck 39, extra deck 15

Plays:

- ygo-agent activates Centur-Ion Trudea ("Special Summon this card")
- ygo-agent chooses position face-up Attack
- ygo-agent answers YES ("Increase its Level by 4?")
- ygo-agent activates Centur-Ion Trudea ("Place this card and 1 "Centur-Ion" monster in the Spell & Trap Zone")
- ygo-agent selects Centur-Ion Primera  _[Select the card(s) to place on the field]_
- ygo-agent activates Centur-Ion Primera ("Special Summon this card")
- ygo-agent chooses position face-up Attack
- ygo-agent declines the effect of Centur-Ion Primera
- ygo-agent enters the Battle Phase
- ygo-agent goes to Main Phase 2
- ygo-agent ends the turn

## Turn 4: Our AI (Branded)'s turn  (LP ygo-agent 6200 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (1): Droll & Lock Bird
  - monsters: Centur-Ion Primera (ATK)
  - spells/traps: Centur-Ion Trudea, Centur-Ion True Awakening (set), Stand Up Centur-Ion!
  - GY (3): Stand Up Centur-Ion!, Ash Blossom & Joyous Spring, Infinite Impermanence
  - deck 32, extra deck 15
- **Our AI**
  - hand (2): Ash Blossom & Joyous Spring, Infinite Impermanence
  - monsters: Keeper of Dragon Magic (ATK)
  - spells/traps: Called by the Grave (set)
  - GY (4): Maxx "C", Foolish Burial, Bystial Saronir, Branded Fusion
  - deck 38, extra deck 15

Plays:

- Our AI sets monster Ash Blossom & Joyous Spring  **(search: chose this over the network's first choice: sets Infinite Impermanence)**
- Our AI ends the turn  **(search: chose this over the network's first choice: sets Infinite Impermanence)**

## Turn 5: ygo-agent (CenturIon)'s turn  (LP ygo-agent 6200 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (2): Droll & Lock Bird, Maxx "C"
  - monsters: Centur-Ion Primera (ATK)
  - spells/traps: Centur-Ion Trudea, Centur-Ion True Awakening (set), Stand Up Centur-Ion!
  - GY (3): Stand Up Centur-Ion!, Ash Blossom & Joyous Spring, Infinite Impermanence
  - deck 31, extra deck 15
- **Our AI**
  - hand (1): Infinite Impermanence
  - monsters: Ash Blossom & Joyous Spring (face-down DEF), Keeper of Dragon Magic (ATK)
  - spells/traps: Called by the Grave (set)
  - GY (4): Maxx "C", Foolish Burial, Bystial Saronir, Branded Fusion
  - deck 38, extra deck 15

Plays:

- ygo-agent activates Centur-Ion Trudea ("Special Summon this card")
- ygo-agent chooses position face-up Attack
- ygo-agent answers YES ("Increase its Level by 4?")
- ygo-agent declines the effect of Stand Up Centur-Ion!
- ygo-agent activates Centur-Ion Trudea ("Place this card and 1 "Centur-Ion" monster in the Spell & Trap Zone")
- ygo-agent selects Centur-Ion Emeth VI  _[Select the card(s) to place on the field]_
- ygo-agent activates Centur-Ion Emeth VI ("Special Summon this card")
- Our AI activates Called by the Grave  **(search: chose this over the network's first choice: passes)**
- ygo-agent activates Maxx "C" ("Draw 1 card for eatch Special Summoned card on opponent's Field")
- ygo-agent chooses position face-up Attack
- ygo-agent declines the effect of Stand Up Centur-Ion!
- ygo-agent Special Summons Centur-Ion Auxila
- ygo-agent selects Centur-Ion Primera  _[Select the card(s) to use as Synchro Material]_
- ygo-agent selects Centur-Ion Emeth VI  _[Select the card(s) to use as Synchro Material]_
- ygo-agent chooses position face-up Attack
- ygo-agent uses the effect of Centur-Ion Auxila
- ygo-agent selects Centur-Ion Primera  _[Select the card(s) to add to your hand]_
- ygo-agent Normal Summons Centur-Ion Primera
- ygo-agent uses the effect of Centur-Ion Primera
- ygo-agent selects Centur-Ion Gargoyle II  _[Select the card(s) to add to your hand]_
- ygo-agent activates Centur-Ion Gargoyle II ("Special Summon this card from your hand")
- ygo-agent selects Stand Up Centur-Ion!  _[Select the card(s) to send to the GY]_
- ygo-agent chooses position face-up Attack
- ygo-agent Special Summons Crimson Dragon
- ygo-agent selects Centur-Ion Primera  _[Select the card(s) to use as Synchro Material]_
- ygo-agent selects Centur-Ion Gargoyle II  _[Select the card(s) to use as Synchro Material]_
- ygo-agent chooses position face-up Defense
- ygo-agent uses the effect of Centur-Ion Gargoyle II
- ygo-agent activates Crimson Dragon ("Return this card to the Extra Deck")
- ygo-agent selects Centur-Ion Auxila  _[Select the card(s) to Special Summon]_
- ygo-agent chooses position face-up Attack
- ygo-agent enters the Battle Phase
- ygo-agent attacks with Centur-Ion Auxila
- ygo-agent selects opponent's face-down card (Monster Zone)  _[Select an attack target]_
- ygo-agent attacks with Centur-Ion Auxila
- ygo-agent selects Keeper of Dragon Magic  _[Select an attack target]_
- ygo-agent ends the turn
- Our AI passes  **(search: chose this over the network's first choice: activates Infinite Impermanence)**
- Our AI activates Infinite Impermanence  **(search: chose this over the network's first choice: passes)**
- Our AI selects Centur-Ion Auxila  _[Select the card(s) to negate its effects]_
- ygo-agent activates Centur-Ion Auxila ("Place 1 of your non-Synchro "Centur-Ion" monsters that is banished or in your GY face-up in your Spell & Trap Zone as a Continuous Trap")
- ygo-agent selects Centur-Ion Primera  _[Select the card(s) to place on the field]_

## Turn 6: Our AI (Branded)'s turn  (LP ygo-agent 6200 / Our AI 6800)

At the start of the turn:

- **ygo-agent**
  - hand (2): Centur-Ion Gargoyle II, Droll & Lock Bird
  - monsters: Centur-Ion Auxila (ATK), Centur-Ion Auxila (ATK)
  - spells/traps: Centur-Ion Trudea, Centur-Ion Primera, Centur-Ion True Awakening (set)
  - GY (6): Stand Up Centur-Ion!, Infinite Impermanence, Maxx "C", Centur-Ion Primera, Centur-Ion Emeth VI, Stand Up Centur-Ion!
  - banished (1): Ash Blossom & Joyous Spring
  - deck 28, extra deck 13
- **Our AI**
  - hand (1): Branded Expulsion
  - monsters: -
  - spells/traps: -
  - GY (8): Maxx "C", Foolish Burial, Bystial Saronir, Branded Fusion, Called by the Grave, Ash Blossom & Joyous Spring, Keeper of Dragon Magic, Infinite Impermanence
  - deck 37, extra deck 15

Plays:

- Our AI sets Branded Expulsion
- Our AI enters the Battle Phase
- Our AI goes to Main Phase 2

## Turn 7: ygo-agent (CenturIon)'s turn  (LP ygo-agent 6200 / Our AI 6800)

At the start of the turn:

- **ygo-agent**
  - hand (3): Centur-Ion Gargoyle II, Droll & Lock Bird, Centur-Ion Bonds
  - monsters: Centur-Ion Auxila (ATK), Centur-Ion Auxila (ATK)
  - spells/traps: Centur-Ion Trudea, Centur-Ion Primera, Centur-Ion True Awakening (set)
  - GY (6): Stand Up Centur-Ion!, Infinite Impermanence, Maxx "C", Centur-Ion Primera, Centur-Ion Emeth VI, Stand Up Centur-Ion!
  - banished (1): Ash Blossom & Joyous Spring
  - deck 27, extra deck 13
- **Our AI**
  - hand (0): -
  - monsters: -
  - spells/traps: Branded Expulsion (set)
  - GY (8): Maxx "C", Foolish Burial, Bystial Saronir, Branded Fusion, Called by the Grave, Ash Blossom & Joyous Spring, Keeper of Dragon Magic, Infinite Impermanence
  - deck 37, extra deck 15

Plays:

- ygo-agent activates Centur-Ion Bonds ("Place 1 "Centur-Ion" monster in your Spell & Trap Zone")
- ygo-agent selects Centur-Ion Emeth VI  _[Select the card(s) to place on the field]_
- ygo-agent activates Centur-Ion Trudea ("Special Summon this card")
- ygo-agent activates Centur-Ion Primera ("Special Summon this card")
- ygo-agent activates Centur-Ion Emeth VI ("Special Summon this card")
- ygo-agent chooses position face-up Attack
- ygo-agent chooses position face-up Attack
- ygo-agent chooses position face-up Attack
- ygo-agent answers NO ("Increase its Level by 4?")
- ygo-agent declines the effect of Centur-Ion Primera
- ygo-agent enters the Battle Phase
- ygo-agent attacks with Centur-Ion Auxila
- ygo-agent attacks with Centur-Ion Auxila
- ygo-agent attacks with Centur-Ion Emeth VI

## Game over: ygo-agent (CenturIon) wins (LP before the final blow: ygo-agent 6200 / Our AI 800)
