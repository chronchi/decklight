# Our AI vs ygo-agent, game 2: ygo-agent (SnakeEyeTear) vs Our AI (SnakeEyeFire)

- Goes first: ygo-agent (SnakeEyeTear)
- Result: **Our AI (SnakeEyeFire) wins** on turn 6
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 6003; 116 decisions in total, 46 searched, 20 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: ygo-agent (SnakeEyeTear)'s turn  (LP ygo-agent 8000 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (5): Snake-Eyes Flamberge Dragon, Nibiru, the Primal Being, Tearlaments Scheiren, Ash Blossom & Joyous Spring, Tearlaments Metanoise
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 37, extra deck 15
- **Our AI**
  - hand (5): Called by the Grave, Snake-Eyes Poplar, Fire King Sanctuary, Nibiru, the Primal Being, Fire King Island
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 15

Plays:

- ygo-agent activates Tearlaments Scheiren ("Special Summon")
- ygo-agent chooses position face-up Attack
- ygo-agent selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to send to the GY]_
- ygo-agent uses the effect of Snake-Eyes Poplar
- ygo-agent selects Snake-Eyes Flamberge Dragon  _[Select the target(s) of the effect]_
- ygo-agent sets Tearlaments Metanoise
- ygo-agent ends the turn

## Turn 2: Our AI (SnakeEyeFire)'s turn  (LP ygo-agent 8000 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (2): Nibiru, the Primal Being, Ash Blossom & Joyous Spring
  - monsters: Tearlaments Scheiren (ATK)
  - spells/traps: Snake-Eyes Flamberge Dragon, Tearlaments Metanoise (set)
  - GY (3): Ash Blossom & Joyous Spring, Fossil Dyna Pachycephalo, Snake-Eyes Poplar
  - deck 34, extra deck 15
- **Our AI**
  - hand (6): Called by the Grave, Snake-Eyes Poplar, Fire King Sanctuary, Nibiru, the Primal Being, Fire King Island, Fire King High Avatar Kirin
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 34, extra deck 15

Plays:

- Our AI Normal Summons Snake-Eyes Poplar
- Our AI uses the effect of Snake-Eyes Poplar
- ygo-agent activates Ash Blossom & Joyous Spring ("Negate that effect")
- Our AI passes  **(search: chose this over the network's first choice: activates Called by the Grave)**
- Our AI activates Fire King Sanctuary ("Place 1 "Fire King Island" from your Deck face-up in your Field Zone?")
- Our AI activates Fire King Island
- Our AI activates Fire King Island ("Destroy 1 monster and search 1 "Fire King" monster")
- Our AI selects Fire King High Avatar Kirin  _[Select the card(s) to destroy]_  **(search: chose this over the network's first choice: selects Nibiru, the Primal Being)**
- Our AI selects Legendary Fire King Ponix  _[Select the card(s) to add to your hand]_
- Our AI activates Legendary Fire King Ponix ("Special Summon this card from your hand")
- Our AI activates Fire King High Avatar Kirin ("Special Summon 1 "Fire King" from your hand or GY")
- Our AI chooses position face-up Attack
- Our AI answers YES ("Destroy 1 card on the field?")
- Our AI selects opponent's face-down card (Spell/Trap Zone)  _[Select the card(s) to destroy]_  **(search: chose this over the network's first choice: selects Tearlaments Scheiren)**
- Our AI uses the effect of Legendary Fire King Ponix
- Our AI Special Summons Hiita the Fire Charmer, Ablaze  **(search: chose this over the network's first choice: Special Summons Linkuriboh)**
- Our AI selects Snake-Eyes Poplar  _[Select the card(s) to use as Link Material]_
- Our AI selects Legendary Fire King Ponix  _[Select the card(s) to use as Link Material]_
- Our AI uses the effect of Snake-Eyes Poplar
- Our AI selects Snake-Eyes Poplar  _[Select the target(s) of the effect]_  **(search: chose this over the network's first choice: selects Fire King High Avatar Kirin)**
- Our AI activates Hiita the Fire Charmer, Ablaze
- Our AI selects Ash Blossom & Joyous Spring  _[Select the card(s) to Special Summon]_  **(search: chose this over the network's first choice: selects Snake-Eyes Poplar)**
- Our AI chooses position face-up Defense
- Our AI activates Called by the Grave  **(search: chose this over the network's first choice: passes)**
- Our AI selects Snake-Eyes Poplar  _[Select the card(s) to banish]_  **(search: chose this over the network's first choice: selects Fossil Dyna Pachycephalo)**
- Our AI Special Summons Promethean Princess, Bestower of Flames
- Our AI selects Ash Blossom & Joyous Spring  _[Select the card(s) to use as Link Material]_
- Our AI selects Hiita the Fire Charmer, Ablaze  _[Select the card(s) to use as Link Material]_
- ygo-agent activates Nibiru, the Primal Being
- ygo-agent chooses position face-up Attack
- ygo-agent chooses position face-up Attack
- Our AI sets Fire King Sky Burn
- Our AI enters the Battle Phase
- Our AI attacks with Primal Being Token
- Our AI selects Nibiru, the Primal Being  _[Select an attack target]_
- Our AI goes to Main Phase 2

## Turn 3: ygo-agent (SnakeEyeTear)'s turn  (LP ygo-agent 6500 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (1): Revolution Synchron
  - monsters: -
  - spells/traps: Snake-Eyes Flamberge Dragon
  - GY (6): Ash Blossom & Joyous Spring, Fossil Dyna Pachycephalo, Tearlaments Metanoise, Ash Blossom & Joyous Spring, Tearlaments Scheiren, Nibiru, the Primal Being
  - banished (1): Snake-Eyes Poplar
  - deck 33, extra deck 15
- **Our AI**
  - hand (1): Nibiru, the Primal Being
  - monsters: Primal Being Token (ATK)
  - spells/traps: Fire King Sky Burn (set), Fire King Sanctuary, Snake-Eyes Poplar, Fire King Island
  - GY (5): Fire King High Avatar Kirin, Legendary Fire King Ponix, Called by the Grave, Hiita the Fire Charmer, Ablaze, Promethean Princess, Bestower of Flames
  - deck 32, extra deck 13

Plays:

- ygo-agent sets monster Revolution Synchron
- ygo-agent enters the Battle Phase
- ygo-agent goes to Main Phase 2

## Turn 4: Our AI (SnakeEyeFire)'s turn  (LP ygo-agent 6500 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (0): -
  - monsters: Revolution Synchron (face-down DEF)
  - spells/traps: Snake-Eyes Flamberge Dragon
  - GY (6): Ash Blossom & Joyous Spring, Fossil Dyna Pachycephalo, Tearlaments Metanoise, Ash Blossom & Joyous Spring, Tearlaments Scheiren, Nibiru, the Primal Being
  - banished (1): Snake-Eyes Poplar
  - deck 33, extra deck 15
- **Our AI**
  - hand (2): Nibiru, the Primal Being, Snake-Eye Ash
  - monsters: Primal Being Token (ATK)
  - spells/traps: Fire King Sky Burn (set), Fire King Sanctuary, Snake-Eyes Poplar, Fire King Island
  - GY (5): Fire King High Avatar Kirin, Legendary Fire King Ponix, Called by the Grave, Hiita the Fire Charmer, Ablaze, Promethean Princess, Bestower of Flames
  - deck 31, extra deck 13

Plays:

- Our AI Normal Summons Snake-Eye Ash  **(search: chose this over the network's first choice: activates Fire King Island ("Destroy 1 monster and search 1 "Fire King" monster"))**
- Our AI uses the effect of Snake-Eye Ash
- Our AI selects Snake-Eye Ash  _[Select the card(s) to add to your hand]_  **(search: chose this over the network's first choice: selects Snake-Eyes Poplar)**
- Our AI activates Fire King Island ("Destroy 1 monster and search 1 "Fire King" monster")
- Our AI selects Nibiru, the Primal Being  _[Select the card(s) to destroy]_
- Our AI selects Sacred Fire King Garunix  _[Select the card(s) to add to your hand]_
- Our AI activates Snake-Eye Ash ("Special Summon 1 "Snake-Eye" from your hand or Deck")
- Our AI selects Fire King Sanctuary  _[Select the card(s) to send to the GY]_  **(search: chose this over the network's first choice: selects Snake-Eyes Poplar)**
- Our AI selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to Special Summon]_
- Our AI chooses position face-up Attack
- Our AI activates Snake-Eyes Flamberge Dragon ("Place 1 monster in its owner's Spell & Trap Zone as a Continuous Spell")
- Our AI selects Fire King High Avatar Kirin  _[Select the target(s) of the effect]_
- Our AI enters the Battle Phase
- Our AI attacks with Snake-Eyes Flamberge Dragon
- Our AI selects opponent's face-down card (Monster Zone)  _[Select an attack target]_
- Our AI attacks with Primal Being Token
- Our AI ends the turn  **(search: chose this over the network's first choice: goes to Main Phase 2)**

## Turn 5: ygo-agent (SnakeEyeTear)'s turn  (LP ygo-agent 2000 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (1): Called by the Grave
  - monsters: -
  - spells/traps: Snake-Eyes Flamberge Dragon
  - GY (7): Ash Blossom & Joyous Spring, Fossil Dyna Pachycephalo, Tearlaments Metanoise, Ash Blossom & Joyous Spring, Tearlaments Scheiren, Nibiru, the Primal Being, Revolution Synchron
  - banished (1): Snake-Eyes Poplar
  - deck 32, extra deck 15
- **Our AI**
  - hand (2): Snake-Eye Ash, Sacred Fire King Garunix
  - monsters: Snake-Eyes Flamberge Dragon (ATK), Primal Being Token (ATK)
  - spells/traps: Fire King High Avatar Kirin, Fire King Sky Burn (set), Snake-Eyes Poplar, Fire King Island
  - GY (7): Legendary Fire King Ponix, Called by the Grave, Hiita the Fire Charmer, Ablaze, Promethean Princess, Bestower of Flames, Nibiru, the Primal Being, Snake-Eye Ash, Fire King Sanctuary
  - deck 28, extra deck 13

Plays:

- ygo-agent activates Called by the Grave
- ygo-agent selects Promethean Princess, Bestower of Flames  _[Select the card(s) to banish]_
- Our AI activates Snake-Eyes Flamberge Dragon ("Special Summon 1 Monster Card treated as a Continuous Spell")
- Our AI selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to Special Summon]_
- Our AI chooses position face-up Attack
- ygo-agent enters the Battle Phase
- ygo-agent goes to Main Phase 2

## Turn 6: Our AI (SnakeEyeFire)'s turn  (LP ygo-agent 2000 / Our AI 8000)

At the start of the turn:

- **ygo-agent**
  - hand (0): -
  - monsters: -
  - spells/traps: -
  - GY (8): Ash Blossom & Joyous Spring, Fossil Dyna Pachycephalo, Tearlaments Metanoise, Ash Blossom & Joyous Spring, Tearlaments Scheiren, Nibiru, the Primal Being, Revolution Synchron, Called by the Grave
  - banished (1): Snake-Eyes Poplar
  - deck 32, extra deck 15
- **Our AI**
  - hand (3): Snake-Eye Ash, Sacred Fire King Garunix, Droll & Lock Bird
  - monsters: Snake-Eyes Flamberge Dragon (ATK), Primal Being Token (ATK), Snake-Eyes Flamberge Dragon (ATK)
  - spells/traps: Fire King High Avatar Kirin, Fire King Sky Burn (set), Snake-Eyes Poplar, Fire King Island
  - GY (6): Legendary Fire King Ponix, Called by the Grave, Hiita the Fire Charmer, Ablaze, Nibiru, the Primal Being, Snake-Eye Ash, Fire King Sanctuary
  - banished (1): Promethean Princess, Bestower of Flames
  - deck 27, extra deck 13

Plays:

- Our AI enters the Battle Phase
- Our AI attacks with Snake-Eyes Flamberge Dragon

## Game over: Our AI (SnakeEyeFire) wins (LP before the final blow: ygo-agent 2000 / Our AI 8000)
