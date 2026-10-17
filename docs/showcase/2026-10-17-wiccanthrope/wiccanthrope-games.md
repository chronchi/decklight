# ygo-zero plays Wiccanthrope (custom cards)

Deck: Zedja's **Wiccanthrope** custom cards ([zedja123/ZedjaCustomCards](https://github.com/zedja123/ZedjaCustomCards)),
list rebuilt from the deck image (40 main / 14 extra; one Extra Deck code, `90448281`, is unknown to every card
database and was left out).

**The AI had never seen these cards.** It was trained only on official cards. Here it reads the custom cards'
texts through a text embedding, plus their Lua scripts' effect metadata, and plays them with the same network and
search it uses for everything else. Nothing about Wiccanthrope was trained or hand-coded.

- Setup: the full product (policy network + value network + search) on **both** sides, so each game is the AI
  against itself, Wiccanthrope vs 3 meta decks, both seats, 2 different deals per seat: 12 games.
- Engine: EDOPro's ygopro-core with Zedja's own card scripts. Every Wiccanthrope card in the deck was activated
  or summoned during testing, with 0 script errors.

## Results: Wiccanthrope won 7 of 12

| Wiccanthrope… | Won |
|---|---|
| going first | 1 / 6 |
| going second | 6 / 6 |

| # | Opponent | Wiccanthrope goes | Winner | Turn | LP before the final blow |
|---|---|---|---|---|---|
| 1 | Snake-Eye | first | **Wiccanthrope** | 3 | Wiccanthrope 3400 / Snake-Eye 2100 |
| 2 | Snake-Eye | first | Snake-Eye | 4 | Wiccanthrope 700 / Snake-Eye 8000 |
| 3 | Snake-Eye | second | **Wiccanthrope** | 2 | Snake-Eye 1300 / Wiccanthrope 8000 |
| 4 | Snake-Eye | second | **Wiccanthrope** | 2 | Snake-Eye 1600 / Wiccanthrope 8000 |
| 5 | Mitsurugi | first | Mitsurugi | 4 | Wiccanthrope 600 / Mitsurugi 6500 |
| 6 | Mitsurugi | first | Mitsurugi | 2 | Wiccanthrope 400 / Mitsurugi 8000 |
| 7 | Mitsurugi | second | **Wiccanthrope** | 2 | Mitsurugi 2600 / Wiccanthrope 8000 |
| 8 | Mitsurugi | second | **Wiccanthrope** | 4 | Mitsurugi 900 / Wiccanthrope 8000 |
| 9 | Tearlaments | first | Tearlaments | 4 | Wiccanthrope 900 / Tearlaments 8000 |
| 10 | Tearlaments | first | Tearlaments | 2 | Wiccanthrope 1200 / Tearlaments 8000 |
| 11 | Tearlaments | second | **Wiccanthrope** | 2 | Tearlaments 3500 / Wiccanthrope 8000 |
| 12 | Tearlaments | second | **Wiccanthrope** | 4 | Tearlaments 2000 / Wiccanthrope 1000 |

## What we saw
- **Going second it is very strong.** It answers the opponent's turn-1 combo with Mulcharmy Purulia/Fuwalos,
  Effect Veiler, Ash Blossom and Impermanence. On its own turn it runs a real Wiccanthrope line and wins on turn 2 (4 games) or turn 4 (2 games):
  Rothockear (summon from hand), Enchantment, Serwapentar/Walioright, Xyz into Spedallacer (negate), Sperelfler and
  Vilecoil, GY revives, then attacks.
- **Going first it has little to do.** Half the main deck is hand traps and Mulcharmy, which do nothing on turn 1.
  The AI usually opens with a Mulcharmy plus one Wiccanthrope monster and passes, and the meta decks win on their
  first or second turn (turn 2 or 4). Game 1 is the exception: it survives at 3400 LP and wins on turn 3 with a full
  Xyz chain.
- Games are short, 2–4 turns, because both sides are aggressive combo decks played by the same AI.

## Caveats
- 12 games is a small sample: the seat split is clear, the exact win rate is not.
- The AI knows its opponent's decklist (not the hand), as in all our tests.
- The search sometimes overrides the network's first choice; the logs below mark those moves "(search: chose this
  over the network's first choice: …)".

## Full game logs
Click a game to expand it. Each turn lists the hands and board, then every action in order.

<details><summary><b>Game 1</b>: Wiccanthrope (first) vs Snake-Eye: Wiccanthrope wins on turn 3</summary>


- Goes first: Wiccanthrope 
- Result: **Wiccanthrope  wins** on turn 3
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261017; 114 decisions in total, 54 searched, 24 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Wiccanthrope 's turn  (LP Wiccanthrope 8000 / Snake-Eye 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (5): Effect Veiler, Wiccanthrope Befaist, Mulcharmy Fuwalos, Mulcharmy Purulia, Ash Blossom & Joyous Spring
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14
- **Snake-Eye**
  - hand (5): Snake-Eye Ash, Ash Blossom & Joyous Spring, Snake-Eyes Flamberge Dragon, Original Sinful Spoils - Snake-Eye, Fire King High Avatar Kirin
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15

Plays:

- Wiccanthrope activates Mulcharmy Purulia ("Apply the "●" effects for the rest of the turn")
- Wiccanthrope Normal Summons Wiccanthrope Befaist
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Snake-Eye activates Ash Blossom & Joyous Spring ("Negate that effect")

## Turn 2: Snake-Eye 's turn  (LP Wiccanthrope 8000 / Snake-Eye 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (3): Effect Veiler, Mulcharmy Fuwalos, Ash Blossom & Joyous Spring
  - monsters: Wiccanthrope Befaist (ATK)
  - spells/traps: -
  - GY (1): Mulcharmy Purulia
  - deck 35, extra deck 14
- **Snake-Eye**
  - hand (5): Snake-Eye Ash, Snake-Eyes Flamberge Dragon, Original Sinful Spoils - Snake-Eye, Fire King High Avatar Kirin, Divine Temple of the Snake-Eye
  - monsters: -
  - spells/traps: -
  - GY (1): Ash Blossom & Joyous Spring
  - deck 35, extra deck 15

Plays:

- Snake-Eye activates Fire King High Avatar Kirin ("Destroy 1 other FIRE monster and Special Summon this card")  **(search: chose this over the network's first choice: activates Divine Temple of the Snake-Eye ("Place 1 "Snake-Eye" monster from your hand, Deck, or GY face-up in its owner's Spell & Trap Zone as a Continuous Spell?"))**
- Snake-Eye selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to destroy]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye activates Divine Temple of the Snake-Eye ("Place 1 "Snake-Eye" monster from your hand, Deck, or GY face-up in its owner's Spell & Trap Zone as a Continuous Spell?")
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster"))**
- Snake-Eye answers YES ("Place 1 "Snake-Eye" monster from your hand, Deck, or GY face-up in its owner's Spell & Trap Zone as a Continuous Spell?")
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to place on the field]_
- Snake-Eye Normal Summons Snake-Eye Ash
- Snake-Eye uses the effect of Snake-Eye Ash
- Wiccanthrope activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster")
- Wiccanthrope selects Snake-Eye Ash  _[Select a face-up card(s)]_
- Snake-Eye activates Original Sinful Spoils - Snake-Eye ("Special Summon 1 Level 1 FIRE monster from your hand/Deck")
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to send to the GY]_
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")
- Snake-Eye uses the effect of Snake-Eyes Poplar
- Snake-Eye selects Snake-Eyes Flamberge Dragon  _[Select the target(s) of the effect]_
- Snake-Eye activates Snake-Eye Ash ("Special Summon 1 "Snake-Eye" from your hand or Deck")  **(search: chose this over the network's first choice: enters the Battle Phase)**
- Snake-Eye selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to send to the GY]_
- Snake-Eye uses the effect of Snake-Eyes Flamberge Dragon
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to Special Summon]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye chooses position face-up Attack
- Snake-Eye uses the effect of Snake-Eyes Poplar
- Snake-Eye selects Original Sinful Spoils - Snake-Eye  _[Select the card(s) to add to your hand]_
- Snake-Eye sets Original Sinful Spoils - Snake-Eye  **(search: chose this over the network's first choice: enters the Battle Phase)**
- Snake-Eye enters the Battle Phase
- Snake-Eye attacks with Snake-Eye Ash  **(search: chose this over the network's first choice: attacks with Fire King High Avatar Kirin)**
- Snake-Eye selects Wiccanthrope Befaist  _[Select an attack target]_
- Snake-Eye attacks with Snake-Eyes Poplar
- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Snake-Eye attacks with Fire King High Avatar Kirin
- Snake-Eye ends the turn

## Turn 3: Wiccanthrope 's turn  (LP Wiccanthrope 3400 / Snake-Eye 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (1): Wiccanthrope Rothockear
  - monsters: -
  - spells/traps: -
  - GY (5): Mulcharmy Purulia, Effect Veiler, Ash Blossom & Joyous Spring, Wiccanthrope Befaist, Mulcharmy Fuwalos
  - deck 34, extra deck 14
- **Snake-Eye**
  - hand (0): -
  - monsters: Snake-Eyes Poplar (ATK), Snake-Eye Ash (ATK), Fire King High Avatar Kirin (ATK)
  - spells/traps: Original Sinful Spoils - Snake-Eye (set), Divine Temple of the Snake-Eye
  - GY (3): Ash Blossom & Joyous Spring, Original Sinful Spoils - Snake-Eye, Snake-Eyes Flamberge Dragon
  - deck 33, extra deck 15

Plays:

- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Attack  **(search: chose this over the network's first choice: chooses position face-up Defense)**
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Wiccanthrope selects Wiccanthrope Road  _[Select the card(s) to send to the GY]_
- Wiccanthrope Special Summons Wiccanthrope Befaist
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects Fire King High Avatar Kirin  _[Select the target(s) of the effect]_
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to banish]_
- Wiccanthrope selects Divine Temple of the Snake-Eye  _[Select the card(s) to destroy]_
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Wiccanthrope Special Summons Wiccanthrope Spedallacer
- Wiccanthrope cancels  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Befaist)**
- Wiccanthrope Special Summons Wiccanthrope Vilecoil  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Wataquera)**
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Befaist)**
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Vilecoil
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Wiccanthrope Vilecoil ("Set 1 of your banished "Wiccanthrope" Spells"))**
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Wiccanthrope Vilecoil ("Set 1 of your banished "Wiccanthrope" Spells"))**
- Wiccanthrope activates Wiccanthrope Vilecoil ("Set 1 of your banished "Wiccanthrope" Spells")
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the Xyz Material(s) to detach]_
- Wiccanthrope Special Summons Wiccanthrope Sperelfler
- Wiccanthrope selects Wiccanthrope Vilecoil  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope activates Wiccanthrope Sperelfler ("Special Summon 1 "Wiccanthrope" monster from your GY")
- Wiccanthrope selects Wiccanthrope Vilecoil  _[Select the Xyz Material(s) to detach]_
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to Special Summon]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope answers YES ("Banish 1 Spell and attach 1 Spell from your Deck to an Xyz Monster?")
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to banish]_
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope activates Wiccanthrope Rothockear ("Special Summon 1 "Wiccanthrope" monster from your GY")
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope Special Summons Wiccanthrope Sperelfler
- Wiccanthrope selects Wiccanthrope Vilecoil  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope selects Snake-Eye Ash  _[Select an attack target]_
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope selects Snake-Eyes Poplar  _[Select an attack target]_
- Snake-Eye uses the effect of Snake-Eyes Poplar
- Snake-Eye selects Snake-Eyes Flamberge Dragon  _[Select the target(s) of the effect]_
- Wiccanthrope attacks with Wiccanthrope Rothockear

## Game over: Wiccanthrope  wins (LP before the final blow: Wiccanthrope 3400 / Snake-Eye 2100)

</details>
<details><summary><b>Game 2</b>: Wiccanthrope (first) vs Snake-Eye: Snake-Eye wins on turn 4</summary>


- Goes first: Wiccanthrope 
- Result: **Snake-Eye  wins** on turn 4
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261018; 124 decisions in total, 54 searched, 29 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Wiccanthrope 's turn  (LP Wiccanthrope 8000 / Snake-Eye 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (5): Wiccanthrope Reason, Ash Blossom & Joyous Spring, Wiccanthrope Enchantment, Mulcharmy Fuwalos, Wiccanthrope Rothockear
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14
- **Snake-Eye**
  - hand (5): Snake-Eyes Poplar, Fire King Sanctuary, Fire King High Avatar Kirin, Bonfire, Divine Temple of the Snake-Eye
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15

Plays:

- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Wiccanthrope selects Wiccanthrope Road  _[Select the card(s) to send to the GY]_
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope sets Wiccanthrope Reason
- Wiccanthrope ends the turn

## Turn 2: Snake-Eye 's turn  (LP Wiccanthrope 8000 / Snake-Eye 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (1): Ash Blossom & Joyous Spring
  - monsters: Wiccanthrope Rothockear (ATK)
  - spells/traps: Wiccanthrope Reason (set)
  - GY (3): Mulcharmy Fuwalos, Wiccanthrope Road, Wiccanthrope Enchantment
  - deck 34, extra deck 14
- **Snake-Eye**
  - hand (6): Snake-Eyes Poplar, Fire King Sanctuary, Fire King High Avatar Kirin, Bonfire, Divine Temple of the Snake-Eye, Diabellstar the Black Witch
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 15

Plays:

- Snake-Eye activates Bonfire ("Search 1 Level 4 or lower Pyro monster")
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Ash Blossom & Joyous Spring ("Negate that effect"))**
- Snake-Eye selects Snake-Eye Ash  _[Select the card(s) to add to your hand]_
- Snake-Eye Normal Summons Snake-Eye Ash
- Snake-Eye uses the effect of Snake-Eye Ash
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")
- Snake-Eye activates Fire King High Avatar Kirin ("Destroy 1 other FIRE monster and Special Summon this card")  **(search: chose this over the network's first choice: passes)**
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to destroy]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye uses the effect of Snake-Eyes Poplar
- Snake-Eye activates Divine Temple of the Snake-Eye ("Place 1 "Snake-Eye" monster from your hand, Deck, or GY face-up in its owner's Spell & Trap Zone as a Continuous Spell?")
- Snake-Eye answers YES ("Place 1 "Snake-Eye" monster from your hand, Deck, or GY face-up in its owner's Spell & Trap Zone as a Continuous Spell?")
- Snake-Eye selects Snake-Eye Oak  _[Select the card(s) to place on the field]_  **(search: chose this over the network's first choice: selects Snake-Eyes Flamberge Dragon)**
- Snake-Eye activates Snake-Eye Ash ("Special Summon 1 "Snake-Eye" from your hand or Deck")  **(search: chose this over the network's first choice: Special Summons Diabellstar the Black Witch)**
- Snake-Eye selects Divine Temple of the Snake-Eye  _[Select the card(s) to send to the GY]_  **(search: chose this over the network's first choice: selects Snake-Eyes Poplar)**
- Snake-Eye chooses position face-up Attack
- Snake-Eye Special Summons Diabellstar the Black Witch
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to send to the GY]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye uses the effect of Diabellstar the Black Witch
- Snake-Eye selects Original Sinful Spoils - Snake-Eye  _[Select the card(s) to Set to the field]_
- Snake-Eye activates Original Sinful Spoils - Snake-Eye ("Special Summon 1 Level 1 FIRE monster from your hand/Deck")
- Snake-Eye selects Snake-Eye Oak  _[Select the card(s) to send to the GY]_
- Snake-Eye selects Legendary Fire King Ponix  _[Select the card(s) to Special Summon]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye uses the effect of Legendary Fire King Ponix
- Snake-Eye activates Fire King Island  **(search: chose this over the network's first choice: activates Snake-Eyes Flamberge Dragon ("Place 1 monster in its owner's Spell & Trap Zone as a Continuous Spell"))**
- Snake-Eye enters the Battle Phase  **(search: chose this over the network's first choice: activates Fire King Island ("Destroy 1 monster and search 1 "Fire King" monster"))**
- Snake-Eye attacks with Snake-Eyes Flamberge Dragon  **(search: chose this over the network's first choice: attacks with Diabellstar the Black Witch)**
- Snake-Eye selects Wiccanthrope Rothockear  _[Select an attack target]_
- Snake-Eye attacks with Fire King High Avatar Kirin  **(search: chose this over the network's first choice: attacks with Diabellstar the Black Witch)**
- Snake-Eye attacks with Diabellstar the Black Witch
- Snake-Eye attacks with Legendary Fire King Ponix
- Snake-Eye ends the turn

## Turn 3: Wiccanthrope 's turn  (LP Wiccanthrope 700 / Snake-Eye 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (1): Fire Formation - Tenki
  - monsters: -
  - spells/traps: Wiccanthrope Reason (set)
  - GY (5): Mulcharmy Fuwalos, Wiccanthrope Road, Wiccanthrope Enchantment, Ash Blossom & Joyous Spring, Wiccanthrope Rothockear
  - deck 33, extra deck 14
- **Snake-Eye**
  - hand (1): Fire King Sanctuary
  - monsters: Snake-Eyes Flamberge Dragon (ATK), Legendary Fire King Ponix (ATK), Fire King High Avatar Kirin (ATK), Diabellstar the Black Witch (ATK)
  - spells/traps: Fire King Island
  - GY (6): Bonfire, Snake-Eye Ash, Divine Temple of the Snake-Eye, Snake-Eyes Poplar, Snake-Eye Oak, Original Sinful Spoils - Snake-Eye
  - deck 29, extra deck 15

Plays:

- Wiccanthrope activates Fire Formation - Tenki
- Wiccanthrope answers YES ("Add 1 Level 4 or lower Beast-Warrior monster from your Deck to your hand?")
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to add to your hand]_
- Wiccanthrope Normal Summons Wiccanthrope Walioright
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Walioright ("Shuffle up to 3 cards into the Decks")
- Wiccanthrope selects Wiccanthrope Reason  _[Select the card(s) to banish]_  **(search: chose this over the network's first choice: selects Wiccanthrope Enchantment)**
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to return to the Deck]_  **(search: chose this over the network's first choice: selects Wiccanthrope Enchantment)**
- Wiccanthrope selects Mulcharmy Fuwalos  _[Select the card(s) to return to the Deck]_  **(search: chose this over the network's first choice: selects Wiccanthrope Enchantment)**
- Wiccanthrope selects Ash Blossom & Joyous Spring  _[Select the card(s) to return to the Deck]_  **(search: chose this over the network's first choice: selects Wiccanthrope Enchantment)**
- Wiccanthrope uses the effect of Wiccanthrope Reason
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Wiccanthrope Enchantment ("Target 1 card on the field"))**
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects Fire King Island  _[Select the target(s) of the effect]_  **(search: chose this over the network's first choice: selects Snake-Eyes Flamberge Dragon)**
- Snake-Eye activates Snake-Eyes Flamberge Dragon ("Special Summon 2 Level 1 FIRE monsters from your GY")
- Snake-Eye activates Diabellstar the Black Witch ("Special Summon this card from the GY")
- Snake-Eye activates Fire King High Avatar Kirin ("Special Summon 1 "Fire King" from your hand or GY")
- Snake-Eye chooses position face-up Defense
- Snake-Eye answers YES ("Destroy 1 card on the field?")
- Snake-Eye selects Wiccanthrope Walioright  _[Select the card(s) to destroy]_
- Snake-Eye selects Legendary Fire King Ponix  _[Select the card(s) to send to the GY]_
- Snake-Eye chooses position face-up Defense
- Snake-Eye selects Snake-Eye Oak  _[Select the card(s) to Special Summon]_
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to Special Summon]_
- Snake-Eye chooses position face-up Defense
- Snake-Eye chooses position face-up Defense
- Snake-Eye activates Snake-Eyes Poplar ("Search 1 "Snake-Eye" Spell/Trap")  **(search: chose this over the network's first choice: activates Diabellstar the Black Witch ("Set 1 "Sinful Spoils" Spell/Trap from your Deck"))**
- Snake-Eye activates Diabellstar the Black Witch ("Set 1 "Sinful Spoils" Spell/Trap from your Deck")
- Snake-Eye activates Snake-Eye Oak ("Add to hand or Special Summon 1 Level 1 FIRE from your GY/banished")
- Snake-Eye selects Snake-Eye Ash  _[Select the target(s) of the effect]_
- Snake-Eye chooses ("Special Summon it")
- Snake-Eye chooses position face-up Defense
- Snake-Eye selects WANTED: Seeker of Sinful Spoils  _[Select the card(s) to Set to the field]_
- Snake-Eye selects Original Sinful Spoils - Snake-Eye  _[Select the card(s) to add to your hand]_
- Snake-Eye uses the effect of Snake-Eye Ash
- Snake-Eye selects Snake-Eye Ash  _[Select the card(s) to add to your hand]_
- Wiccanthrope ends the turn

## Turn 4: Snake-Eye 's turn  (LP Wiccanthrope 700 / Snake-Eye 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (1): Wiccanthrope Walioright
  - monsters: -
  - spells/traps: Fire Formation - Tenki
  - GY (4): Wiccanthrope Road, Wiccanthrope Enchantment, Wiccanthrope Enchantment, Wiccanthrope Walioright
  - banished (1): Wiccanthrope Reason
  - deck 33, extra deck 14
- **Snake-Eye**
  - hand (4): Original Sinful Spoils - Snake-Eye, Snake-Eye Ash, Fire King Sanctuary, Triple Tactics Thrust
  - monsters: Snake-Eyes Poplar (DEF), Snake-Eye Oak (DEF), Snake-Eye Ash (DEF), Diabellstar the Black Witch (DEF)
  - spells/traps: WANTED: Seeker of Sinful Spoils (set)
  - GY (7): Bonfire, Divine Temple of the Snake-Eye, Original Sinful Spoils - Snake-Eye, Fire King Island, Snake-Eyes Flamberge Dragon, Fire King High Avatar Kirin, Legendary Fire King Ponix
  - deck 25, extra deck 15

Plays:

- Snake-Eye passes  **(search: chose this over the network's first choice: activates WANTED: Seeker of Sinful Spoils ("Add 1 "Diabellstar" monster from your Deck or GY to your hand"))**
- Snake-Eye passes  **(search: chose this over the network's first choice: activates WANTED: Seeker of Sinful Spoils ("Add 1 "Diabellstar" monster from your Deck or GY to your hand"))**
- Snake-Eye activates WANTED: Seeker of Sinful Spoils ("Add 1 "Diabellstar" monster from your Deck or GY to your hand")
- Snake-Eye selects Diabellstar the Black Witch  _[Select the card(s) to add to your hand]_
- Snake-Eye Special Summons Diabellstar the Black Witch  **(search: chose this over the network's first choice: changes the position of Diabellstar the Black Witch)**
- Snake-Eye selects Original Sinful Spoils - Snake-Eye  _[Select the card(s) to send to the GY]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye declines the effect of Diabellstar the Black Witch  **(search: chose this over the network's first choice: uses the effect of Diabellstar the Black Witch)**
- Snake-Eye changes the position of Diabellstar the Black Witch
- Snake-Eye enters the Battle Phase
- Snake-Eye attacks with Diabellstar the Black Witch

## Game over: Snake-Eye  wins (LP before the final blow: Wiccanthrope 700 / Snake-Eye 8000)

</details>
<details><summary><b>Game 3</b>: Wiccanthrope (second) vs Snake-Eye: Wiccanthrope wins on turn 2</summary>


- Goes first: Snake-Eye 
- Result: **Wiccanthrope  wins** on turn 2
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261017; 86 decisions in total, 33 searched, 18 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Snake-Eye 's turn  (LP Snake-Eye 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Snake-Eye**
  - hand (5): Diabellstar the Black Witch, Ghost Ogre & Snow Rabbit, Harpie's Feather Duster, Snake-Eye Ash, Triple Tactics Thrust
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Ash Blossom & Joyous Spring, Effect Veiler, Mulcharmy Purulia
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14

Plays:

- Wiccanthrope activates Mulcharmy Purulia ("Apply the "●" effects for the rest of the turn")
- Snake-Eye activates Triple Tactics Thrust ("Set 1 Normal Spell/Trap from your Deck")
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Ash Blossom & Joyous Spring ("Negate that effect"))**
- Snake-Eye selects Infinite Impermanence  _[Select the card(s) to Set to the field]_
- Snake-Eye Normal Summons Snake-Eye Ash
- Snake-Eye uses the effect of Snake-Eye Ash
- Wiccanthrope activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster")
- Snake-Eye Special Summons Diabellstar the Black Witch
- Snake-Eye selects Snake-Eye Ash  _[Select the card(s) to send to the GY]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye uses the effect of Diabellstar the Black Witch
- Snake-Eye selects Original Sinful Spoils - Snake-Eye  _[Select the card(s) to Set to the field]_
- Snake-Eye activates Original Sinful Spoils - Snake-Eye ("Special Summon 1 Level 1 FIRE monster from your hand/Deck")
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")
- Snake-Eye sets Harpie's Feather Duster

## Turn 2: Wiccanthrope 's turn  (LP Snake-Eye 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Snake-Eye**
  - hand (1): Ghost Ogre & Snow Rabbit
  - monsters: -
  - spells/traps: Harpie's Feather Duster (set), Infinite Impermanence (set)
  - GY (4): Triple Tactics Thrust, Snake-Eye Ash, Diabellstar the Black Witch, Original Sinful Spoils - Snake-Eye
  - deck 34, extra deck 15
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment, Wiccanthrope Serwapentar, Ghost Mourner & Moonlit Chill
  - monsters: -
  - spells/traps: -
  - GY (3): Mulcharmy Purulia, Effect Veiler, Ash Blossom & Joyous Spring
  - deck 32, extra deck 14

Plays:

- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Snake-Eye activates Infinite Impermanence
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects opponent's face-down card (Spell/Trap Zone)  _[Select the target(s) of the effect]_
- Wiccanthrope Normal Summons Wiccanthrope Serwapentar  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Walioright)**
- Wiccanthrope Special Summons Wiccanthrope Walioright
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Snake-Eye activates Ghost Ogre & Snow Rabbit
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to add to your hand]_
- Wiccanthrope Special Summons Wiccanthrope Spedallacer
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Rothockear)**
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Spedallacer
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to add to your hand]_  **(search: chose this over the network's first choice: selects Wiccanthrope Enchantment)**
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Wiccanthrope Spedallacer ("Negate the effects of 1 face-up card"))**
- Wiccanthrope activates Wiccanthrope Spedallacer ("Negate the effects of 1 face-up card")
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the Xyz Material(s) to detach]_
- Wiccanthrope answers YES ("Banish 1 Spell and inflict 500 damage?")
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Wiccanthrope Special Summons Wiccanthrope Sperelfler  **(search: chose this over the network's first choice: activates Fire Formation - Tenki)**
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope activates Fire Formation - Tenki
- Wiccanthrope answers YES ("Add 1 Level 4 or lower Beast-Warrior monster from your Deck to your hand?")
- Wiccanthrope activates Wiccanthrope Sperelfler ("Special Summon 1 "Wiccanthrope" monster from your GY")  **(search: chose this over the network's first choice: activates Wiccanthrope Serwapentar ("Special Summon this card from your GY"))**
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the Xyz Material(s) to detach]_  **(search: chose this over the network's first choice: selects Wiccanthrope Rothockear)**
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the card(s) to Special Summon]_  **(search: chose this over the network's first choice: selects Wiccanthrope Serwapentar)**
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope answers YES ("Banish 1 Spell and attach 1 Spell from your Deck to an Xyz Monster?")
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select a face-up card(s)]_
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope Special Summons Wiccanthrope Sperelfler  **(search: chose this over the network's first choice: activates Wiccanthrope Serwapentar ("Special Summon this card from your GY"))**
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope activates Wiccanthrope Serwapentar ("Special Summon this card from your GY")
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to discard]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Serwapentar
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the target(s) of the effect]_
- Wiccanthrope answers YES ("Levels and Ranks of all monsters become 4")  **(search: chose this over the network's first choice: answers NO ("Levels and Ranks of all monsters become 4"))**
- Wiccanthrope sets Wiccanthrope Enchantment
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope attacks with Wiccanthrope Serwapentar

## Game over: Wiccanthrope  wins (LP before the final blow: Snake-Eye 1300 / Wiccanthrope 8000)

</details>
<details><summary><b>Game 4</b>: Wiccanthrope (second) vs Snake-Eye: Wiccanthrope wins on turn 2</summary>


- Goes first: Snake-Eye 
- Result: **Wiccanthrope  wins** on turn 2
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261018; 102 decisions in total, 36 searched, 19 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Snake-Eye 's turn  (LP Snake-Eye 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Snake-Eye**
  - hand (5): Harpie's Feather Duster, Infinite Impermanence, Snake-Eye Ash, Ash Blossom & Joyous Spring, Snake-Eye Ash
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment, Mulcharmy Fuwalos, Wiccanthrope Serwapentar
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14

Plays:

- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn"))**
- Snake-Eye Normal Summons Snake-Eye Ash
- Snake-Eye uses the effect of Snake-Eye Ash
- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to add to your hand]_
- Snake-Eye uses the effect of Snake-Eyes Poplar
- Snake-Eye chooses position face-up Attack
- Snake-Eye uses the effect of Snake-Eyes Poplar
- Snake-Eye selects Divine Temple of the Snake-Eye  _[Select the card(s) to add to your hand]_
- Snake-Eye activates Divine Temple of the Snake-Eye ("Place 1 "Snake-Eye" monster from your hand, Deck, or GY face-up in its owner's Spell & Trap Zone as a Continuous Spell?")
- Snake-Eye answers YES ("Place 1 "Snake-Eye" monster from your hand, Deck, or GY face-up in its owner's Spell & Trap Zone as a Continuous Spell?")
- Snake-Eye selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to place on the field]_
- Snake-Eye activates Snake-Eye Ash ("Special Summon 1 "Snake-Eye" from your hand or Deck")
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to send to the GY]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye activates Snake-Eye Oak ("Add to hand or Special Summon 1 Level 1 FIRE from your GY/banished")
- Snake-Eye selects Snake-Eyes Poplar  _[Select the target(s) of the effect]_
- Snake-Eye activates Snake-Eyes Poplar ("Place 1 FIRE monster in the Spell & Trap Zone as a Continuous Spell")
- Snake-Eye selects Snake-Eye Ash  _[Select the target(s) of the effect]_
- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Snake-Eye chooses ("Special Summon it")
- Snake-Eye chooses position face-up Attack
- Snake-Eye sets Infinite Impermanence
- Snake-Eye activates Snake-Eye Oak ("Special Summon 1 "Snake-Eye" from your hand or Deck")
- Snake-Eye selects Snake-Eyes Poplar  _[Select the card(s) to send to the GY]_
- Snake-Eye selects Snake-Eye Ash  _[Select the card(s) to Special Summon]_
- Snake-Eye chooses position face-up Attack
- Snake-Eye ends the turn
- Wiccanthrope activates Mulcharmy Fuwalos

## Turn 2: Wiccanthrope 's turn  (LP Snake-Eye 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Snake-Eye**
  - hand (3): Harpie's Feather Duster, Snake-Eye Ash, Ash Blossom & Joyous Spring
  - monsters: Snake-Eye Ash (ATK)
  - spells/traps: Snake-Eye Ash, Infinite Impermanence (set), Snake-Eyes Flamberge Dragon, Divine Temple of the Snake-Eye
  - GY (2): Snake-Eye Oak, Snake-Eyes Poplar
  - deck 31, extra deck 15
- **Wiccanthrope**
  - hand (7): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment, Wiccanthrope Serwapentar, Wiccanthrope Befaist, Mulcharmy Fuwalos, Ghost Mourner & Moonlit Chill
  - monsters: -
  - spells/traps: -
  - GY (2): Mulcharmy Fuwalos, Mulcharmy Fuwalos
  - deck 31, extra deck 14

Plays:

- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn"))**
- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")  **(search: chose this over the network's first choice: passes)**
- Wiccanthrope Normal Summons Wiccanthrope Rothockear  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Rothockear)**
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Snake-Eye passes  **(search: chose this over the network's first choice: activates Divine Temple of the Snake-Eye ("Special Summon 1 Monster Card treated as a Continuous Spell"))**
- Snake-Eye activates Infinite Impermanence
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects Divine Temple of the Snake-Eye  _[Select the target(s) of the effect]_
- Wiccanthrope Special Summons Wiccanthrope Walioright
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Snake-Eye activates Ash Blossom & Joyous Spring ("Negate that effect")
- Wiccanthrope Special Summons Wiccanthrope Wataquera  **(search: chose this over the network's first choice: activates Wiccanthrope Walioright ("Shuffle up to 3 cards into the Decks"))**
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Wataquera
- Wiccanthrope selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope activates Wiccanthrope Wataquera ("Destroy 1 card on the field")
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the Xyz Material(s) to detach]_  **(search: chose this over the network's first choice: selects Snake-Eyes Flamberge Dragon)**
- Wiccanthrope selects Snake-Eye Ash  _[Select the card(s) to destroy]_
- Wiccanthrope answers YES ("Banish 1 Spell and Set 1 "Wiccanthrope" Spell from your Deck?")
- Wiccanthrope selects Wiccanthrope Reason  _[Select the card(s) to Set to the field]_  **(search: chose this over the network's first choice: selects Wiccanthrope Enchantment)**
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Wiccanthrope Special Summons Wiccanthrope Sperelfler
- Wiccanthrope selects Wiccanthrope Wataquera  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope activates Wiccanthrope Sperelfler ("Special Summon 1 "Wiccanthrope" monster from your GY")
- Wiccanthrope selects Wiccanthrope Wataquera  _[Select the Xyz Material(s) to detach]_  **(search: chose this over the network's first choice: selects Wiccanthrope Walioright)**
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to Special Summon]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope answers YES ("Banish 1 Spell and attach 1 Spell from your Deck to an Xyz Monster?")
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope uses the effect of Wiccanthrope Reason
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Rothockear ("Special Summon 1 "Wiccanthrope" monster from your GY")
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope Special Summons Wiccanthrope Sperelfler
- Wiccanthrope selects Wiccanthrope Wataquera  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope selects Snake-Eye Ash  _[Select an attack target]_
- Wiccanthrope attacks with Wiccanthrope Rothockear
- Wiccanthrope attacks with Wiccanthrope Sperelfler

## Game over: Wiccanthrope  wins (LP before the final blow: Snake-Eye 1600 / Wiccanthrope 8000)

</details>
<details><summary><b>Game 5</b>: Wiccanthrope (first) vs Mitsurugi: Mitsurugi wins on turn 4</summary>


- Goes first: Wiccanthrope 
- Result: **Mitsurugi  wins** on turn 4
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261017; 114 decisions in total, 39 searched, 24 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Wiccanthrope 's turn  (LP Wiccanthrope 8000 / Mitsurugi 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (5): Effect Veiler, Wiccanthrope Befaist, Mulcharmy Fuwalos, Mulcharmy Purulia, Ash Blossom & Joyous Spring
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14
- **Mitsurugi**
  - hand (5): Ame no Habakiri no Mitsurugi, Mulcharmy Fuwalos, Ame no Murakumo no Mitsurugi, Mitsurugi Prayers, Mitsurugi no Miko, Wousu
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15

Plays:

- Wiccanthrope activates Mulcharmy Purulia ("Apply the "●" effects for the rest of the turn")
- Mitsurugi passes  **(search: chose this over the network's first choice: activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn"))**
- Wiccanthrope Normal Summons Wiccanthrope Befaist
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Mitsurugi activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Wiccanthrope selects Wiccanthrope Road  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Road ("Add 1 "Wiccanthrope" monster from your Deck to your hand")
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")
- Wiccanthrope uses the effect of Wiccanthrope Road
- Wiccanthrope ends the turn
- Wiccanthrope activates Mulcharmy Purulia

## Turn 2: Mitsurugi 's turn  (LP Wiccanthrope 8000 / Mitsurugi 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (5): Wiccanthrope Befaist, Ash Blossom & Joyous Spring, Wiccanthrope Befaist, Effect Veiler, Mulcharmy Fuwalos
  - monsters: -
  - spells/traps: -
  - GY (1): Mulcharmy Purulia
  - banished (1): Wiccanthrope Road
  - deck 33, extra deck 14
- **Mitsurugi**
  - hand (5): Ame no Habakiri no Mitsurugi, Ame no Murakumo no Mitsurugi, Mitsurugi Prayers, Mitsurugi no Miko, Wousu, Mitsurugi Prayers
  - monsters: -
  - spells/traps: -
  - GY (1): Mulcharmy Fuwalos
  - deck 35, extra deck 15

Plays:

- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn"))**
- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")  **(search: chose this over the network's first choice: passes)**
- Mitsurugi activates Mitsurugi Prayers ("Apply 1 of these effects")  **(search: chose this over the network's first choice: passes)**
- Mitsurugi answers YES ("Activate this card by Tributing 1 Reptile monster in your hand or field?")
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to Tribute]_  **(search: chose this over the network's first choice: selects Ame no Habakiri no Mitsurugi)**
- Mitsurugi chooses ("Add 1 "Mitsurugi" monster from your Deck to your hand")  **(search: chose this over the network's first choice: chooses ("Apply both effects, in sequence"))**
- Mitsurugi selects Mitsurugi no Mikoto, Saji  _[Select the card(s) to add to your hand]_
- Mitsurugi uses the effect of Ame no Murakumo no Mitsurugi
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")
- Mitsurugi sets Mitsurugi Prayers  **(search: chose this over the network's first choice: Normal Summons Mitsurugi no Mikoto, Saji)**
- Mitsurugi activates Ame no Habakiri no Mitsurugi ("Special Summon 1 "Mitsurugi" monster from your Deck, then Tribute 1 monster you control")  **(search: chose this over the network's first choice: Normal Summons Mitsurugi no Mikoto, Saji)**
- Mitsurugi selects Mitsurugi no Mikoto, Aramasa  _[Select the card(s) to Special Summon]_
- Mitsurugi chooses position face-up Attack
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Aramasa
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to add to your hand]_  **(search: chose this over the network's first choice: selects Mitsurugi no Mikoto, Kusanagi)**
- Mitsurugi Normal Summons Mitsurugi no Mikoto, Saji
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Saji
- Wiccanthrope activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster")
- Mitsurugi enters the Battle Phase
- Mitsurugi attacks with Mitsurugi no Mikoto, Saji
- Mitsurugi goes to Main Phase 2

## Turn 3: Wiccanthrope 's turn  (LP Wiccanthrope 7000 / Mitsurugi 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (4): Wiccanthrope Befaist, Wiccanthrope Befaist, Infinite Impermanence, Wiccanthrope Serwapentar
  - monsters: -
  - spells/traps: -
  - GY (4): Mulcharmy Purulia, Mulcharmy Fuwalos, Ash Blossom & Joyous Spring, Effect Veiler
  - banished (1): Wiccanthrope Road
  - deck 31, extra deck 14
- **Mitsurugi**
  - hand (3): Mitsurugi no Miko, Wousu, Ame no Murakumo no Mitsurugi, Ame no Habakiri no Mitsurugi
  - monsters: Mitsurugi no Mikoto, Saji (ATK)
  - spells/traps: Mitsurugi Prayers (set)
  - GY (4): Mulcharmy Fuwalos, Ame no Murakumo no Mitsurugi, Mitsurugi Prayers, Mitsurugi no Mikoto, Aramasa
  - deck 32, extra deck 15

Plays:

- Wiccanthrope Normal Summons Wiccanthrope Befaist
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects opponent's face-down card (Spell/Trap Zone)  _[Select the target(s) of the effect]_
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")
- Wiccanthrope selects Mitsurugi no Mikoto, Saji  _[Select the card(s) to destroy]_
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Wiccanthrope sets Infinite Impermanence
- Wiccanthrope sets Infinite Impermanence
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Befaist
- Wiccanthrope goes to Main Phase 2  **(search: chose this over the network's first choice: ends the turn)**

## Turn 4: Mitsurugi 's turn  (LP Wiccanthrope 7000 / Mitsurugi 6500)

At the start of the turn:

- **Wiccanthrope**
  - hand (2): Wiccanthrope Serwapentar, Wiccanthrope Befaist
  - monsters: Wiccanthrope Befaist (ATK)
  - spells/traps: Infinite Impermanence (set), Infinite Impermanence (set)
  - GY (4): Mulcharmy Purulia, Mulcharmy Fuwalos, Ash Blossom & Joyous Spring, Effect Veiler
  - banished (2): Wiccanthrope Road, Wiccanthrope Enchantment
  - deck 29, extra deck 14
- **Mitsurugi**
  - hand (4): Mitsurugi no Miko, Wousu, Ame no Murakumo no Mitsurugi, Ame no Habakiri no Mitsurugi, Mitsurugi Ritual
  - monsters: -
  - spells/traps: -
  - GY (6): Mulcharmy Fuwalos, Ame no Murakumo no Mitsurugi, Mitsurugi Prayers, Mitsurugi no Mikoto, Aramasa, Mitsurugi Prayers, Mitsurugi no Mikoto, Saji
  - deck 31, extra deck 15

Plays:

- Mitsurugi activates Mitsurugi Ritual ("Activate 1 of these effects")
- Mitsurugi chooses ("Ritual Summon 1 Reptile Ritual Monster from your Deck")
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to Special Summon]_
- Mitsurugi selects Ame no Habakiri no Mitsurugi  _[Select the card(s) to Tribute]_
- Mitsurugi chooses position face-up Attack
- Mitsurugi activates Ame no Murakumo no Mitsurugi ("Destroy all monsters your opponent controls")
- Mitsurugi activates Ame no Habakiri no Mitsurugi ("Add 1 "Mitsurugi" card from your Deck to your hand, except "Ame no Habakiri no Mitsurugi"")
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Infinite Impermanence)**
- Mitsurugi selects Mitsurugi Ritual  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Infinite Impermanence  **(search: chose this over the network's first choice: passes)**
- Mitsurugi activates Ame no Murakumo no Mitsurugi ("Your opponent can discard 1 card, or that effect is negated")
- Wiccanthrope chooses ("That effect is negated")
- Wiccanthrope activates Infinite Impermanence  **(search: chose this over the network's first choice: passes)**
- Mitsurugi activates Mitsurugi Ritual ("Activate 1 of these effects")
- Mitsurugi selects Futsu no Mitama no Mitsurugi  _[Select the card(s) to Tribute]_  **(search: chose this over the network's first choice: selects Mitsurugi no Mikoto, Aramasa)**
- Mitsurugi chooses position face-up Attack
- Mitsurugi uses the effect of Futsu no Mitama no Mitsurugi
- Mitsurugi selects Mitsurugi no Mikoto, Kusanagi  _[Select the card(s) to add to your hand]_
- Mitsurugi Normal Summons Mitsurugi no Mikoto, Kusanagi
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Kusanagi
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to add to your hand]_
- Mitsurugi enters the Battle Phase
- Mitsurugi attacks with Ame no Murakumo no Mitsurugi
- Mitsurugi attacks with Ame no Murakumo no Mitsurugi
- Mitsurugi attacks with Mitsurugi no Mikoto, Kusanagi

## Game over: Mitsurugi  wins (LP before the final blow: Wiccanthrope 600 / Mitsurugi 6500)

</details>
<details><summary><b>Game 6</b>: Wiccanthrope (first) vs Mitsurugi: Mitsurugi wins on turn 2</summary>


- Goes first: Wiccanthrope 
- Result: **Mitsurugi  wins** on turn 2
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261018; 62 decisions in total, 32 searched, 17 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Wiccanthrope 's turn  (LP Wiccanthrope 8000 / Mitsurugi 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (5): Wiccanthrope Reason, Ash Blossom & Joyous Spring, Wiccanthrope Enchantment, Mulcharmy Fuwalos, Wiccanthrope Rothockear
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14
- **Mitsurugi**
  - hand (5): Ame no Murakumo no Mitsurugi, Ragged Records of Rites, Mitsurugi no Mikoto, Saji, Pre-Preparation of Rites, Mitsurugi Prayers
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15

Plays:

- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Wiccanthrope selects Wiccanthrope Road  _[Select the card(s) to send to the GY]_
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope sets Wiccanthrope Reason
- Wiccanthrope ends the turn

## Turn 2: Mitsurugi 's turn  (LP Wiccanthrope 8000 / Mitsurugi 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (1): Ash Blossom & Joyous Spring
  - monsters: Wiccanthrope Rothockear (ATK)
  - spells/traps: Wiccanthrope Reason (set)
  - GY (3): Mulcharmy Fuwalos, Wiccanthrope Road, Wiccanthrope Enchantment
  - deck 34, extra deck 14
- **Mitsurugi**
  - hand (6): Ame no Murakumo no Mitsurugi, Ragged Records of Rites, Mitsurugi no Mikoto, Saji, Pre-Preparation of Rites, Mitsurugi Prayers, Mitsurugi no Mikoto, Aramasa
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 15

Plays:

- Mitsurugi activates Mitsurugi Prayers ("Apply 1 of these effects")  **(search: chose this over the network's first choice: passes)**
- Mitsurugi answers YES ("Activate this card by Tributing 1 Reptile monster in your hand or field?")
- Mitsurugi selects Mitsurugi no Mikoto, Saji  _[Select the card(s) to Tribute]_
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Saji
- Mitsurugi selects Mitsurugi Ritual  _[Select the card(s) to add to your hand]_
- Mitsurugi activates Pre-Preparation of Rites ("Add 1 Ritual Spell from your Deck to your hand, and add 1 Ritual Monster from your Deck or GY to your hand whose name is mentioned on that Ritual Spell")
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to add to your hand]_  **(search: chose this over the network's first choice: selects Ame no Habakiri no Mitsurugi)**
- Mitsurugi activates Mitsurugi Ritual ("Activate 1 of these effects")  **(search: chose this over the network's first choice: activates Ragged Records of Rites ("Reveal 1 Ritual Spell in your hand or Deck, and add 1 monster that mentions it from your Deck to your hand"))**
- Mitsurugi chooses ("Ritual Summon 1 Reptile Ritual Monster from your Deck")
- Mitsurugi selects Ame no Habakiri no Mitsurugi  _[Select the card(s) to Special Summon]_
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to Tribute]_
- Mitsurugi chooses position face-up Attack
- Mitsurugi uses the effect of Ame no Murakumo no Mitsurugi
- Mitsurugi selects Mitsurugi no Miko, Wousu  _[Select the card(s) to add to your hand]_
- Mitsurugi Normal Summons Mitsurugi no Mikoto, Aramasa  **(search: chose this over the network's first choice: activates Ragged Records of Rites ("Reveal 1 Ritual Spell in your hand or Deck, and add 1 monster that mentions it from your Deck to your hand"))**
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Aramasa
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to add to your hand]_
- Mitsurugi Special Summons Mitsurugi no Miko, Wousu  **(search: chose this over the network's first choice: activates Ragged Records of Rites ("Reveal 1 Ritual Spell in your hand or Deck, and add 1 monster that mentions it from your Deck to your hand"))**
- Mitsurugi selects Wiccanthrope Rothockear  _[Select the card(s) to Tribute]_  **(search: chose this over the network's first choice: selects Ame no Murakumo no Mitsurugi)**
- Mitsurugi unselects Wiccanthrope Rothockear  _[Select the card(s) to Tribute]_  **(search: chose this over the network's first choice: selects Ame no Murakumo no Mitsurugi)**
- Mitsurugi cancels  _[Select the card(s) to Tribute]_  **(search: chose this over the network's first choice: selects Ame no Murakumo no Mitsurugi)**
- Mitsurugi activates Ragged Records of Rites ("Reveal 1 Ritual Spell in your hand or Deck, and add 1 monster that mentions it from your Deck to your hand")
- Mitsurugi selects Futsu no Mitama no Mitsurugi  _[Select the card to reveal and the card to add to your hand]_
- Mitsurugi Special Summons Mitsurugi no Miko, Wousu
- Mitsurugi selects Wiccanthrope Rothockear  _[Select the card(s) to Tribute]_  **(search: chose this over the network's first choice: selects Ame no Murakumo no Mitsurugi)**
- Mitsurugi selects Futsu no Mitama no Mitsurugi  _[Select the card(s) to Tribute]_  **(search: chose this over the network's first choice: selects Ame no Murakumo no Mitsurugi)**
- Mitsurugi chooses position face-up Attack
- Mitsurugi uses the effect of Futsu no Mitama no Mitsurugi
- Mitsurugi selects Mitsurugi Great Purification  _[Select the card(s) to add to your hand]_  **(search: chose this over the network's first choice: selects Mitsurugi Ritual)**
- Mitsurugi activates Mitsurugi Mirror ("Ritual Summon")
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to Special Summon]_
- Mitsurugi selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to Tribute]_
- Mitsurugi chooses position face-up Attack
- Mitsurugi enters the Battle Phase
- Mitsurugi attacks with Ame no Murakumo no Mitsurugi  **(search: chose this over the network's first choice: attacks with Mitsurugi no Miko, Wousu)**
- Mitsurugi attacks with Mitsurugi no Miko, Wousu
- Mitsurugi attacks with Ame no Habakiri no Mitsurugi
- Mitsurugi attacks with Mitsurugi no Mikoto, Aramasa

## Game over: Mitsurugi  wins (LP before the final blow: Wiccanthrope 400 / Mitsurugi 8000)

</details>
<details><summary><b>Game 7</b>: Wiccanthrope (second) vs Mitsurugi: Wiccanthrope wins on turn 2</summary>


- Goes first: Mitsurugi 
- Result: **Wiccanthrope  wins** on turn 2
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261017; 110 decisions in total, 46 searched, 22 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Mitsurugi 's turn  (LP Mitsurugi 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Mitsurugi**
  - hand (5): Mitsurugi no Mikoto, Aramasa, Ash Blossom & Joyous Spring, Pot of Prosperity, End of the World Ruler, Called by the Grave
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Ash Blossom & Joyous Spring, Effect Veiler, Mulcharmy Purulia
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14

Plays:

- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Mulcharmy Purulia ("Apply the "●" effects for the rest of the turn"))**
- Mitsurugi Normal Summons Mitsurugi no Mikoto, Aramasa
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Aramasa
- Wiccanthrope activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster")
- Mitsurugi activates Called by the Grave
- Wiccanthrope activates Mulcharmy Purulia ("Apply the "●" effects for the rest of the turn")  **(search: chose this over the network's first choice: passes)**
- Mitsurugi selects Mitsurugi no Miko, Wousu  _[Select the card(s) to add to your hand]_
- Mitsurugi activates Pot of Prosperity
- Mitsurugi declares the number 6
- Mitsurugi selects S:P Little Knight  _[Select the card(s) to banish]_
- Mitsurugi selects I:P Masquerena  _[Select the card(s) to banish]_
- Mitsurugi selects W:P Fancy Ball  _[Select the card(s) to banish]_
- Mitsurugi selects Albion the Branded Dragon  _[Select the card(s) to banish]_
- Mitsurugi selects Dharc the Dark Charmer, Gloomy  _[Select the card(s) to banish]_
- Mitsurugi selects Ecclesia and the Dark Dragon  _[Select the card(s) to banish]_
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")
- Mitsurugi activates End of the World Ruler ("Apply that card's activation effect that Ritual Summons a monster")
- Mitsurugi selects Mitsurugi Ritual  _[Select the card(s) to banish]_
- Mitsurugi selects Ame no Habakiri no Mitsurugi  _[Select the card(s) to Special Summon]_
- Mitsurugi chooses position face-up Attack
- Mitsurugi uses the effect of Mitsurugi no Miko, Wousu
- Mitsurugi selects End of the World Ruler  _[Select the card(s) to discard]_
- Mitsurugi ends the turn

## Turn 2: Wiccanthrope 's turn  (LP Mitsurugi 6000 / Wiccanthrope 8000)

At the start of the turn:

- **Mitsurugi**
  - hand (2): Ash Blossom & Joyous Spring, Mitsurugi no Miko, Wousu
  - monsters: Mitsurugi no Mikoto, Aramasa (ATK), Ame no Habakiri no Mitsurugi (ATK)
  - spells/traps: -
  - GY (3): Called by the Grave, Pot of Prosperity, End of the World Ruler
  - banished (7): S:P Little Knight, I:P Masquerena, W:P Fancy Ball, Dharc the Dark Charmer, Gloomy, Albion the Branded Dragon, Ecclesia and the Dark Dragon, Mitsurugi Ritual
  - deck 33, extra deck 9
- **Wiccanthrope**
  - hand (3): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment
  - monsters: -
  - spells/traps: -
  - GY (2): Mulcharmy Purulia, Ash Blossom & Joyous Spring
  - banished (1): Effect Veiler
  - deck 34, extra deck 14

Plays:

- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Defense  **(search: chose this over the network's first choice: chooses position face-up Attack)**
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Mitsurugi activates Ash Blossom & Joyous Spring ("Negate that effect")
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the target(s) of the effect]_  **(search: chose this over the network's first choice: selects Ame no Habakiri no Mitsurugi)**
- Wiccanthrope Special Summons Wiccanthrope Walioright
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Walioright ("Shuffle up to 3 cards into the Decks")  **(search: chose this over the network's first choice: Normal Summons Wiccanthrope Befaist)**
- Wiccanthrope selects Mitsurugi Ritual  _[Select the card(s) to return to the Deck]_
- Wiccanthrope selects Ash Blossom & Joyous Spring  _[Select the card(s) to return to the Deck]_
- Wiccanthrope selects Effect Veiler  _[Select the card(s) to return to the Deck]_
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Wiccanthrope activates Fire Formation - Tenki
- Wiccanthrope answers YES ("Add 1 Level 4 or lower Beast-Warrior monster from your Deck to your hand?")
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to add to your hand]_
- Wiccanthrope Normal Summons Wiccanthrope Befaist
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to banish]_
- Wiccanthrope selects Ame no Habakiri no Mitsurugi  _[Select the card(s) to destroy]_
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Aramasa
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Aramasa
- Mitsurugi selects Mitsurugi no Mikoto, Saji  _[Select the card(s) to add to your hand]_
- Wiccanthrope Special Summons Wiccanthrope Vilecoil  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Spedallacer)**
- Wiccanthrope cancels  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Walioright)**
- Wiccanthrope Special Summons Wiccanthrope Wataquera  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Spedallacer)**
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Walioright)**
- Wiccanthrope unselects Wiccanthrope Befaist  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Rothockear)**
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Walioright)**
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Wataquera
- Wiccanthrope activates Wiccanthrope Wataquera ("Destroy 1 card on the field")
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the Xyz Material(s) to detach]_  **(search: chose this over the network's first choice: selects Wiccanthrope Rothockear)**
- Wiccanthrope selects Ame no Habakiri no Mitsurugi  _[Select the card(s) to destroy]_
- Wiccanthrope answers YES ("Banish 1 Spell and Set 1 "Wiccanthrope" Spell from your Deck?")
- Wiccanthrope selects Wiccanthrope Reason  _[Select the card(s) to Set to the field]_  **(search: chose this over the network's first choice: selects Wiccanthrope Enchantment)**
- Wiccanthrope Special Summons Wiccanthrope Sperelfler  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Befaist)**
- Wiccanthrope selects Wiccanthrope Wataquera  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope Special Summons Wiccanthrope Befaist
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope Special Summons Wiccanthrope Vilecoil
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Vilecoil
- Wiccanthrope activates Wiccanthrope Vilecoil ("Set 1 of your banished "Wiccanthrope" Spells")
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the Xyz Material(s) to detach]_
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to Set to the field]_
- Wiccanthrope answers NO ("Banish 1 Spell and return 1 banished "Wiccanthrope" Spell to the GY?")
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Vilecoil
- Wiccanthrope attacks with Wiccanthrope Sperelfler

## Game over: Wiccanthrope  wins (LP before the final blow: Mitsurugi 2600 / Wiccanthrope 8000)

</details>
<details><summary><b>Game 8</b>: Wiccanthrope (second) vs Mitsurugi: Wiccanthrope wins on turn 4</summary>


- Goes first: Mitsurugi 
- Result: **Wiccanthrope  wins** on turn 4
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261018; 108 decisions in total, 1 searched, 0 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Mitsurugi 's turn  (LP Mitsurugi 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Mitsurugi**
  - hand (5): Pot of Prosperity, Dominus Impulse, End of the World Ruler, Mulcharmy Purulia, End of the World Ruler
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment, Mulcharmy Fuwalos, Wiccanthrope Serwapentar
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14

Plays:

- Mitsurugi activates Mulcharmy Purulia ("Apply the "●" effects for the rest of the turn")
- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Mitsurugi activates Pot of Prosperity
- Mitsurugi declares the number 6
- Mitsurugi selects I:P Masquerena  _[Select the card(s) to banish]_
- Mitsurugi selects S:P Little Knight  _[Select the card(s) to banish]_
- Mitsurugi selects W:P Fancy Ball  _[Select the card(s) to banish]_
- Mitsurugi selects Dharc the Dark Charmer, Gloomy  _[Select the card(s) to banish]_
- Mitsurugi selects Ecclesia and the Dark Dragon  _[Select the card(s) to banish]_
- Mitsurugi selects Albion the Branded Dragon  _[Select the card(s) to banish]_
- Mitsurugi selects Mitsurugi no Mikoto, Saji  _[Select the card(s) to add to your hand]_
- Mitsurugi orders Ame no Murakumo no Mitsurugi
- Mitsurugi orders Pre-Preparation of Rites
- Mitsurugi orders Pre-Preparation of Rites
- Mitsurugi orders Mitsurugi Great Purification
- Mitsurugi Normal Summons Mitsurugi no Mikoto, Saji
- Mitsurugi uses the effect of Mitsurugi no Mikoto, Saji
- Mitsurugi selects Mitsurugi Ritual  _[Select the card(s) to add to your hand]_
- Mitsurugi sets Dominus Impulse
- Mitsurugi ends the turn

## Turn 2: Wiccanthrope 's turn  (LP Mitsurugi 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Mitsurugi**
  - hand (3): End of the World Ruler, End of the World Ruler, Mitsurugi Ritual
  - monsters: Mitsurugi no Mikoto, Saji (ATK)
  - spells/traps: Dominus Impulse (set)
  - GY (2): Mulcharmy Purulia, Pot of Prosperity
  - banished (6): S:P Little Knight, I:P Masquerena, W:P Fancy Ball, Dharc the Dark Charmer, Gloomy, Albion the Branded Dragon, Ecclesia and the Dark Dragon
  - deck 34, extra deck 9
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment, Wiccanthrope Serwapentar, Mulcharmy Fuwalos
  - monsters: -
  - spells/traps: -
  - GY (1): Mulcharmy Fuwalos
  - deck 34, extra deck 14

Plays:

- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Wiccanthrope selects Wiccanthrope Road  _[Select the card(s) to send to the GY]_
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects Mitsurugi no Mikoto, Saji  _[Select the target(s) of the effect]_
- Wiccanthrope Special Summons Wiccanthrope Walioright
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to add to your hand]_
- Wiccanthrope Normal Summons Wiccanthrope Befaist
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to banish]_
- Wiccanthrope selects opponent's face-down card (Spell/Trap Zone)  _[Select the card(s) to destroy]_
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Wiccanthrope activates Wiccanthrope Road ("Add 1 "Wiccanthrope" monster from your Deck to your hand")
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Walioright ("Shuffle up to 3 cards into the Decks")
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to banish]_
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to return to the Deck]_
- Wiccanthrope selects Mulcharmy Fuwalos  _[Select the card(s) to return to the Deck]_
- Wiccanthrope selects Mulcharmy Fuwalos  _[Select the card(s) to return to the Deck]_
- Wiccanthrope Special Summons Wiccanthrope Spedallacer
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Spedallacer
- Wiccanthrope selects Wiccanthrope Road  _[Select the card(s) to add to your hand]_
- Wiccanthrope Special Summons Wiccanthrope Sperelfler
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope sets Wiccanthrope Road
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Befaist
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope goes to Main Phase 2
- Wiccanthrope ends the turn

## Turn 3: Mitsurugi 's turn  (LP Mitsurugi 3400 / Wiccanthrope 8000)

At the start of the turn:

- **Mitsurugi**
  - hand (4): End of the World Ruler, End of the World Ruler, Mitsurugi Ritual, Droll & Lock Bird
  - monsters: -
  - spells/traps: -
  - GY (4): Mulcharmy Purulia, Pot of Prosperity, Mitsurugi no Mikoto, Saji, Dominus Impulse
  - banished (6): S:P Little Knight, I:P Masquerena, W:P Fancy Ball, Dharc the Dark Charmer, Gloomy, Albion the Branded Dragon, Ecclesia and the Dark Dragon
  - deck 33, extra deck 9
- **Wiccanthrope**
  - hand (2): Wiccanthrope Serwapentar, Wiccanthrope Serwapentar
  - monsters: Wiccanthrope Befaist (ATK), Wiccanthrope Sperelfler (ATK)
  - spells/traps: Wiccanthrope Road (set)
  - GY (1): Wiccanthrope Road
  - banished (1): Wiccanthrope Enchantment
  - deck 32, extra deck 12

Plays:

- Mitsurugi sets monster Droll & Lock Bird
- Mitsurugi sets Mitsurugi Ritual
- Mitsurugi enters the Battle Phase
- Mitsurugi goes to Main Phase 2
- Wiccanthrope activates Wiccanthrope Road ("Add 1 "Wiccanthrope" monster from your Deck to your hand")
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to add to your hand]_

## Turn 4: Wiccanthrope 's turn  (LP Mitsurugi 3400 / Wiccanthrope 8000)

At the start of the turn:

- **Mitsurugi**
  - hand (2): End of the World Ruler, End of the World Ruler
  - monsters: Droll & Lock Bird (face-down DEF)
  - spells/traps: Mitsurugi Ritual (set)
  - GY (4): Mulcharmy Purulia, Pot of Prosperity, Mitsurugi no Mikoto, Saji, Dominus Impulse
  - banished (6): S:P Little Knight, I:P Masquerena, W:P Fancy Ball, Dharc the Dark Charmer, Gloomy, Albion the Branded Dragon, Ecclesia and the Dark Dragon
  - deck 33, extra deck 9
- **Wiccanthrope**
  - hand (4): Wiccanthrope Serwapentar, Wiccanthrope Walioright, Wiccanthrope Serwapentar, Foolish Burial
  - monsters: Wiccanthrope Befaist (ATK), Wiccanthrope Sperelfler (ATK)
  - spells/traps: -
  - GY (2): Wiccanthrope Road, Wiccanthrope Road
  - banished (1): Wiccanthrope Enchantment
  - deck 30, extra deck 12

Plays:

- Wiccanthrope Special Summons Wiccanthrope Walioright
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the card(s) to add to your hand]_
- Wiccanthrope Normal Summons Wiccanthrope Serwapentar
- Wiccanthrope uses the effect of Wiccanthrope Serwapentar
- Wiccanthrope answers YES ("Levels and Ranks of all monsters become 4")
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects opponent's face-down card (Monster Zone)  _[Select the target(s) of the effect]_
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope attacks with Wiccanthrope Befaist

## Game over: Wiccanthrope  wins (LP before the final blow: Mitsurugi 900 / Wiccanthrope 8000)

</details>
<details><summary><b>Game 9</b>: Wiccanthrope (first) vs Tearlaments: Tearlaments wins on turn 4</summary>


- Goes first: Wiccanthrope 
- Result: **Tearlaments  wins** on turn 4
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261017; 120 decisions in total, 45 searched, 25 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Wiccanthrope 's turn  (LP Wiccanthrope 8000 / Tearlaments 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (5): Effect Veiler, Wiccanthrope Befaist, Mulcharmy Fuwalos, Mulcharmy Purulia, Ash Blossom & Joyous Spring
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14
- **Tearlaments**
  - hand (5): Tearlaments Cryme, Primeval Planet Perlereino, Tearlaments Merrli, Tearlaments Merrli, Keldo the Sacred Protector
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 38, extra deck 15

Plays:

- Wiccanthrope activates Mulcharmy Purulia ("Apply the "●" effects for the rest of the turn")
- Wiccanthrope Normal Summons Wiccanthrope Befaist
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Wiccanthrope selects Wiccanthrope Road  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Road ("Add 1 "Wiccanthrope" monster from your Deck to your hand")
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")
- Wiccanthrope uses the effect of Wiccanthrope Road
- Wiccanthrope ends the turn
- Wiccanthrope activates Mulcharmy Purulia

## Turn 2: Tearlaments 's turn  (LP Wiccanthrope 8000 / Tearlaments 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (5): Wiccanthrope Befaist, Ash Blossom & Joyous Spring, Wiccanthrope Befaist, Effect Veiler, Mulcharmy Fuwalos
  - monsters: -
  - spells/traps: -
  - GY (1): Mulcharmy Purulia
  - banished (1): Wiccanthrope Road
  - deck 33, extra deck 14
- **Tearlaments**
  - hand (6): Tearlaments Cryme, Primeval Planet Perlereino, Tearlaments Merrli, Tearlaments Merrli, Keldo the Sacred Protector, Primeval Planet Perlereino
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 37, extra deck 15

Plays:

- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn"))**
- Tearlaments Normal Summons Tearlaments Merrli  **(search: chose this over the network's first choice: activates Primeval Planet Perlereino)**
- Tearlaments uses the effect of Tearlaments Merrli
- Wiccanthrope activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster")  **(search: chose this over the network's first choice: activates Ash Blossom & Joyous Spring ("Negate that effect"))**
- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")  **(search: chose this over the network's first choice: passes)**
- Tearlaments activates Primeval Planet Perlereino
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")
- Tearlaments enters the Battle Phase  **(search: chose this over the network's first choice: sets Primeval Planet Perlereino)**
- Tearlaments attacks with Tearlaments Merrli
- Tearlaments ends the turn  **(search: chose this over the network's first choice: goes to Main Phase 2)**

## Turn 3: Wiccanthrope 's turn  (LP Wiccanthrope 6700 / Tearlaments 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (3): Wiccanthrope Befaist, Wiccanthrope Befaist, Infinite Impermanence
  - monsters: -
  - spells/traps: -
  - GY (4): Mulcharmy Purulia, Effect Veiler, Mulcharmy Fuwalos, Ash Blossom & Joyous Spring
  - banished (1): Wiccanthrope Road
  - deck 32, extra deck 14
- **Tearlaments**
  - hand (4): Tearlaments Cryme, Tearlaments Merrli, Keldo the Sacred Protector, Primeval Planet Perlereino
  - monsters: Tearlaments Merrli (ATK)
  - spells/traps: Primeval Planet Perlereino
  - GY (0): -
  - deck 37, extra deck 15

Plays:

- Wiccanthrope sets Infinite Impermanence  **(search: chose this over the network's first choice: Normal Summons Wiccanthrope Befaist)**
- Wiccanthrope Normal Summons Wiccanthrope Befaist
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects Primeval Planet Perlereino  _[Select the target(s) of the effect]_
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")
- Wiccanthrope selects Tearlaments Merrli  _[Select the card(s) to destroy]_
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Tearlaments activates Tearlaments Merrli ("Fusion Summon")
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Merrli  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Merrli
- Tearlaments chooses position face-up Defense
- Tearlaments uses the effect of Tearlaments Kitkallos
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to add to your hand]_
- Tearlaments chooses ("Add the card(s) to your hand")
- Wiccanthrope enters the Battle Phase  **(search: chose this over the network's first choice: ends the turn)**
- Wiccanthrope attacks with Wiccanthrope Befaist  **(search: chose this over the network's first choice: goes to Main Phase 2)**
- Wiccanthrope selects Tearlaments Kitkallos  _[Select an attack target]_
- Wiccanthrope goes to Main Phase 2

## Turn 4: Tearlaments 's turn  (LP Wiccanthrope 6700 / Tearlaments 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (2): Wiccanthrope Befaist, Effect Veiler
  - monsters: Wiccanthrope Befaist (ATK)
  - spells/traps: Infinite Impermanence (set)
  - GY (4): Mulcharmy Purulia, Effect Veiler, Mulcharmy Fuwalos, Ash Blossom & Joyous Spring
  - banished (2): Wiccanthrope Road, Wiccanthrope Enchantment
  - deck 30, extra deck 14
- **Tearlaments**
  - hand (5): Keldo the Sacred Protector, Tearlaments Reinoheart, Primeval Planet Perlereino, Tearlaments Cryme, Tearlaments Reinoheart
  - monsters: -
  - spells/traps: -
  - GY (2): Primeval Planet Perlereino, Tearlaments Kitkallos
  - deck 37, extra deck 14

Plays:

- Tearlaments Normal Summons Tearlaments Reinoheart  **(search: chose this over the network's first choice: activates Primeval Planet Perlereino)**
- Tearlaments uses the effect of Tearlaments Reinoheart
- Wiccanthrope activates Infinite Impermanence
- Tearlaments activates Primeval Planet Perlereino
- Tearlaments answers YES ("Add 1 "Tearlaments" monster or 1 "Visas Starfrost" from your Deck to your hand?")
- Tearlaments selects Tearlaments Scheiren  _[Select the card(s) to add to your hand]_
- Tearlaments activates Tearlaments Scheiren ("Special Summon")
- Tearlaments chooses position face-up Attack
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to send to the GY]_  **(search: chose this over the network's first choice: selects Keldo the Sacred Protector)**
- Tearlaments activates Tearlaments Reinoheart ("Special Summon")
- Tearlaments activates Kelbek the Ancient Vanguard ("Mill the top 5 cards")
- Tearlaments chooses position face-up Attack
- Tearlaments activates Shaddoll Beast ("Draw 1 Card")
- Tearlaments activates Agido the Ancient Sentinel ("Send to GY")
- Tearlaments activates Tearlaments Havnis ("Fusion Summon")
- Tearlaments passes  **(search: chose this over the network's first choice: activates Tearlaments Sulliek ("Search"))**
- Tearlaments selects Tearlaments Rulkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to use as Fusion Material]_
- Tearlaments chooses position face-up Attack
- Tearlaments activates Tearlaments Merrli ("Fusion Summon")
- Tearlaments activates Tearlaments Scheiren ("Fusion Summon")
- Tearlaments activates Primeval Planet Perlereino ("Destroy 1 card on the field")
- Tearlaments selects Wiccanthrope Befaist  _[Select the card(s) to destroy]_
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster"))**
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Merrli  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Merrli
- Tearlaments chooses position face-up Attack
- Tearlaments uses the effect of Tearlaments Kitkallos
- Wiccanthrope activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster")
- Wiccanthrope selects Tearlaments Rulkallos  _[Select a face-up card(s)]_
- Tearlaments selects Tearlaments Havnis  _[Select the card(s) to add to your hand]_
- Tearlaments chooses ("Add the card(s) to your hand")
- Tearlaments enters the Battle Phase
- Tearlaments attacks with Tearlaments Scheiren  **(search: chose this over the network's first choice: attacks with Tearlaments Rulkallos)**
- Tearlaments attacks with Tearlaments Rulkallos
- Tearlaments attacks with Tearlaments Kitkallos

## Game over: Tearlaments  wins (LP before the final blow: Wiccanthrope 900 / Tearlaments 8000)

</details>
<details><summary><b>Game 10</b>: Wiccanthrope (first) vs Tearlaments: Tearlaments wins on turn 2</summary>


- Goes first: Wiccanthrope 
- Result: **Tearlaments  wins** on turn 2
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261018; 62 decisions in total, 34 searched, 20 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Wiccanthrope 's turn  (LP Wiccanthrope 8000 / Tearlaments 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (5): Wiccanthrope Reason, Ash Blossom & Joyous Spring, Wiccanthrope Enchantment, Mulcharmy Fuwalos, Wiccanthrope Rothockear
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14
- **Tearlaments**
  - hand (5): Tearlaments Reinoheart, Tearlaments Metanoise, Tearlaments Scheiren, Tearlaments Merrli, Tearlaments Scream
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 38, extra deck 15

Plays:

- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Wiccanthrope selects Wiccanthrope Road  _[Select the card(s) to send to the GY]_
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope sets Wiccanthrope Reason
- Wiccanthrope ends the turn

## Turn 2: Tearlaments 's turn  (LP Wiccanthrope 8000 / Tearlaments 8000)

At the start of the turn:

- **Wiccanthrope**
  - hand (1): Ash Blossom & Joyous Spring
  - monsters: Wiccanthrope Rothockear (ATK)
  - spells/traps: Wiccanthrope Reason (set)
  - GY (3): Mulcharmy Fuwalos, Wiccanthrope Road, Wiccanthrope Enchantment
  - deck 34, extra deck 14
- **Tearlaments**
  - hand (6): Tearlaments Reinoheart, Tearlaments Metanoise, Tearlaments Scheiren, Tearlaments Merrli, Tearlaments Scream, Tearlaments Cryme
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 37, extra deck 15

Plays:

- Tearlaments Normal Summons Tearlaments Reinoheart
- Tearlaments uses the effect of Tearlaments Reinoheart
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")
- Tearlaments activates Tearlaments Scream  **(search: chose this over the network's first choice: activates Tearlaments Scheiren ("Special Summon"))**
- Tearlaments activates Tearlaments Scheiren ("Special Summon")
- Tearlaments chooses position face-up Attack
- Tearlaments activates Tearlaments Scheiren ("Fusion Summon")  **(search: chose this over the network's first choice: activates Tearlaments Merrli ("Fusion Summon"))**
- Tearlaments activates Tearlaments Merrli ("Fusion Summon")
- Tearlaments activates Tearlaments Scream ("Send the top 3 cards of your Deck to the GY")
- Tearlaments passes  **(search: chose this over the network's first choice: activates Mudora the Sword Oracle ("Shuffle to Deck"))**
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Reinoheart  **(search: chose this over the network's first choice: orders Tearlaments Merrli)**
- Tearlaments chooses position face-up Attack
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to Special Summon]_  **(search: chose this over the network's first choice: selects Tearlaments Rulkallos)**
- Tearlaments selects Tearlaments Havnis  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Scheiren  **(search: chose this over the network's first choice: orders Tearlaments Havnis)**
- Tearlaments chooses position face-up Attack
- Tearlaments activates Tearlaments Kitkallos ("Search or send to GY")
- Tearlaments passes  **(search: chose this over the network's first choice: activates Agido the Ancient Sentinel ("Send to GY"))**
- Tearlaments passes  **(search: chose this over the network's first choice: activates Mudora the Sword Oracle ("Shuffle to Deck"))**
- Tearlaments selects Tearlaments Scheiren  _[Select the card(s) to add to your hand]_
- Tearlaments chooses ("Send the card(s) to the GY")  **(search: chose this over the network's first choice: chooses ("Add the card(s) to your hand"))**
- Tearlaments activates Mudora the Sword Oracle ("Shuffle to Deck")
- Tearlaments selects Wiccanthrope Enchantment  _[Select the card(s) to return to the Deck]_  **(search: chose this over the network's first choice: selects Mulcharmy Fuwalos)**
- Tearlaments selects Wiccanthrope Road  _[Select the card(s) to return to the Deck]_  **(search: chose this over the network's first choice: selects Mulcharmy Fuwalos)**
- Tearlaments finishes the selection  _[Select the card(s) to return to the Deck]_
- Tearlaments sets Tearlaments Metanoise  **(search: chose this over the network's first choice: enters the Battle Phase)**
- Tearlaments activates Tearlaments Kitkallos ("Special Summon")  **(search: chose this over the network's first choice: enters the Battle Phase)**
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to send to the GY]_  **(search: chose this over the network's first choice: selects Tearlaments Kitkallos)**
- Tearlaments chooses position face-up Attack
- Tearlaments uses the effect of Tearlaments Reinoheart
- Tearlaments chooses position face-up Attack
- Tearlaments enters the Battle Phase
- Tearlaments attacks with Tearlaments Kitkallos
- Tearlaments selects Wiccanthrope Rothockear  _[Select an attack target]_
- Tearlaments attacks with Tearlaments Reinoheart  **(search: chose this over the network's first choice: attacks with Tearlaments Scheiren)**
- Tearlaments attacks with Tearlaments Scheiren
- Tearlaments attacks with Tearlaments Scheiren
- Tearlaments attacks with Tearlaments Kitkallos

## Game over: Tearlaments  wins (LP before the final blow: Wiccanthrope 1200 / Tearlaments 8000)

</details>
<details><summary><b>Game 11</b>: Wiccanthrope (second) vs Tearlaments: Wiccanthrope wins on turn 2</summary>


- Goes first: Tearlaments 
- Result: **Wiccanthrope  wins** on turn 2
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261017; 115 decisions in total, 56 searched, 28 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Tearlaments 's turn  (LP Tearlaments 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Tearlaments**
  - hand (5): Tearlaments Grief, Tearlaments Havnis, Shaddoll Squamata, Tearlaments Havnis, Tearlaments Reinoheart
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 38, extra deck 15
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Ash Blossom & Joyous Spring, Effect Veiler, Mulcharmy Purulia
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14

Plays:

- Wiccanthrope activates Mulcharmy Purulia ("Apply the "●" effects for the rest of the turn")
- Tearlaments Normal Summons Tearlaments Reinoheart
- Tearlaments uses the effect of Tearlaments Reinoheart
- Wiccanthrope activates Effect Veiler ("Negate the effects of 1 opponent's Effect Monster")
- Tearlaments activates Tearlaments Grief ("Special Summon")
- Wiccanthrope activates Ash Blossom & Joyous Spring ("Negate that effect")

## Turn 2: Wiccanthrope 's turn  (LP Tearlaments 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Tearlaments**
  - hand (3): Tearlaments Havnis, Shaddoll Squamata, Tearlaments Havnis
  - monsters: Tearlaments Reinoheart (ATK)
  - spells/traps: -
  - GY (1): Tearlaments Grief
  - deck 38, extra deck 15
- **Wiccanthrope**
  - hand (4): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment, Wiccanthrope Serwapentar
  - monsters: -
  - spells/traps: -
  - GY (3): Mulcharmy Purulia, Effect Veiler, Ash Blossom & Joyous Spring
  - deck 33, extra deck 14

Plays:

- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Tearlaments activates Tearlaments Havnis ("Special Summon")
- Tearlaments chooses position face-up Defense
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to send to the GY]_
- Tearlaments uses the effect of Tearlaments Merrli
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Merrli
- Tearlaments chooses position face-up Defense
- Tearlaments uses the effect of Tearlaments Kitkallos
- Tearlaments selects Tearlaments Scheiren  _[Select the card(s) to add to your hand]_
- Tearlaments chooses ("Add the card(s) to your hand")
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")  **(search: chose this over the network's first choice: passes)**
- Wiccanthrope selects Tearlaments Havnis  _[Select the target(s) of the effect]_  **(search: chose this over the network's first choice: selects Tearlaments Kitkallos)**
- Tearlaments uses the effect of Tearlaments Havnis
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Havnis  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Havnis
- Tearlaments chooses position face-up Defense
- Wiccanthrope Special Summons Wiccanthrope Walioright
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the card(s) to add to your hand]_
- Wiccanthrope Normal Summons Wiccanthrope Serwapentar  **(search: chose this over the network's first choice: activates Wiccanthrope Walioright ("Shuffle up to 3 cards into the Decks"))**
- Wiccanthrope Special Summons Wiccanthrope Spedallacer  **(search: chose this over the network's first choice: activates Wiccanthrope Walioright ("Shuffle up to 3 cards into the Decks"))**
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Spedallacer
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Wiccanthrope Spedallacer ("Negate the effects of 1 face-up card"))**
- Wiccanthrope activates Wiccanthrope Spedallacer ("Negate the effects of 1 face-up card")
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the Xyz Material(s) to detach]_
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to negate its effects]_  **(search: chose this over the network's first choice: selects Tearlaments Kitkallos)**
- Wiccanthrope answers YES ("Banish 1 Spell and inflict 500 damage?")
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to banish]_
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Wiccanthrope Special Summons Wiccanthrope Sperelfler
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope activates Wiccanthrope Sperelfler ("Special Summon 1 "Wiccanthrope" monster from your GY")
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the Xyz Material(s) to detach]_  **(search: chose this over the network's first choice: selects Wiccanthrope Walioright)**
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the card(s) to Special Summon]_  **(search: chose this over the network's first choice: selects Wiccanthrope Serwapentar)**
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope answers YES ("Banish 1 Spell and attach 1 Spell from your Deck to an Xyz Monster?")
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select a face-up card(s)]_
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope Special Summons Wiccanthrope Stormgnarl  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Sperelfler)**
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope activates Wiccanthrope Serwapentar ("Special Summon this card from your GY")
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to discard]_  **(search: chose this over the network's first choice: selects Wiccanthrope Serwapentar)**
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Serwapentar
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the target(s) of the effect]_
- Wiccanthrope answers YES ("Levels and Ranks of all monsters become 4")
- Wiccanthrope Special Summons Wiccanthrope Spedallacer  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Befaist)**
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope Special Summons Wiccanthrope Befaist  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Sperelfler)**
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Wiccanthrope selects Wiccanthrope Reason  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Sperelfler)**
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to banish]_
- Wiccanthrope activates Wiccanthrope Reason ("Xyz Summon 1 "Wiccanthrope" Xyz Monster")  **(search: chose this over the network's first choice: passes)**
- Wiccanthrope selects Wiccanthrope Sperelfler  _[Select the card(s) to Special Summon]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope selects Tearlaments Kitkallos  _[Select the card(s) to destroy]_
- Tearlaments uses the effect of Tearlaments Kitkallos
- Tearlaments activates Tearlaments Scheiren ("Fusion Summon")  **(search: chose this over the network's first choice: activates Kelbek the Ancient Vanguard ("Mill the top 5 cards"))**
- Tearlaments passes  **(search: chose this over the network's first choice: activates Kelbek the Ancient Vanguard ("Mill the top 5 cards"))**
- Tearlaments selects Tearlaments Rulkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to use as Fusion Material]_
- Tearlaments chooses position face-up Attack
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Sperelfler  **(search: chose this over the network's first choice: attacks with Wiccanthrope Stormgnarl)**
- Wiccanthrope selects Tearlaments Rulkallos  _[Select an attack target]_
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope selects Tearlaments Kitkallos  _[Select an attack target]_
- Wiccanthrope attacks with Wiccanthrope Befaist
- Wiccanthrope attacks with Wiccanthrope Stormgnarl

## Game over: Wiccanthrope  wins (LP before the final blow: Tearlaments 3500 / Wiccanthrope 8000)

</details>
<details><summary><b>Game 12</b>: Wiccanthrope (second) vs Tearlaments: Wiccanthrope wins on turn 4</summary>


- Goes first: Tearlaments 
- Result: **Wiccanthrope  wins** on turn 4
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 20261018; 189 decisions in total, 68 searched, 26 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Tearlaments 's turn  (LP Tearlaments 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Tearlaments**
  - hand (5): Kelbek the Ancient Vanguard, Primeval Planet Perlereino, Polymerization, Called by the Grave, Keldo the Sacred Protector
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 38, extra deck 15
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment, Mulcharmy Fuwalos, Wiccanthrope Serwapentar
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 14

Plays:

- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn"))**
- Tearlaments activates Primeval Planet Perlereino
- Tearlaments answers YES ("Add 1 "Tearlaments" monster or 1 "Visas Starfrost" from your Deck to your hand?")
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to add to your hand]_
- Tearlaments Normal Summons Tearlaments Reinoheart
- Tearlaments uses the effect of Tearlaments Reinoheart
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn"))**
- Tearlaments selects Tearlaments Scheiren  _[Select the card(s) to send to the GY]_
- Tearlaments uses the effect of Tearlaments Scheiren
- Wiccanthrope activates Mulcharmy Fuwalos ("Apply the "●" effects for the rest of the turn")
- Tearlaments activates Called by the Grave
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Scheiren
- Tearlaments chooses position face-up Attack
- Tearlaments activates Tearlaments Kitkallos ("Search or send to GY")
- Tearlaments activates Primeval Planet Perlereino ("Destroy 1 card on the field")
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to destroy]_
- Tearlaments selects Tearlaments Sulliek  _[Select the card(s) to add to your hand]_
- Tearlaments chooses ("Add the card(s) to your hand")
- Tearlaments uses the effect of Tearlaments Kitkallos
- Tearlaments activates Tearlaments Merrli ("Fusion Summon")
- Tearlaments activates Shaddoll Beast ("Draw 1 Card")
- Tearlaments activates Tearlaments Metanoise ("Add to hand")
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to add to your hand]_
- Tearlaments selects El Shaddoll Winda  _[Select the card(s) to Special Summon]_
- Tearlaments selects Shaddoll Beast  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Merrli
- Tearlaments chooses position face-up Attack
- Tearlaments sets Tearlaments Sulliek
- Tearlaments activates Keldo the Sacred Protector ("Special Summon")
- Tearlaments chooses position face-up Attack
- Tearlaments selects Kelbek the Ancient Vanguard  _[Select the card(s) to add to your hand]_
- Tearlaments uses the effect of Kelbek the Ancient Vanguard
- Tearlaments uses the effect of Agido the Ancient Sentinel
- Tearlaments sets Polymerization
- Tearlaments ends the turn

## Turn 2: Wiccanthrope 's turn  (LP Tearlaments 8000 / Wiccanthrope 8000)

At the start of the turn:

- **Tearlaments**
  - hand (2): Ash Blossom & Joyous Spring, Kelbek the Ancient Vanguard
  - monsters: Keldo the Sacred Protector (ATK), El Shaddoll Winda (ATK)
  - spells/traps: Polymerization (set), Tearlaments Sulliek (set), Primeval Planet Perlereino
  - GY (15): Called by the Grave, Tearlaments Metanoise, Primeval Planet Perlereino, Ghost Belle & Haunted Mansion, Kelbek the Ancient Vanguard, Agido the Ancient Sentinel, Tearlaments Merrli, Tearlaments Merrli, Super Polymerization, Tearlaments Reinoheart, Shaddoll Beast, Tearlaments Cryme, Tearlaments Havnis, Tearlaments Grief, Shaddoll Squamata
  - deck 22, extra deck 14
- **Wiccanthrope**
  - hand (5): Wiccanthrope Walioright, Wiccanthrope Rothockear, Wiccanthrope Enchantment, Wiccanthrope Serwapentar, Wiccanthrope Befaist
  - monsters: -
  - spells/traps: -
  - GY (10): Mulcharmy Fuwalos, Wiccanthrope Befaist, Mulcharmy Fuwalos, Ghost Mourner & Moonlit Chill, Wiccanthrope Befaist, Wiccanthrope Enchantment, Wiccanthrope Rothockear, Mulcharmy Purulia, Wiccanthrope Rothockear, Ash Blossom & Joyous Spring
  - banished (1): Mulcharmy Fuwalos
  - deck 24, extra deck 14

Plays:

- Wiccanthrope Normal Summons Wiccanthrope Walioright  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Rothockear)**
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Tearlaments activates Ash Blossom & Joyous Spring ("Negate that effect")
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")  **(search: chose this over the network's first choice: passes)**
- Wiccanthrope selects opponent's face-down card (Spell/Trap Zone)  _[Select the target(s) of the effect]_  **(search: chose this over the network's first choice: selects Primeval Planet Perlereino)**
- Wiccanthrope Special Summons Super Starslayer TY-PHON - Sky Crisis  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Befaist)**
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Tearlaments activates Tearlaments Sulliek
- Tearlaments passes  **(search: chose this over the network's first choice: activates Keldo the Sacred Protector ("Shuffle to Deck"))**
- Tearlaments activates Keldo the Sacred Protector ("Shuffle to Deck")
- Tearlaments selects Wiccanthrope Enchantment  _[Select the card(s) to return to the Deck]_
- Tearlaments selects Wiccanthrope Enchantment  _[Select the card(s) to return to the Deck]_
- Tearlaments selects Wiccanthrope Befaist  _[Select the card(s) to return to the Deck]_
- Wiccanthrope activates Super Starslayer TY-PHON - Sky Crisis ("Return 1 monster on the field to the hand")
- Wiccanthrope selects El Shaddoll Winda  _[Select the card(s) to return to the hand]_
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Super Starslayer TY-PHON - Sky Crisis
- Wiccanthrope goes to Main Phase 2

## Turn 3: Tearlaments 's turn  (LP Tearlaments 5100 / Wiccanthrope 8000)

At the start of the turn:

- **Tearlaments**
  - hand (2): Kelbek the Ancient Vanguard, Foolish Burial
  - monsters: -
  - spells/traps: Tearlaments Sulliek, Primeval Planet Perlereino
  - GY (17): Called by the Grave, Tearlaments Metanoise, Primeval Planet Perlereino, Ghost Belle & Haunted Mansion, Kelbek the Ancient Vanguard, Agido the Ancient Sentinel, Tearlaments Merrli, Tearlaments Merrli, Super Polymerization, Tearlaments Reinoheart, Shaddoll Beast, Tearlaments Cryme, Tearlaments Havnis, Tearlaments Grief, Shaddoll Squamata, Ash Blossom & Joyous Spring, Polymerization
  - banished (1): Keldo the Sacred Protector
  - deck 21, extra deck 15
- **Wiccanthrope**
  - hand (3): Wiccanthrope Rothockear, Wiccanthrope Serwapentar, Wiccanthrope Befaist
  - monsters: Super Starslayer TY-PHON - Sky Crisis (ATK)
  - spells/traps: -
  - GY (9): Mulcharmy Fuwalos, Mulcharmy Fuwalos, Ghost Mourner & Moonlit Chill, Wiccanthrope Befaist, Wiccanthrope Rothockear, Mulcharmy Purulia, Wiccanthrope Rothockear, Ash Blossom & Joyous Spring, Wiccanthrope Walioright
  - banished (1): Mulcharmy Fuwalos
  - deck 27, extra deck 13

Plays:

- Tearlaments activates Foolish Burial
- Tearlaments selects Tearlaments Havnis  _[Select the card(s) to send to the GY]_
- Tearlaments uses the effect of Tearlaments Havnis
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to use as Fusion Material]_
- Tearlaments orders Tearlaments Havnis
- Tearlaments chooses position face-up Attack
- Tearlaments activates Tearlaments Kitkallos ("Search or send to GY")
- Tearlaments activates Primeval Planet Perlereino ("Destroy 1 card on the field")
- Tearlaments selects Super Starslayer TY-PHON - Sky Crisis  _[Select the card(s) to destroy]_
- Tearlaments activates Tearlaments Sulliek ("Negate")
- Tearlaments selects Tearlaments Scheiren  _[Select the card(s) to add to your hand]_
- Tearlaments chooses ("Add the card(s) to your hand")
- Tearlaments uses the effect of Tearlaments Kitkallos
- Tearlaments activates Tearlaments Reinoheart ("Special Summon")
- Tearlaments chooses position face-up Attack
- Tearlaments activates Tearlaments Reinoheart ("Special Summon")
- Tearlaments activates Tearlaments Scheiren ("Fusion Summon")
- Tearlaments selects Tearlaments Rulkallos  _[Select the card(s) to Special Summon]_
- Tearlaments selects Tearlaments Kitkallos  _[Select the card(s) to use as Fusion Material]_
- Tearlaments chooses position face-up Attack
- Tearlaments selects Tearlaments Havnis  _[Select the card(s) to send to the GY]_
- Tearlaments Normal Summons Kelbek the Ancient Vanguard
- Tearlaments enters the Battle Phase
- Tearlaments attacks with Tearlaments Rulkallos  **(search: chose this over the network's first choice: attacks with Kelbek the Ancient Vanguard)**
- Tearlaments attacks with Tearlaments Reinoheart  **(search: chose this over the network's first choice: attacks with Kelbek the Ancient Vanguard)**
- Tearlaments attacks with Kelbek the Ancient Vanguard
- Tearlaments ends the turn

## Turn 4: Wiccanthrope 's turn  (LP Tearlaments 5100 / Wiccanthrope 1000)

At the start of the turn:

- **Tearlaments**
  - hand (0): -
  - monsters: Tearlaments Reinoheart (ATK), Kelbek the Ancient Vanguard (ATK), Tearlaments Rulkallos (ATK)
  - spells/traps: Tearlaments Sulliek, Primeval Planet Perlereino
  - GY (22): Called by the Grave, Tearlaments Metanoise, Primeval Planet Perlereino, Ghost Belle & Haunted Mansion, Kelbek the Ancient Vanguard, Agido the Ancient Sentinel, Tearlaments Merrli, Tearlaments Merrli, Super Polymerization, Shaddoll Beast, Tearlaments Cryme, Tearlaments Havnis, Tearlaments Grief, Shaddoll Squamata, Ash Blossom & Joyous Spring, Polymerization, Foolish Burial, Tearlaments Reinoheart, Polymerization, Ghost Belle & Haunted Mansion, Tearlaments Scream, Tearlaments Havnis
  - banished (1): Keldo the Sacred Protector
  - deck 16, extra deck 14
- **Wiccanthrope**
  - hand (4): Wiccanthrope Rothockear, Wiccanthrope Serwapentar, Wiccanthrope Befaist, Ash Blossom & Joyous Spring
  - monsters: -
  - spells/traps: -
  - GY (10): Mulcharmy Fuwalos, Mulcharmy Fuwalos, Ghost Mourner & Moonlit Chill, Wiccanthrope Befaist, Wiccanthrope Rothockear, Mulcharmy Purulia, Wiccanthrope Rothockear, Ash Blossom & Joyous Spring, Wiccanthrope Walioright, Super Starslayer TY-PHON - Sky Crisis
  - banished (1): Mulcharmy Fuwalos
  - deck 26, extra deck 13

Plays:

- Wiccanthrope Special Summons Wiccanthrope Rothockear
- Wiccanthrope chooses position face-up Defense
- Wiccanthrope uses the effect of Wiccanthrope Rothockear
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to send to the GY]_
- Wiccanthrope Normal Summons Wiccanthrope Serwapentar  **(search: chose this over the network's first choice: Normal Summons Wiccanthrope Befaist)**
- Wiccanthrope uses the effect of Wiccanthrope Serwapentar
- Wiccanthrope selects Mulcharmy Fuwalos  _[Select the target(s) of the effect]_  **(search: chose this over the network's first choice: selects Keldo the Sacred Protector)**
- Wiccanthrope activates Wiccanthrope Serwapentar ("Add it to your hand instead?")  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Befaist)**
- Tearlaments activates Tearlaments Sulliek ("Negate")  **(search: chose this over the network's first choice: passes)**
- Tearlaments selects Wiccanthrope Rothockear  _[Select the card(s) to negate its effects]_
- Tearlaments selects Tearlaments Reinoheart  _[Select the card(s) to send to the GY]_
- Wiccanthrope uses the effect of Wiccanthrope Enchantment
- Wiccanthrope Special Summons Wiccanthrope Befaist
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Befaist
- Wiccanthrope selects Wiccanthrope Enchantment  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Enchantment ("Target 1 card on the field")
- Wiccanthrope selects Tearlaments Rulkallos  _[Select the target(s) of the effect]_
- Tearlaments uses the effect of Tearlaments Rulkallos
- Tearlaments chooses position face-up Attack
- Wiccanthrope activates Wiccanthrope Befaist ("Destroy 1 card on the field")
- Wiccanthrope selects Tearlaments Rulkallos  _[Select the card(s) to destroy]_
- Wiccanthrope Special Summons Wiccanthrope Spedallacer
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope unselects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Serwapentar)**
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Spedallacer
- Wiccanthrope selects Wiccanthrope Walioright  _[Select the card(s) to add to your hand]_
- Wiccanthrope activates Wiccanthrope Spedallacer ("Negate the effects of 1 face-up card")
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the Xyz Material(s) to detach]_
- Wiccanthrope selects Kelbek the Ancient Vanguard  _[Select the card(s) to negate its effects]_
- Wiccanthrope Special Summons Wiccanthrope Sperelfler
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope activates Wiccanthrope Sperelfler ("Special Summon 1 "Wiccanthrope" monster from your GY")
- Wiccanthrope selects Wiccanthrope Spedallacer  _[Select the Xyz Material(s) to detach]_
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to Special Summon]_  **(search: chose this over the network's first choice: selects Wiccanthrope Serwapentar)**
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope Special Summons Wiccanthrope Wataquera  **(search: chose this over the network's first choice: Special Summons Wiccanthrope Walioright)**
- Wiccanthrope selects Wiccanthrope Befaist  _[Select the card(s) to use as Xyz Material]_  **(search: chose this over the network's first choice: selects Wiccanthrope Rothockear)**
- Wiccanthrope selects Wiccanthrope Rothockear  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Wataquera
- Wiccanthrope selects Primeval Planet Perlereino  _[Select the card(s) to use as Xyz Material]_
- Wiccanthrope passes  **(search: chose this over the network's first choice: activates Wiccanthrope Wataquera ("Destroy 1 card on the field"))**
- Wiccanthrope activates Wiccanthrope Wataquera ("Destroy 1 card on the field")
- Wiccanthrope selects Primeval Planet Perlereino  _[Select the Xyz Material(s) to detach]_
- Wiccanthrope selects Kelbek the Ancient Vanguard  _[Select the card(s) to destroy]_
- Wiccanthrope Special Summons Wiccanthrope Walioright
- Wiccanthrope chooses position face-up Attack
- Wiccanthrope uses the effect of Wiccanthrope Walioright
- Wiccanthrope selects Wiccanthrope Serwapentar  _[Select the card(s) to add to your hand]_
- Wiccanthrope enters the Battle Phase
- Wiccanthrope attacks with Wiccanthrope Sperelfler
- Wiccanthrope attacks with Wiccanthrope Wataquera

## Game over: Wiccanthrope  wins (LP before the final blow: Tearlaments 2000 / Wiccanthrope 1000)

</details>
