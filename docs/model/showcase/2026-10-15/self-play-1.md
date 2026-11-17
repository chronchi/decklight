# Our AI vs itself, game 1: Our AI #1 (mitsurugi-meta1) vs Our AI #2 (snake-eye)

- Goes first: Our AI #1 (mitsurugi-meta1)
- Result: **Our AI #2 (snake-eye) wins** on turn 4
- Decks are named by their file in decks/ (e.g. the pool file `branded` is a Dracotail list).
- Deal seed 12345; 190 decisions in total, 79 searched, 38 changed by the search (changes of zone or between identical-looking options are not marked)
- Spectator view: both hands are shown at each turn start (the players could not see each other's hands).
- Passes and zone choices are left out; card selections show the engine's prompt in _[brackets]_.

## Turn 1: Our AI #1 (mitsurugi-meta1)'s turn  (LP Our AI #1 8000 / Our AI #2 8000)

At the start of the turn:

- **Our AI #1**
  - hand (5): End of the World Ruler, The Fallen & The Virtuous, Mitsurugi no Mikoto, Aramasa, End of the World Ruler, Mitsurugi no Mikoto, Saji
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 15
- **Our AI #2**
  - hand (5): Fire King Avatar Barong, Original Sinful Spoils - Snake-Eye, Legendary Fire King Ponix, Snake-Eyes Poplar, Called by the Grave
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 36, extra deck 15

Plays:

- Our AI #1 Normal Summons Mitsurugi no Mikoto, Saji
- Our AI #1 uses the effect of Mitsurugi no Mikoto, Saji
- Our AI #1 selects Mitsurugi Ritual  _[Select the card(s) to add to your hand]_
- Our AI #1 activates Mitsurugi Ritual ("Activate 1 of these effects")
- Our AI #1 selects Ame no Habakiri no Mitsurugi  _[Select the card(s) to Special Summon]_
- Our AI #1 selects Mitsurugi no Mikoto, Saji  _[Select the card(s) to Tribute]_
- Our AI #1 chooses position face-up Attack
- Our AI #1 uses the effect of Mitsurugi no Mikoto, Aramasa
- Our AI #1 selects Mitsurugi no Mikoto, Kusanagi  _[Select the card(s) to add to your hand]_
- Our AI #1 activates End of the World Ruler ("Apply that card's activation effect that Ritual Summons a monster")
- Our AI #1 selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to Special Summon]_
- Our AI #1 chooses position face-up Attack
- Our AI #1 uses the effect of Ame no Habakiri no Mitsurugi
- Our AI #1 selects Mitsurugi no Mikoto, Aramasa  _[Select the card(s) to add to your hand]_
- Our AI #1 answers YES ("Special Summon this card?")
- Our AI #1 chooses position face-up Attack
- Our AI #1 sets The Fallen & The Virtuous
- Our AI #1 ends the turn

## Turn 2: Our AI #2 (snake-eye)'s turn  (LP Our AI #1 6000 / Our AI #2 8000)

At the start of the turn:

- **Our AI #1**
  - hand (4): Mitsurugi no Mikoto, Aramasa, End of the World Ruler, End of the World Ruler, Mitsurugi no Mikoto, Kusanagi
  - monsters: Ame no Murakumo no Mitsurugi (ATK), Ame no Habakiri no Mitsurugi (ATK)
  - spells/traps: The Fallen & The Virtuous (set)
  - GY (3): Mitsurugi no Mikoto, Aramasa, Mitsurugi no Mikoto, Saji, Mitsurugi Ritual
  - banished (1): Mitsurugi Ritual
  - deck 29, extra deck 15
- **Our AI #2**
  - hand (6): Fire King Avatar Barong, Original Sinful Spoils - Snake-Eye, Legendary Fire King Ponix, Snake-Eyes Poplar, Called by the Grave, Triple Tactics Thrust
  - monsters: -
  - spells/traps: -
  - GY (0): -
  - deck 35, extra deck 15

Plays:

- Our AI #2 activates Called by the Grave  **(search: chose this over the network's first choice: Normal Summons Snake-Eyes Poplar)**
- Our AI #2 selects Mitsurugi no Mikoto, Saji  _[Select the card(s) to banish]_
- Our AI #1 activates Ame no Murakumo no Mitsurugi ("Your opponent can discard 1 card, or that effect is negated")  **(search: chose this over the network's first choice: passes)**
- Our AI #2 chooses ("That effect is negated")  **(search: chose this over the network's first choice: chooses ("Discard 1 card"))**
- Our AI #2 activates Triple Tactics Thrust ("Set 1 Normal Spell/Trap from your Deck")
- Our AI #1 activates The Fallen & The Virtuous ("Activate 1 of these effects")  **(search: chose this over the network's first choice: passes)**
- Our AI #1 selects Albion the Branded Dragon  _[Select the card(s) to send to the GY]_
- Our AI #1 selects Triple Tactics Thrust  _[Select the card(s) to destroy]_
- Our AI #2 selects Harpie's Feather Duster  _[Select the card to Set or add to your hand]_
- Our AI #2 answers YES ("Add it to your hand instead?")
- Our AI #2 Normal Summons Legendary Fire King Ponix  **(search: chose this over the network's first choice: Normal Summons Snake-Eyes Poplar)**
- Our AI #2 uses the effect of Legendary Fire King Ponix
- Our AI #2 selects Fire King Sanctuary  _[Select the card(s) to add to your hand]_
- Our AI #2 activates Original Sinful Spoils - Snake-Eye ("Special Summon 1 Level 1 FIRE monster from your hand/Deck")  **(search: chose this over the network's first choice: activates Fire King Sanctuary ("Place 1 "Fire King Island" from your Deck face-up in your Field Zone?"))**
- Our AI #2 selects Snake-Eye Oak  _[Select the card(s) to Special Summon]_
- Our AI #2 chooses position face-up Defense  **(search: chose this over the network's first choice: chooses position face-up Attack)**
- Our AI #2 uses the effect of Snake-Eye Oak
- Our AI #2 chooses ("Add the card(s) to your hand")  **(search: chose this over the network's first choice: chooses ("Special Summon it"))**
- Our AI #2 activates Fire King Sanctuary ("Place 1 "Fire King Island" from your Deck face-up in your Field Zone?")
- Our AI #2 answers YES ("Place 1 "Fire King Island" from your Deck face-up in your Field Zone?")
- Our AI #2 activates Fire King Island ("Destroy 1 monster and search 1 "Fire King" monster")  **(search: chose this over the network's first choice: activates Snake-Eye Oak ("Special Summon 1 "Snake-Eye" from your hand or Deck"))**
- Our AI #2 selects Fire King Avatar Barong  _[Select the card(s) to destroy]_  **(search: chose this over the network's first choice: selects Snake-Eyes Poplar)**
- Our AI #2 selects Fire King High Avatar Kirin  _[Select the card(s) to add to your hand]_
- Our AI #2 uses the effect of Legendary Fire King Ponix
- Our AI #2 chooses position face-up Attack
- Our AI #2 activates Fire King High Avatar Kirin ("Destroy 1 other FIRE monster and Special Summon this card")  **(search: chose this over the network's first choice: activates Snake-Eye Oak ("Special Summon 1 "Snake-Eye" from your hand or Deck"))**
- Our AI #2 selects Legendary Fire King Ponix  _[Select the card(s) to destroy]_  **(search: chose this over the network's first choice: selects Snake-Eyes Poplar)**
- Our AI #2 chooses position face-up Attack
- Our AI #2 activates Snake-Eye Oak ("Special Summon 1 "Snake-Eye" from your hand or Deck")
- Our AI #2 selects Fire King Island  _[Select the card(s) to send to the GY]_  **(search: chose this over the network's first choice: selects Fire King Sanctuary)**
- Our AI #2 selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to Special Summon]_
- Our AI #2 chooses position face-up Attack
- Our AI #2 activates Snake-Eyes Flamberge Dragon ("Special Summon 2 Level 1 FIRE monsters from your GY")
- Our AI #2 activates Fire King High Avatar Kirin ("Special Summon 1 "Fire King" from your hand or GY")
- Our AI #2 selects Fire King Avatar Barong  _[Select the card(s) to Special Summon]_
- Our AI #2 chooses position face-up Attack
- Our AI #2 answers YES ("Destroy 1 card on the field?")
- Our AI #2 selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to destroy]_
- Our AI #2 selects Snake-Eye Oak  _[Select the card(s) to Special Summon]_
- Our AI #2 chooses position face-up Attack
- Our AI #2 chooses position face-up Attack
- Our AI #2 Special Summons Promethean Princess, Bestower of Flames  **(search: chose this over the network's first choice: Special Summons Hiita the Fire Charmer, Ablaze)**
- Our AI #2 selects Legendary Fire King Ponix  _[Select the card(s) to use as Link Material]_
- Our AI #2 selects Snake-Eye Oak  _[Select the card(s) to use as Link Material]_
- Our AI #2 selects Fire King Avatar Barong  _[Select the card(s) to use as Link Material]_
- Our AI #2 activates Promethean Princess, Bestower of Flames ("Special Summon 1 FIRE monster from your GY")
- Our AI #2 selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to Special Summon]_
- Our AI #2 chooses position face-up Attack
- Our AI #2 activates Snake-Eyes Flamberge Dragon ("Place 1 monster in its owner's Spell & Trap Zone as a Continuous Spell")
- Our AI #2 selects Ame no Habakiri no Mitsurugi  _[Select the target(s) of the effect]_
- Our AI #2 activates Harpie's Feather Duster
- Our AI #2 enters the Battle Phase
- Our AI #2 attacks with Snake-Eyes Flamberge Dragon
- Our AI #2 attacks with Promethean Princess, Bestower of Flames
- Our AI #2 goes to Main Phase 2  **(search: chose this over the network's first choice: ends the turn)**
- Our AI #2 ends the turn
- Our AI #1 uses the effect of Albion the Branded Dragon
- Our AI #1 chooses ("Add the card(s) to your hand")

## Turn 3: Our AI #1 (mitsurugi-meta1)'s turn  (LP Our AI #1 300 / Our AI #2 8000)

At the start of the turn:

- **Our AI #1**
  - hand (6): End of the World Ruler, Mitsurugi no Mikoto, Kusanagi, End of the World Ruler, Mitsurugi no Mikoto, Aramasa, The Fallen & The Virtuous, Super Polymerization
  - monsters: -
  - spells/traps: -
  - GY (7): Mitsurugi no Mikoto, Aramasa, Mitsurugi no Mikoto, Saji, Mitsurugi Ritual, Albion the Branded Dragon, The Fallen & The Virtuous, Ame no Murakumo no Mitsurugi, Ame no Habakiri no Mitsurugi
  - banished (1): Mitsurugi Ritual
  - deck 27, extra deck 14
- **Our AI #2**
  - hand (1): Snake-Eyes Poplar
  - monsters: Snake-Eyes Flamberge Dragon (ATK), Promethean Princess, Bestower of Flames (ATK)
  - spells/traps: Fire King Sanctuary
  - GY (9): Called by the Grave, Triple Tactics Thrust, Original Sinful Spoils - Snake-Eye, Fire King Island, Fire King High Avatar Kirin, Fire King Avatar Barong, Snake-Eye Oak, Legendary Fire King Ponix, Harpie's Feather Duster
  - deck 29, extra deck 14

Plays:

- Our AI #1 passes  **(search: chose this over the network's first choice: activates Super Polymerization ("Fusion Summon"))**
- Our AI #1 activates The Fallen & The Virtuous ("Activate 1 of these effects")  **(search: chose this over the network's first choice: activates Super Polymerization ("Fusion Summon"))**
- Our AI #1 selects Ecclesia and the Dark Dragon  _[Select the card(s) to send to the GY]_  **(search: chose this over the network's first choice: selects The Dragon that Devours the Dogma)**
- Our AI #1 selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to destroy]_
- Our AI #1 passes  **(search: chose this over the network's first choice: activates Super Polymerization ("Fusion Summon"))**
- Our AI #2 uses the effect of Snake-Eyes Flamberge Dragon
- Our AI #2 selects Snake-Eye Oak  _[Select the card(s) to Special Summon]_
- Our AI #2 chooses position face-up Defense
- Our AI #2 chooses position face-up Defense
- Our AI #1 activates Super Polymerization ("Fusion Summon")
- Our AI #1 selects End of the World Ruler  _[Select the card(s) to discard]_
- Our AI #1 selects Promethean Princess, Bestower of Flames  _[Select the card(s) to use as Fusion Material]_
- Our AI #1 unselects Promethean Princess, Bestower of Flames  _[Select the card(s) to use as Fusion Material]_  **(search: chose this over the network's first choice: selects Snake-Eye Oak)**
- Our AI #1 selects Snake-Eye Oak  _[Select the card(s) to use as Fusion Material]_
- Our AI #1 selects Promethean Princess, Bestower of Flames  _[Select the card(s) to use as Fusion Material]_
- Our AI #1 chooses position face-up Attack
- Our AI #1 activates Mudragon of the Swamp ("Change attribute")
- Our AI #1 declares attribute
- Our AI #1 Normal Summons Mitsurugi no Mikoto, Kusanagi  **(search: chose this over the network's first choice: activates Ecclesia and the Dark Dragon ("Shuffle both those cards and this card into the Deck"))**
- Our AI #1 uses the effect of Mitsurugi no Mikoto, Kusanagi
- Our AI #1 selects Mitsurugi Ritual  _[Select the card(s) to add to your hand]_
- Our AI #1 activates Ecclesia and the Dark Dragon ("Shuffle both those cards and this card into the Deck")
- Our AI #1 selects Fire King Sanctuary  _[Select the card(s) to return to the Deck]_
- Our AI #1 activates Mitsurugi Ritual ("Activate 1 of these effects")
- Our AI #1 selects Ame no Murakumo no Mitsurugi  _[Select the card(s) to Special Summon]_
- Our AI #1 selects Mitsurugi no Mikoto, Aramasa  _[Select the card(s) to Tribute]_
- Our AI #1 chooses position face-up Attack
- Our AI #1 activates Mitsurugi no Mikoto, Aramasa ("Add 1 "Mitsurugi" monster from your Deck to your hand, except "Mitsurugi no Mikoto, Aramasa"")
- Our AI #1 activates Ame no Murakumo no Mitsurugi ("Destroy all monsters your opponent controls")
- Our AI #1 selects Mitsurugi no Miko, Wousu  _[Select the card(s) to add to your hand]_
- Our AI #1 enters the Battle Phase
- Our AI #1 attacks with Ame no Murakumo no Mitsurugi  **(search: chose this over the network's first choice: attacks with Mudragon of the Swamp)**
- Our AI #1 attacks with Mudragon of the Swamp
- Our AI #1 goes to Main Phase 2
- Our AI #1 ends the turn

## Turn 4: Our AI #2 (snake-eye)'s turn  (LP Our AI #1 300 / Our AI #2 2900)

At the start of the turn:

- **Our AI #1**
  - hand (2): End of the World Ruler, Mitsurugi no Miko, Wousu
  - monsters: Mudragon of the Swamp (ATK), Ame no Murakumo no Mitsurugi (ATK)
  - spells/traps: -
  - GY (11): Mitsurugi no Mikoto, Aramasa, Mitsurugi no Mikoto, Saji, The Fallen & The Virtuous, Ame no Murakumo no Mitsurugi, Ame no Habakiri no Mitsurugi, The Fallen & The Virtuous, End of the World Ruler, Super Polymerization, Mitsurugi no Mikoto, Aramasa, Mitsurugi no Mikoto, Kusanagi, Mitsurugi Ritual
  - banished (1): Mitsurugi Ritual
  - deck 25, extra deck 14
- **Our AI #2**
  - hand (2): Snake-Eyes Poplar, Diabellstar the Black Witch
  - monsters: -
  - spells/traps: -
  - GY (11): Called by the Grave, Triple Tactics Thrust, Original Sinful Spoils - Snake-Eye, Fire King Island, Fire King High Avatar Kirin, Fire King Avatar Barong, Harpie's Feather Duster, Snake-Eyes Flamberge Dragon, Snake-Eye Oak, Promethean Princess, Bestower of Flames, Legendary Fire King Ponix
  - deck 29, extra deck 14

Plays:

- Our AI #1 activates Mudragon of the Swamp ("Change attribute")  **(search: chose this over the network's first choice: passes)**
- Our AI #1 declares attribute
- Our AI #1 activates Ame no Murakumo no Mitsurugi ("Your opponent can discard 1 card, or that effect is negated")  **(search: chose this over the network's first choice: passes)**
- Our AI #2 chooses ("That effect is negated")  **(search: chose this over the network's first choice: chooses ("Discard 1 card"))**
- Our AI #2 activates Original Sinful Spoils - Snake-Eye ("Search 1 Level 1 FIRE monster")
- Our AI #2 selects Snake-Eyes Flamberge Dragon  _[Select the card(s) to return to the Deck]_
- Our AI #2 selects Snake-Eye Ash  _[Select the card(s) to add to your hand]_
- Our AI #2 Normal Summons Snake-Eye Ash
- Our AI #2 uses the effect of Snake-Eye Ash
- Our AI #2 selects Snake-Eye Ash  _[Select the card(s) to add to your hand]_
- Our AI #2 Special Summons Diabellstar the Black Witch
- Our AI #2 selects Snake-Eyes Poplar  _[Select the card(s) to send to the GY]_
- Our AI #2 chooses position face-up Attack
- Our AI #2 activates Diabellstar the Black Witch ("Set 1 "Sinful Spoils" Spell/Trap from your Deck")
- Our AI #2 activates Snake-Eyes Poplar ("Place 1 FIRE monster in the Spell & Trap Zone as a Continuous Spell")
- Our AI #2 selects Fire King High Avatar Kirin  _[Select the target(s) of the effect]_
- Our AI #2 selects Original Sinful Spoils - Snake-Eye  _[Select the card(s) to Set to the field]_
- Our AI #2 activates Snake-Eye Ash ("Special Summon 1 "Snake-Eye" from your hand or Deck")
- Our AI #2 selects Fire King High Avatar Kirin  _[Select the card(s) to send to the GY]_
- Our AI #2 chooses position face-up Attack
- Our AI #2 activates Snake-Eyes Flamberge Dragon ("Place 1 monster in its owner's Spell & Trap Zone as a Continuous Spell")
- Our AI #2 selects Ame no Murakumo no Mitsurugi  _[Select the target(s) of the effect]_
- Our AI #2 enters the Battle Phase
- Our AI #2 attacks with Snake-Eyes Flamberge Dragon
- Our AI #2 selects Mudragon of the Swamp  _[Select an attack target]_

## Game over: Our AI #2 (snake-eye) wins (LP before the final blow: Our AI #1 300 / Our AI #2 2900)
