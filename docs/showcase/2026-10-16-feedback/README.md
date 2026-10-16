# Feedback games (2026-10-16)

Four games of **ygo-zero playing itself**. Both seats run the current product, P (policy network) + search
(honest determinized search, k 4, 4 worlds, horizon 1) + value_p2 (value network); see `scripts/product.sh`. The
players never see each other's hands; the write-ups show both hands for spectators.

| game | decks (first player first) | deck status | result | turns | searched / changed by search |
|---|---|---|---|---|---|
| unseen-1 | Gold Pride vs Ragnaraika | **cards never seen in training** | Ragnaraika wins | 6 | 48 / 22 |
| unseen-2 | Dragunity vs Dinomorphia | **cards never seen in training** | Dinomorphia wins | 8 | 57 / 23 |
| seen-1 | Snake-Eye vs Tearlaments | training decks | Tearlaments wins | 2 | 31 / 11 |
| seen-2 | Sky Striker vs Dragon Link | training decks | Sky Striker wins | 7 | 96 / 53 |

- "Never seen" means the archetypes have zero cards in any training data (`decks/newcards`, ADR 0007). The model
  only reads those cards' text and stats.
- Each game has a Markdown write-up (`.md`, turn by turn, marking where search overrode the network) and an EDOPro
  replay (`.yrp`).
  - Put the `.yrp` in `ProjectIgnis/replay/` and open it from EDOPro's Replay menu.
  - Every replay was verified against EDOPro's own engine.
- Laptop CPU, one core per game: 5–11 min per game. Launcher: `scripts/launch/laptop-feedback-games.sh`, deal seed
  20261016.

## Our own read (before outside feedback)
- **unseen-1**: Ragnaraika plays sensibly: Hunting Dance at the right times, Mantis Monk searches, steady
  pressure. Gold Pride is passive: it never uses Start Your Engines, Triple Tactics Talent or Lightning Storm and
  ends turns 3 and 5 doing almost nothing.
- **unseen-2**: Dinomorphia comes back from 26 LP. On turn 7 it plays Frenzy → Kentregina (copy Frenzy) → Rexterm,
  plus a Solemn/Sonic stack against the Gae Dearg line, then wins. Dragunity does little until turn 7.
- **seen-1**: Tearlaments OTKs on turn 2 through Ash Blossom and Nibiru. Snake-Eye's turn 1 extends through
  Poplar/Oak/Ash/Flamberge, then ends **without any Extra Deck summon**, a weak end board.
- **seen-2**: a grind game. Shark Cannon and Impermanence answer the Dragon Link opening, the Sky Strikers recur, and
  Widow Anchor steals Levianeer for the win.

## Questions for viewers
Per game:
1. the worst play (turn, what, and what you would do);
2. the best play;
3. any play no human would make;
4. the strength you'd guess for each side: casual / locals / regionals / top-cut;
5. your own level and format (Master Duel, EDOPro, paper).

## Mitsurugi games (user: "play a couple mitsurugi decks and tell me the lines it played")
`GAMES="sample mitsurugi-meta2:snake-eye,sample tearlaments:mitsurugi-meta1" bash scripts/launch/laptop-feedback-games.sh mitsurugi-20261016`

| game | result | turns | searched / changed | wall |
|---|---|---|---|---|
| mitsurugi-1: Mitsurugi (meta2, first) vs Snake-Eye | Mitsurugi wins | 3 | 49 / 25 | 362 s |
| mitsurugi-2: Tearlaments (first) vs Mitsurugi (meta1) | Mitsurugi wins | 6 | 110 / 53 | 1047 s |

- **mitsurugi-1**: turn 1 bricks (Pre-Preparation is Ashed). On turn 2 Mitsurugi defends with Purulia, Ash, Droll
  (held until after Poplar's first search) and Dominus Impulse. On turn 3 it OTKs:
  Ragged Records → Fallen & Virtuous on Temple → Habakiri pulls Saji → Ritual → Murakumo wipes the board → a second
  Ritual tributing Aramasa → Kusanagi → three attacks.
- **mitsurugi-2**: Mitsurugi breaks a full Tearlaments board (Rulkallos, Kitkallos, Kelbek, Merrli) through Fuwalos,
  Mudora, Ash, the Rulkallos negate and Kaleido-Heart. Line: Prayers (tribute Kusanagi) → Aramasa → Habakiri →
  Ritual → Murakumo wipe → Mirror re-ritual → three attacks. Turn 4: Dyna Mondo, then Gorgon, then Prayers with
  both effects. Turn 6: Dharc steals Kitkallos for lethal.
