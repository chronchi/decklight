# Game 0: BestowedDragon vs BestowedDragon2

- Searching seat: 0 (BestowedDragon); went first: True
- Result: **player 0 wins** (BestowedDragon)
- 86 decisions, 11 searched, 5 overridden; labels: 0 inacc, 1 mistakes, 0 blunders; mean regret 3.19 win-pp (ACPL analogue)
- Evals are P(win | both sides continue like our policy P): relative to policy, not absolute.
- Full-curve evals (v_comb) from the value net at every decision of both seats.

## Turns

| turn | player | LP (own-opp) | hand | field | GY | banished | extra used | card_adv | eval p0 | tempo |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P0 (to move) | 8000-8000 | 5 | 0 | 0 | 0 | 0 | +0 | 58.9% | -2.6pp |
| 1 | P1 | 8000-8000 | 5 | 0 | 0 | 0 | 0 | +0 |  |  |
| 2 | P0 | 8000-8000 | 3 | 3 | 1 | 0 | 0 | +0 | 56.3% |  |
| 2 | P1 (to move) | 8000-8000 | 6 | 0 | 0 | 0 | 0 | +0 |  | -15.2pp |
| 3 | P0 (to move) | 6300-8000 | 4 | 0 | 4 | 0 | 0 | +1 | 71.5% |  |
| 3 | P1 | 8000-6300 | 0 | 3 | 3 | 0 | 0 | -1 |  |  |

## Labelled moves

### Turn 3 decision 59 (mistake, seat 0, own)

Played **activates Tenpai Dragon Paidra ("Synchro Summon using this card you control")** over search-best **attacks with Tenpai Dragon Chundra**: regret 12.7 win-pp (gap/SE 56.1).

- activates Tenpai Dragon Paidra ("Synchro Summon using this card you control") — 87.3% ± 0.2pp <-- played
- activates Tenpai Dragon Chundra ("Synchro Summon using this card you control") — 89.4% ± 3.7pp
- attacks with Tenpai Dragon Chundra — 100.0% ± 0.0pp <-- search-best
- attacks with Tenpai Dragon Paidra — 87.2% ± 12.8pp

