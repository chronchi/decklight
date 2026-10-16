# Game 2: BestowedDragon vs BestowedDragon2

- Searching seat: 0 (BestowedDragon); went first: True
- Result: **player 0 wins** (BestowedDragon)
- 150 decisions, 33 searched, 14 overridden; labels: 0 inacc, 0 mistakes, 1 blunders; mean regret 3.09 win-pp (ACPL analogue)
- Evals are P(win | both sides continue like our policy P): relative to policy, not absolute.

## Turns

| turn | player | LP (own-opp) | hand | field | GY | banished | extra used | card_adv | eval p0 | tempo |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P0 (to move) | 8000-8000 | 5 | 0 | 0 | 0 | 0 | +0 |  |  |
| 1 | P1 | 8000-8000 | 5 | 0 | 0 | 0 | 0 | +0 |  |  |
| 2 | P0 | 8000-8000 | 1 | 3 | 3 | 0 | 1 | -1 | 48.4% |  |
| 2 | P1 (to move) | 8000-8000 | 5 | 0 | 1 | 0 | 0 | +1 |  | -17.0pp |
| 3 | P0 (to move) | 8000-8000 | 1 | 2 | 5 | 0 | 1 | +0 | 65.5% | +9.2pp |
| 3 | P1 | 8000-8000 | 1 | 2 | 3 | 0 | 0 | +0 |  |  |
| 4 | P0 | 8000-7200 | 0 | 3 | 7 | 0 | 1 | +1 | 74.7% |  |
| 4 | P1 (to move) | 7200-8000 | 2 | 0 | 5 | 0 | 0 | -1 |  | -23.6pp |
| 5 | P0 (to move) | 8000-7200 | 2 | 3 | 8 | 0 | 1 | +3 | 98.3% |  |
| 5 | P1 | 7200-8000 | 1 | 1 | 7 | 0 | 0 | -3 |  |  |

## Labelled moves

### Turn 4 decision 86 (blunder, seat 0, window)

Played **passes** over search-best **activates Hieratic Seal of the Heavenly Spheres**: regret 19.7 win-pp (gap/SE 6.6).

- passes — 55.0% ± 2.8pp <-- played
- activates Hieratic Seal of the Heavenly Spheres — 74.7% ± 3.9pp <-- search-best

