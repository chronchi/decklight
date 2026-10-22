# Results

Every number below comes from a paired evaluation on fixed deals. The run name in each row identifies the job that
produced it. Win rates are for our seat; 95% confidence intervals are in brackets.

## The benchmark
- **Opponent:** [ygo-agent](https://github.com/sbl1996/ygo-agent), an open-source reinforcement-learning Yu-Gi-Oh!
  agent (checkpoint `0546_26550M`), playing its own deck lists.
- **Deals:** 31 decks, each against its next 4, 6 seeds per pairing, both seats. That is 1,488 games per card on the
  same deals for every arm, paired by (decks, seed, seat).
- **Rules:** one game per deal (no side decking). The engine is EDOPro's core (edo9300 ygopro-core).
- ygo-agent is the only external opponent measured so far. **Nothing here shows human-level or superhuman play**;
  human feedback games are the next step.

## The product today
**Decklight = policy P + value net value_p2 + honest search with the honest override (`--ho`).**

| | vs ygo-agent | n | run |
|---|---|---|---|
| **Decklight (P + value_p2 + search + ho)** | **0.623** [0.598, 0.647] | 1,488 | `full-p-ho-20261021-1204` |
| same without the override rule | 0.629 [0.604, 0.653] | 1,488 | `value-p2-20261015-2044` |
| P alone (no search) | 0.487 [0.456, 0.518] | 992 | `guardfix-20261014-1704` |

- **The override rule:** search replaces the policy's move only when the gap is statistically significant.
  - Paired against the version without it: −0.6 pp (SE 1.2), so the same strength within noise.
  - Overrides drop from 54% to 10% of searched decisions.
  - Games where the bot fires Maxx "C" on its own turn drop from 23% to 16%.
- **What search adds:** on the same deals, P + search beats P alone by +11.9 pp [+8.7, +15.1]
  (`guardfix-20261014-1704`, with value_p).

## How it got here (vs ygo-agent, same benchmark)

| step | vs ygo-agent | run / note |
|---|---|---|
| first behaviour-cloning policy | ≈ 0.08 | 160 gauntlet games analysed |
| + ygo-agent's own games as teacher data | 0.250 | `rc-20261005-1636` |
| base_r3 (BC + PPO) | 0.410 | `final-20261010-1227` |
| base_r3 + honest search | 0.512 [0.486, 0.537] | `final-20261010-1227` |
| P (rebuilt BC + PPO), alone | 0.487 | `guardfix-20261014-1704` |
| P + search + value_p | 0.606 [0.575, 0.636] | `guardfix-20261014-1704` |
| P + search + value_p2 | 0.629 [0.604, 0.653] | `value-p2-20261015-2044` |
| P + search + value_p2 + override rule | 0.623 [0.598, 0.647] | `full-p-ho-20261021-1204` |

Rows before `guardfix` were measured with a loop-guard bug that cost our side about 4 pp. Relative comparisons within
that period are fair; absolute numbers are understated.

## Decks it never trained on
- **Holdout decks** (17 deck lists never trained on; their cards mostly seen) and **new-card decks** (6 archetypes with
  zero cards in any training data).
- Search vs the raw policy (raw vs raw would be 0.5), `final-20261010-1227`:
  - holdout 0.676 [0.647, 0.703] (n 1,080);
  - new-card 0.656 [0.628, 0.684] (n 1,080).
- P vs base_r3, raw policies, Stage P card:
  - holdout 0.528 [0.507, 0.549] (n 2,176);
  - new-card 0.514 [0.492, 0.536] (n 2,058).

## Things that did not help
- **A bigger architecture (v3: longer history, categorical game-phase inputs) + the same RL**, run `stage-p3-…`:
  - −1.6 pp vs P alone;
  - −12 pp with search (`full-p3-20261021-0947`). Search added ~16 pp to P but ~2 pp to v3.
- **Continuing PPO from P:** no gain past the gate.
- **Distilling search decisions back into the policy** (student runs): neutral or worse.
- **A league with a privileged critic** (rl4) and **DAGS start states:** no gain over the base.
- **Searching turn 1 when going first:** +1.2 pp [−2.8, +5.2], not shown.
- **Single-copy tech-card swaps:** 15 real swaps, all within ±1.5 pp (controls pass).

## Known weaknesses
- Going first is weaker than going second (e.g. first 0.554 vs second 0.652 on the first 724 deals of
  `full-p-ho-20261021-1204`).
- The policy learned some hand-trap timing habits from ygo-agent's games (Maxx "C" in its own Draw/Standby phase in
  about 10% of games without search). The override rule removes most of the search-made ones.
- Long looping games are slow to search. A per-game search cap exists (`--max-searches`).
