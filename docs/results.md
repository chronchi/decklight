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
**decklight = policy P + value net value_p2 + honest search with the honest override (`--ho`).**

| | vs ygo-agent | n | run |
|---|---|---|---|
| **decklight (P + value_p2 + search + ho)** | **0.623** [0.598, 0.647] | 1,488 | `full-p-ho-20261021-1204` |
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

## Fine-tuning on one deck
`scripts/finetune_deck.py` continues the product's self-play PPO (policy and value head, KL-anchored to the base
model) with one deck in half the seats. Pilot: 0.75 GPU-hour per deck on an RTX 4060 Ti 16 GB (~66 iterations), then
`scripts/deck_card.py`: the deck against the other 30 benchmark decks piloted by ygo-agent, both seats, the same deals
for base and fine-tuned (paired).

| deck | mode | games per model | base | fine-tuned | difference (95% CI) |
|---|---|---|---|---|---|
| Blue-Eyes | raw policy | 480 | 0.277 | 0.315 | +3.8 pp (−0.3, +7.8) |
| Blue-Eyes | product (search) | 180 | 0.367 | 0.444 | +7.8 pp (+0.6, +14.9) |
| Branded | raw policy | 480 | 0.567 | 0.644 | +7.7 pp (+2.9, +12.5) |
| Branded | product (search) | 180 | 0.611 | 0.772 | +16.1 pp (+8.8, +23.4) |

- The gains are broad: the fine-tuned Branded model does better against 20 of 30 opponents and worse against 2.
- The visible change is that the fine-tuned model commits to its deck's engine more often. For example, it activates
  Branded Fusion where the base model normal-summoned.
- Two decks are not a distribution. Untested so far: how a fine-tuned model plays *other* decks (assume worse; use it
  only for its deck).

### Longer fine-tune: 2 hours on Blue-Eyes
Snapshots every 40 iterations, each played on separate validation deals (360 games per model, raw policy); the best
was then confirmed on the 480 deals above.

| iterations | 40 | 80 | 120 | 160 | 182 (2 h) |
|---|---|---|---|---|---|
| validation, difference vs base | +5.0 | +4.2 | +4.6 | +8.1 | +8.1 |

- Test (the 480 deals above): base 0.277 → 0.356, **+7.9 pp (+3.9, +11.9)**, against +3.8 pp after 0.75 h. The base
  model scored 0.277 again, on a different machine.
- The curve is flat for most of the run and then steps up, so 2 h is now the default (`--hours 2`); it may not be the
  top either. One deck, one run.

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
