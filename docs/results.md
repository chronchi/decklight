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
**decklight = policy r1 (the Forge release) + value net value_p2 + honest search with the honest override (`--ho`).**
r1 is the earlier policy P with 16 archetype specialists distilled into it ([The Forge](#the-forge)).

| | vs ygo-agent | n | run |
|---|---|---|---|
| **decklight (r1 + value_p2 + search + ho)** | **0.624** [0.599, 0.648] | 1,488 | `full-r1-20261030-1609` |
| previous product (P + value_p2 + search + ho) | 0.623 [0.598, 0.647] | 1,488 | `full-p-ho-20261021-1204` |
| P + value_p2 + search, without the override rule | 0.629 [0.604, 0.653] | 1,488 | `value-p2-20261015-2044` |
| P alone (no search) | 0.487 [0.456, 0.518] | 992 | `guardfix-20261014-1704` |

- **Why r1 replaced P:** against ygo-agent it is level with P (paired +0.1 pp, SE 1.3), and on the Forge archetypes
  it is much stronger (+11.0 pp with search, below). Same value net and search settings; only the policy changed.

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
| r1 (Forge release) + search + value_p2 + override rule | 0.624 [0.599, 0.648] | `full-r1-20261030-1609` |

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
model) with one deck in half the seats. The runs below started from P, the product at the time. Pilot: 0.75 GPU-hour
per deck on an RTX 4060 Ti 16 GB (~66 iterations), then
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

## The Forge
Per-deck fine-tunes help one deck each. The Forge distills many of them back into the shared model, so a single
release plays every deck better. Pipeline (`scripts/vast_forge_round.sh`):
1. **Specialists:** one fine-tune per archetype (`finetune_deck.py`, ~1 GPU-hour), packed with `scripts/forge.py
   pack` (safetensors weights, deck lists, config, metrics).
2. **Verify:** a sequential paired test per pack against the current release on its own decks (opponent: the
   release's raw policy, held-out deals). Stops early when the answer is clear (3.3× fewer games than a fixed size).
3. **Distill:** behaviour cloning on the packs' own self-play games, pure KL to each pack's teacher on its own decks,
   plus anchor games of the release on 138 other decks (KL to itself) so nothing else drifts.
4. **Release card:** keep ≥ 50% of each specialist's gain; no loss on earlier archetypes, on a reserve of unrelated
   lists, or against ygo-agent.

Releases so far: **P** (the base model) → **r0** (P + 5 archetype specialists) → **r1** (r0 + 11 more), so r1 has
been distilled from 16 archetype specialists in all.

### Release r0 (5 archetypes, raw policy, r0 − P)
| archetype | vs P: r0 − P (95% CI) | specialist − P | kept | vs ygo-agent: r0 − P (95% CI) |
|---|---|---|---|---|
| branded (Light and Darkness ritual) | +6.6 (+4.5, +8.7) | +6.3 | 104% | +5.1 (+2.2, +8.0) |
| invoked | +17.6 (+14.8, +20.4) | +20.4 | 86% | +12.7 (+8.7, +16.7) |
| radiant-typhoon | +22.4 (+19.4, +25.4) | +29.2 | 77% | +19.7 (+15.6, +23.8) |
| sacred-beasts | +13.1 (+10.6, +15.5) | +15.8 | 83% | +17.9 (+13.9, +21.9) |
| atlantean-mermail | +19.1 (+16.4, +21.9) | +28.1 | 68% | +20.2 (+16.1, +24.2) |

- Left: against P's raw policy (seeds 29000+). Right: our lists of these archetypes against ygo-agent piloting its 31
  benchmark lists (pooled +15.1 (+13.4, +16.8), 3,720 games), so the gains are not an artifact of playing against P.
  (ygo-agent does not know most cards of these lists, so its absolute win rates flatter us; the paired differences
  hold.)
- Related lists (sharing an engine with the 5) gain +7.5 (+5.0, +9.9) vs P; unrelated lists +1.1 (−0.4, +2.6);
  ygo-agent's benchmark +1.1 (−0.1, +2.3).
- What changed in play: r0 commits to each deck's engine and boss lines, e.g. Radiant-typhoon summons and attacks with
  Fonix far more (selected in 25% → 66% of games), Invoked makes Hyperinvoked Aeon (35% → 66%), Sacred-beasts builds
  around Hamon.

### Release r1 (11 archetypes, raw policy, r1 − r0, opponent r0)
| archetype | r0 | r1 − r0 (95% CI) | specialist − r0 | kept |
|---|---|---|---|---|
| blitzclique | 0.737 | +7.9 (+5.7, +10.2) | +7.6 | 1.05 |
| elfnote | 0.570 | +12.3 (+9.6, +15.0) | +13.4 | 0.92 |
| fiendsmith-yummy | 0.393 | +33.7 (+31.0, +36.5) | +39.8 | 0.85 |
| ignister-maliss | 0.254 | +19.5 (+16.9, +22.2) | +28.9 | 0.68 |
| kewl-tune | 0.470 | +16.9 (+14.1, +19.7) | +16.4 | 1.03 |
| lunalight | 0.335 | +11.9 (+9.0, +14.7) | +12.7 | 0.93 |
| magistus-invoked-fairy-tail | 0.422 | +19.1 (+16.4, +21.9) | +18.9 | 1.01 |
| mitsurugi | 0.633 | +11.4 (+9.0, +13.8) | +22.5 | 0.51 |
| power-patron-artmage | 0.667 | +13.2 (+10.8, +15.7) | +13.9 | 0.95 |
| sky-striker | 0.534 | +16.3 (+13.6, +19.1) | +17.1 | 0.95 |
| trains | 0.553 | +13.8 (+11.0, +16.5) | +15.4 | 0.90 |
| **pooled** (17,048 games) | | **+16.0** | | **0.85** |

- Nothing else lost: earlier archetypes +0.2 (−0.7, +1.1) (7,680 games), a reserve of 16 unrelated lists
  +2.4 (+0.8, +4.0), ygo-agent's benchmark 0.0 (−1.2, +1.1) (5,952 games).
- A cheaper distillation (3× fewer teacher games) keeps 92% of r1's gain and is now the default. DAgger (relabelling
  the student's own games) added nothing.
- Contributors can train and submit packs: [contributing-packs.md](contributing-packs.md).

### r1 as the product (with search)
The product card plays r1 and P with the same search and value net on the same deals:

| comparison | r1 − P (95% CI) | n |
|---|---|---|
| vs ygo-agent, its 31 benchmark decks | +0.1 (−2.4, +2.6) | 1,488 |
| on the 11 Forge archetypes (opponent: r0's raw policy) | **+11.0 (+7.3, +14.7)** | 601 |
| same, on deck lists held out from distillation | +7.8 (+0.7, +15.0) | subset |

- Pre-registered gate: not worse against ygo-agent (lower 90% ≥ −2 pp, each seat ≥ −3), better on the Forge decks,
  and no new hand-trap waste (Maxx "C" on its own turn: 16.0% → 16.5% of games, within noise). It passed
  (runs `full-r1-20261030-1609`, `full-p-fresh-20261031-0608`).
- 10 of the 11 archetypes gain; Blitzclique is flat (0.0).

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
- Going first is weaker than going second (r1 product: first 0.593 vs second 0.655, `full-r1-20261030-1609`).
- The policy learned some hand-trap timing habits from ygo-agent's games (Maxx "C" in its own Draw/Standby phase in
  about 10% of games without search). The override rule removes most of the search-made ones.
- Long looping games are slow to search. A per-game search cap exists (`--max-searches`).
