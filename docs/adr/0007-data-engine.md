# ADR 0007 — The data engine: labelled decisions that do not depend on ygo-agent

Date: 2026-10-05. Status: proposed (user: "I don't want to only rely on ygo-agent as distillation … new cards come
and then what … basically we need labelled data"). Extends ADR 0006 (the loop and the report card stay).

## Context
- What worked: 619k ygo-agent decisions doubled our win rate vs ygo-agent (11.5% → 25%) and helped on unseen pool
  decks; more teacher share was better (not saturated yet). Imitation of a competent labeller at volume works.
- What does not scale: ygo-agent knows 864 cards and ~31 lists (2024). It cannot label new cards or decks, and our
  policy can never exceed it by copying it. LLM full games are slow and noisy (gold games saturated at 103).
- What we have that does scale: the engine runs every card that has a script (CardScripts, updated on release),
  our model reads card text (it generalizes to unseen decks: holdout agreement rose with teacher data), and search
  labels are card-agnostic: the turn-1 search beat the expert lines (6.1 vs 5.0 field) with no human input.

## Decision: one data engine, three parts — states × labellers × gates

**1. States (where decisions come from), cheapest first**
- *Self-play of the current policy* on all pool decks (and new decks on release): unlimited, natural states.
  ~4-5M decisions/h on a $X/h box (RL v0 rollouts ran at ~1300 decisions/s on 16 cores).
- *DAgger states*: our policy's games vs a stronger opponent (ygo-agent now, our own best later).
- *Muse scenarios*: puzzle starts for rare but decisive situations (hand-trap windows, lethal into backrow, board
  breaking). Enrichment, not bulk: their realism is the risk (ADR 0006 amendment).
- *ygo-agent self-play*: bootstrap only, its share decays as our labellers pass it.
- Human replays: ruled out (user decision 2026-10-05). State diversity has to come from deck variety,
  sampled self-play, the opponent league and muse scenarios.

**2. Labellers (what the right decision is), in order of how they scale**
- *Search over our own policy* (card-agnostic, improves with the policy — the core of the engine):
  - v1 now: one-turn rollouts + hand-written position score (turn-1 search, window search).
  - v2: **learned value**: a value head trained on self-play outcomes (every finished game labels all its states
    win/loss for free; millions per dollar). Rollouts to the end of the turn are then scored by the value net instead
    of the heuristic, so labels capture long-term effects (resource use, attrition) that a board count misses.
  - v3: determinized search at key decisions (sample the opponent's hidden cards from what is public, search each
    sample, vote): removes the perfect-information bias of v1/v2.
  Each iteration: better policy → better rollouts → better labels → better policy (expert iteration). This is how we
  pass ygo-agent instead of converging to it.
- *Outcome labels* (value targets): free from every self-play game, all decks.
- *ygo-agent labels* (DAgger + self-play): while it is stronger than us on its decks; its share drops when our
  search labels agree with it or beat it on the report card.
- *LLM knowledge* (muse) where text understanding is the bottleneck: new cards / archetypes on release day — combo
  lines from card text and guides (ADR 0003 pipeline, engine-verified) as the first labels for a new deck, scenario
  design, and adjudication where labellers disagree. Never an unverified label.

**3. Gates (keep a data source only if it earns it)**
- Every new mix is judged on the report card (ADR 0006): vs previous base, vs ygo-agent, holdout decks, combo test,
  take rates. A source that does not move it is dropped or down-weighted.
- **New-card test** (added to the report card): a *card-level holdout*: a few archetypes whose cards never appear in
  any training data (states, labels, scenarios). Win rate and agreement on them measure "new cards come, then what".
- Label quality probes: agreement between labellers on shared states (search vs ygo-agent on its decks, search v1 vs
  v2), and spot audits by muse on disagreements.

## Collapse risk (training on our own labels)
The engine grounds us: legality and win/loss are real, so this is not LLM-style collapse (degenerating on unchecked
self-generated text). The real failure modes are narrower:
- *No new information*: retraining on the policy's own argmax adds nothing and reinforces mistakes. Rule: train
  only on targets that search improved, or on real outcomes, never on raw policy output.
- *Goodhart on the score*: search maximizes the heuristic, not winning. Already seen: ft-search raised turn-1 boards
  (combo 2.38) but lost win rate vs ygo-agent (9.5%). Fix: a learned value calibrated on real outcomes, checked
  by predicted-vs-actual win rate per bucket.
- *Narrowing states*: self-play only visits lines our policy likes; the value never learns lines nobody plays. Fix:
  temperature sampling for states, all pool decks plus deck variants, muse scenarios, and a league of opponents
  (frozen past bases, ygo-agent, greedy), not only the latest self.
- *Forgetting / cycling*: a new base beats the last one but loses to an older one or to a different style. Fix: keep
  a fixed anchor share (ygo-agent and gold data) in every mix, and accept a new base only if it beats the previous
  base AND does not regress vs ygo-agent, holdout decks, frozen older bases or the card-level holdout. Roll back
  otherwise.
Warning signs logged per run: policy entropy, the gap between self-play win rate and report-card win rate, score
up with wins flat, value calibration error.

## Throughput and cost (measured)
- Search labels: 1200 turn-1 hands / 38k searched decisions in 21 min on a $X/h box ≈ 1M decisions per $X.
  Window search: ~5 s per game on the laptop with the stand-in policy (box numbers from t2-20261005-1616).
- ygo-agent games: 3003 games / 619k decisions for $X.
- Self-play states + outcomes: ~1300 decisions/s per 16-core box.
- Distillation: ~20-40 min of GPU per fine-tune ($X). Report card: $X for 4 checkpoints.
Data is cheap; the scarce resources are GPU-hours for distillation and our review time. Store everything as replays
(~1 KB/game + targets), so the 50 GB cap is not a constraint.

## Muse in parallel (no GPU, no waiting on results)
Muse is a labeller only through the engine: what it proposes is replayed and verified, or scored by search. Its jobs:
- *Scenario designer* (running now: 30 puzzle starts, 6 themes): grows into a library of situations across all pool
  decks; our search labels them.
- *New-deck bootstrap*: on a new card or archetype, muse writes the combo lines from card text (ADR 0003 pipeline),
  the engine verifies them, and they become the first labels before any self-play exists. Rehearse it now on the
  card-level holdout archetypes.
- *Adjudicator*: on states where search and ygo-agent disagree, muse explains which is right and why. This is an
  audit of the labellers, not a source of training labels.
- *Implementer*: mechanical code for this plan (self-play state generator, value-head plumbing, card-holdout split,
  league opponent sampling) under the delegation workflow.

## Order (each step gated by the report card)
1. Now: mix teacher + DAgger + window + search on the new base (ft-teach-h); scale ygo-agent games 3k → 13k to find
   where it saturates.
2. Value head v2: train on self-play outcomes (pool decks + gauntlet), check calibration (predicted vs actual win
   rate by bucket), then swap it into the window search score.
3. Self-play states on all pool decks labelled by search v2 (bulk), muse scenarios as enrichment; ygo-agent share
   reduced.
4. Card-level holdout in the report card.
5. Determinized search v3 at key decisions; then RL fine-tuning with the value head (ADR 0006 M4).
