# Search plan (ADR 0006 M2/M3 → M4/M5), after the PPO plateau

_2026-10-06. User: "can we do 1 and 3 in parallel?" (1 = search, 3 = a long PPO run as a scale test, ≈ $X)._
Why now: PPO from base_r3 plateaus at ≈ 0.51 vs base_r3 with three recipes (NOTEBOOK 19:30). Round-2 search labels
(value v2, h1, argmax playouts) also saturated (NOTEBOOK 11:10): the value net predates the current policy and the
search was shallow.

## Principle (AlphaZero / expert iteration)
A search over the policy is only a useful teacher if it plays clearly better than the policy alone. Measure that
first and cheaply; distil only if the gap exists.

## Steps (each gated by a number)
- **S0 Gate: test-time search match** (`scripts/search_match.py`): base_r3 + search at its uncertain decisions vs raw
  base_r3, both seats, the report card's 8 pool decks.
  - Search: windows plus own-turn decisions with ≥ 2 options where the policy's top probability is < 0.9. It tries
    the k = 4 policy candidates with 4 sampled playouts each to the end of the turn, scored by a value.
  - Variants: value v2 vs the board score; horizon 1 vs 2.
  - Gate: S wins **≥ 58%** (≈ significant at n ≈ 450).
  - Also measures seconds per searched decision, which sizes S2.
- **S1 Value v3**: a value for the current policy. The cheapest source is the PPO value head. The next PPO run must
  keep it (today `vast_rl.sh` drops `last.pt` and `policy_state` strips `value_head.`).
  - Then compare v2 vs v3 inside the S0 harness: the better search value wins.
- **S2 Distil (ExIt round)**: search labels on base_r3's own games at the decisions S0 shows matter (window
  search with the S0 settings), fine-tune base_r3, report card vs base_r3.
- **S3 RL from the distilled policy** (M4), then **S4 test-time search** at play (M5) if S0's gap holds at play
  speed.

## Costs
S0 is CPU-heavy: clone-by-replay plus playouts; a many-core GPU box, ≈ $X. S2 is similar plus a fine-tune.

---

# Plan v2 (2026-10-06 22:40), after S0 passed: what to fix before distilling, and in what order

_User (2026-10-06): "I identified at least a few things that we can improve here … the actual value net and the values
you're using to select … you select one of the follow-up choices and but then you don't know what else to do … are we
using any kind of reward for this value net … alpha zero is actually relatively old now. So, what are the new
advances … plan, and then you can take the decision here." Claude decides; this section is the decision._

## What S0 really measured (facts from the code)
1. **The search peeks.** Branch clones replay the game with the same seed, so the playouts know the opponent's hand,
   both deck orders and future draws (`window.py` docstring: "the rollouts see the true hidden state"). ADR 0006
   said "hidden cards determinized"; that was never built. S0's 0.636 = an oracle search vs an honest policy. It is a
   valid *training* teacher (oracle guiding, below) but **not** usable at play time, and its labels contain
   decisions the student can't justify from what it sees.
2. **Only the root decision is improved.** Inside a playout every later decision (targets, follow-ups, the
   opponent's replies) is the sampled raw policy. Follow-ups are searched only when the real game reaches them *and*
   they are eligible. On the opponent's turn only "activate in a chain window" is eligible; the targets and choices
   after activating never are.
3. **The reward is the final win/loss, through the value net.** Value v2 predicts P(win) from the policy's
   observation and is trained on finished games' outcomes (±1). Look-ahead is 1–2 turns, then v2 estimates the rest.
   v2 is the weak link: 5k games of ygo-agent/DAgger play (not our current policy), sign accuracy 0.74 overall and
   0.59 on turn 1 (NOTEBOOK 2026-10-05 19:15).
4. **Labels are argmax picks** of 4 candidates with 4 samples each; round-2 argmax labels saturated (NOTEBOOK 11:10).

## Newer methods considered (what we take, what we don't)
- **Gumbel AlphaZero/MuZero** (Danihelka et al. 2022): with few simulations, sample k actions without replacement,
  share the budget by sequential halving, and distil the *improved policy* softmax(logits + σ(Q)), not the argmax.
  This guarantees policy improvement even at 4–16 simulations, which is our regime. **Take**: soft targets, budget
  spent on the contenders.
- **Oracle guiding / perfect-training, imperfect-execution** (Suphx 2020 Mahjong, PerfectDou 2022): a critic or
  teacher that sees hidden information trains a policy that doesn't. **Take** for the value net and the PPO critic (a
  privileged value head: lower-variance advantages, a likely cause of the PPO plateau). The oracle search stays a
  teacher only, alongside an honest one.
- **Determinized search** (PIMC, the standard for card games): re-sample the hidden cards consistently with what the
  player has seen, then search. **Take, if the engine allows it** (investigate first; engine work, Claude).
- **ReBeL (2020) / Student of Games (2023)**: sound search for imperfect information (public belief states plus
  CFR). **Not now**: heavy, and YGO's hidden state and action space are huge. Revisit if determinized search is
  exploited.
- **DeepNash R-NaD (Stratego 2022), DouZero (2021)**: model-free self-play at scale (no search) reached top level in
  imperfect-information games. **Kept in mind**: volume of on-policy self-play matters (rl3long's slow climb fits
  this). R-NaD replaces PPO only if the league stays exploitable.
- **MuZero-style learned dynamics**: not needed, we have the exact simulator.

## Order (each step gated; Claude decides go/no-go from the numbers)
0. **Close tonight's jobs** (no spend):
   - rl3long report card; it becomes the base if it significantly beats base_r3 without holdout regression.
   - The S0 variants: value v2 vs board score, 1 vs 2 turns ahead.
   - Log both.
1. **Value v3, on a GPU box (≈ $X)**:
   - Data: self-play games of the base, one honest head (the policy's observation) and one privileged head (plus the
     opponent's hand and both deck orders), outcome ±1.
   - Gate: honest-head sign accuracy > v2's 0.74 on held-out games, by turn (turn 1 especially).
   - It serves search (privileged head for the oracle teacher, honest head for honest search) and becomes the PPO
     critic.
2. **Honest search, on the laptop (engine only)**: can a clone re-sample the opponent's unseen cards (hand plus
   deck) and both players' unknown future draws while keeping the replayed prefix legal? Measure where the 14 s per
   search goes (prefix replay, playouts, forwards).
3. **S0b gate, on a many-core box (≈ $X)**, same harness as S0:
   - (a) honest search + v3 honest;
   - (b) oracle search + v3 privileged.
   - Search-side changes: extend eligibility to every opponent-turn decision of mine (targets, which card to
     negate); sequential halving.
   - The gap between (a) and (b) is how much S0 owed to peeking.
   - Gate: (a) ≥ 0.58 makes honest search the teacher and test-time search viable. Otherwise the oracle is the
     teacher (oracle guiding) and test-time search waits.
4. **S2 distil (≈ $X)**: Gumbel-style improved-policy soft targets from the winning teacher, on the base's own games,
   as a mix with KL to the base. Report card vs base.
5. **S3 PPO from the distilled policy with the privileged critic (v3)**, league opponents (self snapshots, base,
   ygo-agent). Report card.
6. **Loop 1–5** with the stronger policy (new self-play → value → search → distil → RL); test-time search (S4) at the
   end, honest search only.

Budget for 1–5 ≈ $X of the ≈ $X credit.

## Status 2026-10-07 02:45 (NOTEBOOK)
- Step 0: rl3long is not accepted (h2h 60.7% but new-card 45%, combo halved); base_r3 stays.
- Steps 2–3 done:
  - S0b: honest + value_ppo3 0.688 and oracle + value_ppo3 0.703, so **honest search is the teacher**.
  - value_ppo3 (the PPO critic) replaces v2 as the search value. Value v3's honest/privileged heads drop in
    priority: the PPO critic already serves.
- Next:
  - S2: distil honest-search labels (value_ppo3, h1) with soft targets.
  - Meanwhile ft-r3: distillation of the oracle sw3 labels, already paid for.
  - Wind-Up investigation.
  - `--det-fallback skip`.
