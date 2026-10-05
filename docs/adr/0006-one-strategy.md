# ADR 0006 — One strategy: policy + search teacher, distilled and RL-tuned, judged by one report card

Date: 2026-10-05. Status: accepted (user: "consolidate all the plans into one cohesive strategy").
Supersedes the open-ended thread list in STATUS; PLAN §0 is the short form of this ADR.

## Context (what the last 5 days showed; NOTEBOOK 2026-10-05)
- BC imitation saturates: gold agreement 0.684 / 0.686 / 0.690 on 50 / 75 / 103 gold games; better imitation
  did not turn into wins (ft-gold vs base 52%, CI 46-59). More LLM gold games (~$X each, hours) won't move it.
- Model axes gave nothing: salient history H 32-512 and the 3x model (mix-l) are flat.
- Vs ygo-agent we lose ~94%, by attrition: BC never reaches the payoff cards, takes core engine plays half as often,
  and fires interruptions in 3% of legal windows vs 21-35%. Those are decisions imitation of mostly-pass labels does
  not teach; winning-driven methods (search, RL) do.
- Evaluation was broken: uncapped matches stalled every playtest of the 5 jobs (fixed: 3000-decision cap).

## Decision
One loop, one yardstick, one thread on the GPU at a time.

**The loop:** `policy → (search teacher) → better decisions → distill (BC) → RL fine-tune → stronger policy → …`
(expert iteration). The search teacher is our own policy plus engine look-ahead (clone by replay, hidden cards
determinized), so it works on any deck the engine runs, including unseen ones (the project goal). ygo-agent is a
one-time bootstrap (teacher games on its 31 lists) and a fixed opponent, not the long-term teacher.

**The yardstick (one report card per checkpoint, same command, capped):**
1. vs ygo-agent 0546 on its 31 lists (`decks/gauntlet`, 248 games) — the external bar;
2. vs the previous best checkpoint on pool train + holdout decks — progress and generalization;
3. turn-1 combo test (end board vs expert line, greedy and best-of-K) — the search teacher's first target;
4. behaviour: per-card take rate when legal (`offered.py`: interruptions, payoffs), cut/stuck rate.
Never self-play win rate. A step "worked" only if the report card says so.

## Milestones (each: exit criterion, then go/no-go)
- **M0 Yardstick** (now): report card = `playtest.py` (capped, 31-deck gauntlet) + `offered.py`; re-run it on the BC
  candidates (base, g103 ft-gold / ft-mix / mix) and pick **one base model**. Exit: complete report cards.
- **M1 Cheap boosts**: read RL v0 (PPO from BC, KL to BC); BC with ygo-agent teacher games as a minority mix.
  Exit: either moves the report card (ygo-agent win rate, interruption take rate) without holdout regression; the best
  becomes the base.
- **M2 Turn-1 search teacher**: going first, policy top-k candidates + short look-ahead in the engine, scored by
  end-of-turn board; distill its choices. Exit: teacher beats greedy on the combo test; distilled policy improves
  the combo test without losing matches.
- **M3 Response-window teacher**: at opponent-turn windows, fire-vs-hold decided by playing both out (value head
  from RL); distill. Exit: interruption take rates and vs-ygo-agent win rate go up.
- **M4 RL + league + periodic ExIt**: PPO from the M3 policy, opponents = self snapshots + frozen BC + ygo-agent,
  search demos regenerated with the current policy. Exit: > 50% vs ygo-agent on its lists, no holdout regression.
- **M5 Test-time search** (turn-1 search at play time, then determinized search later in the game).

## Paused or stopped (code stays; revisit only if the report card points at it)
- Gold LLM games: **stopped** (saturated). Muse stays as reviewer/auditor and for plumbing.
- Memory sweep (H), card-knowledge runs (`--known`), LSTM, bigger model: **paused** — no gain so far; revisit if
  report-card failures look memory- or capacity-shaped.
- DAgger with muse corrections, hand-variant expansions, the old datagen driver: **stopped**.
- Deeper card knowledge (Phase 2 effect tokens): **paused** while the frozen text embeddings are not the bottleneck
  (check holdout vs train gap on the report card).

## Budget (credit ≈ $X, floor $X since 2026-10-05 15:30; warn the user near $X)
M0-M1 ≈ $X (GPU evals + RL v0 + teacher BC); M2-M3 ≈ $X (search is CPU: many-core boxes, distill on GPU);
M4 the rest. Search and engine work are developed on the laptop (no ML there); every ML run is a capped vast job.

## Amendment 2026-10-05 18:50 — teachers for every turn, muse as scenario designer (user request)
User: "more teacher examples in diverse scenarios … many different turns and situations … when to interrupt a combo
… distill ygo-agent but also improve using muse … avoid running millions of games". Accepted; it widens M3:
- **DAgger labels** (`scripts/gen_dagger.py`): our policy plays ygo-agent on its 31 lists, ygo-agent labels every
  one of our decisions (all turns; the states we actually reach). Kinds `dagger` / `dagger_pass`.
- **Window search** (`src/ygo_zero/search/window.py`, `scripts/gen_window.py`): on recorded games (teacher, gold,
  DAgger), every opponent-turn chain window where we can activate is searched (PASS vs each activation, played out to
  the end of the turn by the policy for both players, position score = board diff + 0.5 hand diff + LP diff/2000),
  plus 10% of own-turn decisions from turn 2. Any deck. Kinds `window` / `window_own`. Records carry `targets`
  (searched option per decision) while the replay follows the recorded choices.
- **Muse as scenario designer** (next): muse writes mid-game situations as engine puzzle starts (both sides' cards,
  LP, turn), the engine validates them, the search teachers label them; muse also adjudicates states where the
  teachers disagree. Grounded labels, cents per scenario, no full LLM games (those stay stopped).
The report card (ADR 0006) is unchanged and decides which data mixes stay.
