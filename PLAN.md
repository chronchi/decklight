# ygo-zero — plan

Goal: give the system a `.ydk`; it learns to pilot that deck at (eventually) beyond-pro level,
and a *new* deck should need little or no extra training. Planned 2026-10-01.

## 0a. Next steps after the strategy review (2026-10-08 — user-agreed; supersedes the M-order below where they differ)

Why: since base_r3 (vs ygo-agent 42.3%) ~10 attempts were neutral or worse; they all trained, selected and judged
against base_r3 itself, and the RL league's ygo-agent games mostly used decks ygo-agent cannot read (bug).
Reviews: `docs/reviews/2026-10-08-{opus-5.5,muse-spark-1.3}.md`; decisions: NOTEBOOK 2026-10-08 03:20–04:40.

0. **E0 (laptop, $X):** deck-aware league (ygo-agent only on gauntlet-vs-gauntlet episodes, fixed extra opponents,
   per-opponent win rates) + paired ygo-agent gate on ~744 deals (sign test) as the primary acceptance metric.
1. **Search vs ygo-agent (E2, ~$X):** base_r3 + honest test-time search vs ygo-agent on the card's deals,
   paired against base_r3's raw games. Decides step 2.
2. **If search wins: large search-teacher round (teacher → student):** many games played with search, including vs
   ygo-agent and the mixed opponents; train on **soft targets** (full score distribution per searched decision,
   `--soft-targets gumbel`) plus game-outcome value labels; **200–400k labels** with the current $X credit (lighter
   search: k=4 × 1 rollout, only uncertain decisions; size after a ~$X throughput probe), BC from scratch or a
   large fine-tune, then repeat (expert iteration).
3. **RL with the fixed league** (gauntlet-focused, mixed opponents: ygo-agent on its decks, base_r3, older checkpoints,
   greedy) plus Kolibri **pass-rate curriculum** (§3.2.2.5: train on positions the agent sometimes solves) and
   **hinted self-distillation** (OPSD §3.2.1.4: on failures show the expert/search line in context, distil hinted →
   unhinted). If search does not beat ygo-agent, this becomes step 2.
   Gaps filled from the muse league-plan check (`docs/reviews/2026-10-08-league-plan-muse.md`; the plan above stands,
   these are additions):
   - **Opponent shares made explicit:** ygo-agent ≈ 25% of episodes, all gauntlet pairs (`--ga-episode-share 0.5
     --ga-ygo-share 0.5`); self 30%; ft-swv (base_r2) fixed 10%; greedy 5%; the rest base_r3 + snapshots (PFSP). Do
     not go ygo-agent-heavy: that overfits one fixed opponent and its 31 lists, against the unseen-deck goal.
   - **Pre-registered gate and kill rule:** the gate is the paired 744-deal ygo-agent sign test (≥ +4 pp, p < 0.05),
     with guards that new-card and holdout results stay ≥ −5 pp. Kill at it80 if in-run vs ygo-agent < base − 2 pp.
     h2h vs base_r3 is only a guard.
   - **Curriculum detail:** drop zero-variance groups (positions solved K/K or 0/K) from the PPO batch; their
     advantage is pure noise (the VRPO lesson). Positions are DAGS starts and dealt hands.
   - **Hinting, honestly named:** we have no hint-conditioned policy, so "hinted → unhinted" means fine-tuning on
     successes banked with the search line's help. Gate it on play-outs vs two opponents, never on field count (r1–r3
     learned resource-spending hacks).
   - **Extra option, kickstarting:** a decaying KL toward ygo-agent's own move probabilities on gauntlet states. It is
     the only teacher stronger than us; needs a new loss term.
   - **Disagreement (open):** muse would run the league alone first and add the curriculum and hinting in a second run,
     to avoid confounding. Our plan runs them together. The curriculum and hinting code is not built yet.
   - **Expectation:** +2–5 pp from RL at our budget; 50% vs ygo-agent by RL alone probably needs ~10× the budget.

**Outcome of step 2 (2026-10-08, NOTEBOOK 15:40):** the search-teacher students failed. 92.7k labels from live
searched games; soft c5/c10, min-max c5 and hard labels all scored ≤ base_r3 on the paired 744 deals (hard −3.6 pp,
p = 0.08). The search's gain is per-game lookahead over near-ties, not a pattern the policy absorbs. **Decision**
(user, 2026-10-08: "Search at play + RL (Recommended)"):
- base_r3 + honest search counts as our player from now on (57.1% vs ygo-agent, E2);
- step 3 RL goes ahead (league + curriculum; the hint term, which is also search copying, is a later ablation);
- RL checkpoints are gated both raw and with search;
- the DAGS starts are rebuilt from train decks only (`starts-train.jsonl`).

**Outcome of step 3 (2026-10-09, NOTEBOOK 06:40–07:20):** rl5 (league + curriculum) and rl6 (+ gated hint bank)
both card at 39–40% vs ygo-agent (base_r3 44.0%), n.s. vs base_r3, with a late drift away from ygo-agent. Fitting the
banked winning lines directly memorises them (trained hard starts 0.10 → 0.58) with zero transfer to held-out hard
starts. Proposed (awaiting the user): the product is base_r3 + search; spend on the search, not on policy training.

**Plan after step 3 (2026-10-09, user: "Ok I approve except the scale up, for now let's focus on everything else";
review `docs/reviews/2026-10-09-search-plan-opus.md`, NOTEBOOK 08:10):** the product is base_r3 + honest search.
- P0 ($X): `search_match --opp ygoagent` reproduces the card's ygo-agent games move for move (≥ 99% of the 744 deals;
  the two harnesses disagree at ygo-agent chain windows today); per-sample rollout scores logged per searched decision.
- P1 (~$X): E2 re-run (raw + search) on the old 744 deals vs the fixed ygo-agent = the new reference.
- P2 (~$X), each paired with P1's search arm: own-turn-1 search on going-first deals; override only on a clear
  margin; k2s2 (budget-curve calibration with the per-sample logs).
- Value lever (~$X): ~~a value trained on ≥ 50k raw-game outcomes~~ → first rl6's privileged critic as the honest
  search's leaf value (it reads the sampled world's hidden cards; init value_ppo3), as arm `hpriv` in P1, paired vs hppo
  (NOTEBOOK 09:15: value_ppo3 already came from far more games than 50k; an offline MSE gate on real games would hand the
  privileged critic the true hidden cards). The 50k-raw-games value stays a fallback if hpriv ties.
- ExIt pilot (~$X): ~20k clean (margin ≥ 2 SE), decorrelated labels, own turn 2 going second + opponent-turn-1
  windows, MPO-style loss + KL anchor, students on 25/50/100%; pass = held-out agreement rises with data, raw going-second
  ≥ +2 pp, held-out CE not rising from epoch 1; else the policy route is closed. Scale-up: not approved for now.
- P3–P4 (~$X): holdout/new-card guards, final card of the chosen config on fresh deals + single-game latency;
  the config is picked by a pre-registered rule (largest paired gain over P1 with p < 0.1), accepted if ≥ +3 pp, guards
  ≥ −3 pp.

**Hidden-information bound (2026-10-09/10, user: "Can you write step 2 into a plan and implement it and start now?";
review `docs/reviews/2026-10-09-history-arch-opus.md` Stage 1, NOTEBOOK 2026-10-10 00:55).** Question: is hidden-card
knowledge a real lever against ygo-agent, and does history predict hidden cards? Before any LSTM / history-aware model.
- (a) Oracle arm (~$X): hppo's search with the true hidden state (`--honest 0`), 744 deals vs ygo-agent, paired with
  P1 hppo. Gate: oracle − honest ≥ +3 pp (p < 0.1) → hidden information is worth working on for the search.
- (b) Belief head (~$X): predict "card c is in the opponent's hidden hand" for each distinct code of the opponent's
  list, from base_r3's trunk + remembered cards (`known`) + long history, trained on replays (labels = `priv_code`).
  Baseline = hypergeometric over the unseen part of the list (what the determinizer samples from today). Gate: belief
  NLL ≤ 0.9 × baseline on held-out games (≤ 0.95 on held-out decks) → history is informative about hidden cards.
  A no-history variant isolates history's share.
- Both fail → history/belief is dead for the search; no LSTM. Both pass → belief-weighted determinization (~$X).
  Oracle passes, belief fails → the lever is real but needs a different inference route (e.g. more worlds).

**Stop:** self-imitation rounds, long PPO from base_r3, distilling the search into the policy (argmax or soft), field/critic/val-agreement
proxies for selection, head-to-head vs base_r3 as the gate, infra work before a signal.

## 0. Current strategy (2026-10-05, ADR 0006 — read this first; §1-5 below are the original plan and background)

**One loop:** policy → **search teacher** (our policy + engine look-ahead, hidden cards determinized) → better
decisions → **distill** (BC) → **RL** fine-tune (PPO, KL to BC; league: snapshots + BC + ygo-agent) → stronger policy
→ repeat (expert iteration). ygo-agent = one-time bootstrap data (its 31 lists) + fixed opponent.

**One yardstick** (report card per checkpoint): vs ygo-agent on its 31 lists (248 games) · vs previous best on pool
train + holdout decks · turn-1 combo test (greedy + best-of-K) · per-card take rates (interruptions, payoffs) and
cut rate. Never self-play win rate.

**Milestones:** M0 report card on the BC candidates → pick one base · M1 RL v0 + ygo-agent teacher BC · M2 turn-1
search teacher + distill · M3 response-window (interrupt fire/hold) teacher + distill · M4 RL + league + periodic
ExIt (exit: > 50% vs ygo-agent, no holdout regression) · M5 test-time search.

**Stopped/paused:** more gold LLM games (saturated), memory/LSTM/bigger-model sweeps, card-knowledge runs, muse
DAgger, hand-variant expansions, old datagen driver. Code stays; revisit only if the report card points there.

## 1. What we have, and why it didn't generalize

| Project | What it is | Useful parts | Why it stalled |
|---|---|---|---|
| `ygo-agent` (upstream, last commit 2024-08) | envpool C++ env (`ygoenv`) + JAX PPO/LSTM agent | Fast env (thousands of SPS), obs/action encoding (`docs/`), released ckpt `0546_*` | Card ID → learned embedding table (13.4k ids); 2024 card pool; `mcts/alphazero` is empty — no search |
| `ygorl` (your fork) | ygo-agent + league (PFSP, exploiters) + deck specialization + mpnet→Voyage aligned embeddings | vast.ai scripts, league code, training log | Win rate vs. own pool stuck at ~0.52 — **that number can't measure strength** (it's ~0.5 by construction). No fixed external benchmark. Embeddings are a ridge-projected text vector, not effect semantics. Sparse reward over ~80-step games with GAE. |
| `ygosolver` | MCTS combo solver (ocgcore via ctypes) | Phase 1–2 only (DB, deck parser, engine bridge) | Unfinished |
| `ygo-architect` | LLM → WindBot C# executors | Lua-script → `CardEffect` schema ingestion (Step 1), combo-guide → combo tree (Step 3), ocgcore ctypes with all the binary-format gotchas solved, WindBot duel runner | Hand-written bots cap out at the LLM's understanding; no learning |

My read on the generalization problem, in order of importance:
1. **The card representation is per-ID.** The policy memorizes "card 14558127 is good here" instead of "a
   hand trap that negates a search is good here". An unseen card has no useful features.
2. **There is no external yardstick.** Self-play win rates hovered near 50%, so we could never tell real progress from churn.
3. **Exploration is hopeless from scratch.** A 10-card combo with sparse reward is basically never
   found by PPO; the agent learns "good-enough" lines and stops.
4. **There is no search.** AlphaGo-level play comes from policy + search, not policy alone. Going-first turn 1
   is *nearly perfect-information solitaire*: you know your hand, and only the opponent's hand traps are hidden. That's
   very searchable.

## 2. Target architecture

```
 card knowledge (offline, once per card; see Phase 2):
   Lua script ─► per-effect mechanics (timing/type/cost/target/OPT)   [exact, no LLM]
   card text  ─► split on PSCT ":" ";" ─► segment embeddings          [local embedder]
   LLM (free model) ─► fixed-label role tags                           [validated vs Lua]
   ═► set of effect tokens per card (no per-ID learned table)
 edo-ocgcore gym env (C++) ─► obs ─► Transformer over cards + action history (+LSTM) ─► policy / value / Q
                                         │
 Training data, cheapest → most expensive:
   a. Verified synthetic demos: LLM proposes lines from combo guides, engine executes and
      verifies them, and only lines that reach the stated end board are kept.
   b. Search-generated demos: turn-1 combo search (MCTS/beam in the engine) → expert iteration
   c. Behavior cloning on a + b (+ ygo-agent ckpt rollouts, + WindBot executors)
   d. RL fine-tune: PPO w/ KL-to-BC anchor, Q-critic/expected-SARSA advantages (VRPO),
      episodes started from mid-game states sampled from demos (DAGS)
 Inference: policy + test-time search / test-time RL on the current subgame
```

Relevant 2026 work this draws on:
- Test-time RL in imperfect-information games (Kubicek, Lisý, Sandholm, arXiv 2608.30635). Regularized
  policy-gradient fine-tuning at decision time, with guarantees against degrading the blueprint. This is the
  "learning while playing" idea, and it maps directly onto our PPO stack.
- GAE falls short in imperfect-info self-play (arXiv 2605.19235). It proposes VRPO: a Q-critic plus expected-SARSA(λ)
  in place of GAE, and it's PPO-compatible. This is a drop-in change to `switch.py`'s GAE.
- Data-Augmented Game Starts (arXiv 2605.14379). Start RL episodes from states drawn from demonstrations, and add a
  "started-from-augmented-state" flag to remove the bias that introduces.
- Cardsformer / MTG generalized card representations (text-grounded card features → unseen cards).

## 3. Phases (each has an exit criterion; GPU money is only spent from Phase 4 on)

### Phase 0 — Infra (laptop, $X)
- **Engine = EDOPro's stack**, not ygoenv's 2024 Fluorohydride core:
  `edo9300/ygopro-core` (the `edopro/ocgcore` submodule; ygo-architect already builds it and documents
  every binary-format gotcha in its CLAUDE.md) + ProjectIgnis CardScripts + BabelCDB = current card pool,
  the same rules EDOPro players use. The EDOPro GUI itself is only a reference: `gframe/duelclient.cpp`
  (message handling), `gframe/replay.cpp` (.yrpX format).
- **Gym env = port of ygoenv's C++ to the edo core API** (pybind11 + envpool batching). Keep ygoenv's
  observation/action featurization (`ygo-agent/docs/`); rewrite message parsing/responses for edo
  formats. A Python/ctypes env (ygo-architect) is the correctness oracle for the C++ port, but too slow for RL.
- Side benefits of EDOPro compat: WindBot opponents for free (Phase 1), and **.yrpX replays** from
  real players → human demonstration data, plus replay-based state cloning uses the same machinery.
- Repo `ygo-zero/`, layout per `docs/adr/0001`. Build on the laptop; measure CPU SPS (target: ≥ ygoenv's ~thousands/s per box).
- **Engine features we need, which ygoenv doesn't have:**
  - *state clone* via deterministic replay (seed + response log → rebuild duel); needed for search.
  - *puzzle start*: build a duel from an arbitrary hand/board (ocgcore supports placing cards before
    start; ygo-architect's scenario YAML already models this). Needed for DAGS and tests.
  - (edo core gives us the current card pool, so the 2024-pool problem goes away.)
- **vast setup script**: stock PyTorch image → `uv sync` builds core + extension on the instance (minutes), so no
  custom image is needed.
- **`vast` launcher** (Python SDK/CLI): search offer under a $/h cap → create → rsync code →
  run → pull results to the laptop → **always destroy** (storage is billed even when stopped). Hard
  wall-clock limit per job, plus a check of remaining credit before each launch.
- Exit: one end-to-end dummy job on the cheapest GPU for under $X, with artifacts landing on the laptop.

### Phase 1 — Yardstick (laptop CPU, $X)
- Fixed gauntlet: random, ygo-agent `0546` ckpt, best ygorl generalist, (later) WindBot
  executors. Report Elo/TrueSkill with confidence intervals, and never use self-pool win rate.
- Deck-specific metrics: **turn-1 end-board score** (does it reach the combo guide's end board?
  how often, from how many 5-card hands?), plus win rate going first and going second.
- **Held-out deck protocol:** train on N decks, test on unseen decks. This is how we measure "generalizable".
- Exit: one command → report card for any checkpoint × any deck.

### Phase 2 — Card knowledge (laptop CPU + free LLM, $X)
No LLM rewrite in the main path: we embed the official text directly. The text's timing
(PSCT: `condition : cost ; effect`, "if" vs "when", "and" / "then" / "also", Quick Effect, OPT) is
precisely what generic sentence embedders blur, so we represent it **explicitly**, with the Lua script
as ground truth:
1. **Per-effect mechanics from Lua** (deterministic, exact): one record per `Effect.CreateEffect` with
   type (IGNITION / TRIGGER_O|F / QUICK_O|F / ACTIVATE / continuous), event code (`EVENT_CHAINING`,
   `EVENT_SUMMON_SUCCESS`, …), `EFFECT_FLAG_DELAY` ("if": can't miss timing), `EFFECT_FLAG_CARD_TARGET`,
   range, count limit (HOPT id vs soft), category bits, has condition/cost/target. These map one-to-one to
   PSCT: before the colon = `SetCode`+`SetCondition`, colon→semicolon = `SetCost`, after = `SetTarget`/`SetOperation`.
2. **Segmented text embeddings**: split each effect sentence on PSCT punctuation into
   trigger/condition, cost and resolution; embed each segment plus the whole effect with a local embedding model.
   Effects are aligned to Lua records via `SetDescription(aux.Stringid(id,n))` ↔ cdb `str1..16` + order.
3. **Fixed-label tags** (role, what it interrupts, going first/second value) via LongCat-2.5-Preview-Free
   (free) with a fixed answer set; validated against layer 1 where both exist. Jev stays an option later.
- **Model input** = a *set of effect tokens per card* (layers 1 + 2) + card stats + tags, so the policy
  attends at the effect level and an unseen card is just new effect tokens. There is no per-ID table.
- Later option: fine-tune the text encoder contrastively to predict the layer-1 mechanics from text,
  giving an encoder that actually understands PSCT timing.
- Exit: `card_tokens(code)` works for any card; nearest-neighbour sanity checks pass; tags agree with Lua.

### Phase 3 — Synthetic and search demos (laptop CPU, $X vast)
- **Verified LLM lines:** combo guide + decklist → LLM proposes action sequences → executed in the engine
  → keep only lines that execute and reach the claimed board. The engine acts as the verifier, so we get
  rejection-sampled, ground-truth-correct data.
- **Turn-1 combo search:** MCTS/beam over the engine with clone-by-replay; opponent hand traps sampled
  from a meta prior (ygosolver's original idea). Output: best line per sampled hand.
- Recover your earlier "cloning works well" setup and fold it in here.
- Exit: ≥10k verified decision steps for 2–3 decks; the search finds the guide's end board on ≥X% of hands.

### Phase 3b — Data at all levels (ADR 0004)
- L1 combo lines (ADR 0003), L2 going second, L3 combos under interruption (replan from the interrupted state),
  L4 full multi-turn games with outcomes. Full-game generator: planner per main phase from the current state, engine-
  verified, rule-based responder, then the policy itself (DAgger). More decks (30–50) with fixed held-out decks.
- Exit: full games between pool decks with recorded winners; BC v2 trains on the level mix.
- 2026-10-03 (NOTEBOOK 17:15, 20:30): full games with two-sided interaction are the priority; hand-variant expansions
  stop (no unseen-deck gain). One curation pipeline (replay check, loops/stalls, judge tier, near-duplicates, holdout
  decontamination) before mixing; muse judges a sample, a distilled cheap scorer rates everything (Kolibri §2.3.2.3).

### Phase 4 — BC model (first GPU spend, ~$X)
- Model: ygo-agent-style transformer, but card input = frozen card vec + small adapter; no ID table.
- Train BC on Phase 3 data (+ ygo-agent ckpt rollouts as a "general play" prior).
- Exit: beats ygo-agent ckpt on the trained decks *and* holds up on a held-out deck.
- Mix chosen from trained candidates scored on holdout accuracy, interruption accuracy and the playtest (each capability
  its own term); scarce data repeated ≤ ~4×; last-K checkpoint merge (NOTEBOOK 2026-10-03 20:30).
- Always report turn-level play (paired combo test, best-of-K headroom, matches) next to accuracy.

### Phase 5 — RL on top (~$X)
- PPO from the BC init, KL penalty to BC (decayed), VRPO advantages, DAGS starts from demo states,
  small league (BC policy + snapshots + ygo-agent ckpt as fixed opponents).
- Expert iteration: periodically regenerate search demos with the current policy as prior.
- Exit: Elo gain on the gauntlet over BC, with no held-out regression.
- Curriculum by success rate (hard / trainable / solved hands and positions; sample the trainable ones), drop zero-
  variance groups, DAgger with the verifier/muse as the corrective "hint" on failures; narrated games every run to catch
  exploits (stalling, loops). Search at decision time (policy-guided verifier DFS / MCTS) is part of "superhuman".

### Phase 6 — Test-time play
- Turn-1: policy-guided search (fully available, mostly perfect info).
- Later turns: test-time regularized PG on the current subgame (Kubicek et al.) or determinized search.
- Serve through `ygoinf` → play against it in Neos / EDOPro.

## 3b. Deck pool (TCG, Sep 21 2026 F&L list)
Chosen for spread of play patterns, not only tier. Lists come from recent tournament tops (YGOPRODeck),
and the pool gets refreshed each banlist.
- **Meta / strong:** Elfnote, Ritual of Light & Darkness (ROLAD), Mitsurugi, Sky Striker, Dracotail /
  Branded Dracotail, Invoked, DoomZ, Power Patron, Artmage, Blitzclique, Kewl Tune (post-Rotary ban)
- **Rogue / gimmick:** X-Saber (Invoker back), Wind-Up (Zenmaity back, loops), Goblin Bikers (rank-3
  spam), Mikanko, Toon (Magnificent Maestros, check that the card DB has it)
- **Different axes:** stun/floodgate (Runick stun, TG stun), grind/GY (Gravekeeper), going-second
  board breakers (Kaiju), burn/FTK if available, plus older ygorl decks (Blue-Eyes, Fluffal, Lunalight)
- **Held out from all training** for the generalization test: 3 decks, one per bucket.
- The BabelCDB snapshot (2026-03-20) has all archetypes above except possibly the newest set; pull a fresh copy.

## 4. Budget ($X vast credit)
- A cheap 1×3090/4090 runs ~$X/h → ~15–25 GPU-hours total. Env sim is CPU-bound, so filter offers
  for **many vCPUs** (≥32), not big GPUs. Use interruptible offers + frequent checkpoint pulls to the laptop.
- Everything through Phase 3 runs on the laptop for $X GPU. No league runs of 6–8 hours like in ygorl.
- Every job has a time cap and auto-destroy; no stopped instances are left behind.

## 4b. Model / API providers (OpenCode Go subscription, key configured)
| Need | Model | Why |
|---|---|---|
| Hard reasoning / synthetic data (combo-line proposals, analysis) | `opencode/muse-spark-1.3-contributor-free` | large reasoning-request allowance |
| Bulk cheap generation | `opencode-go/deepseek-v4.1-flash` | more requests per window than v4-flash |
| Classification / tagging | `opencode-go/longcat-2.5-preview-free` | free; preview models can disappear, so cache every answer |
| Later, if classification volume explodes | Jev (TypeSafe, $X/M input) | no key yet |
| Text embeddings | local open embedding model (laptop CPU) | $X |
- Access: `opencode run --pure -m opencode-go/<model> "<prompt>"`, or the OpenAI-compatible
  `https://opencode.ai/zen/go/v1/chat/completions` (key in `~/.local/share/opencode/auth.json`).
- Throttle and cache every call by prompt hash; the Go terms target coding-agent traffic.
- No LLM in the play loop.

## 5. Decisions taken (2026-10-01)
- All work lives in `ygo-zero/`. Engine = edo9300 ocgcore (EDOPro stack).
- Deck pool §3b approved. Hold-outs: Invoked (meta), Goblin Bikers (rogue), Modern Gravekeeper (grind axis).
- The user's earlier BC setup isn't accessible; we rebuild our own.
- Card text is embedded directly (no LLM rewrite); timing comes from Lua.
- **PyTorch** for all models. Machines: laptop does everything except GPU training; vast = GPU only,
  renting needs no confirmation within the $X cap; homelab not used (ADR 0001).
