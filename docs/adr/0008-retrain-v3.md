# ADR 0008 — Retrain from scratch on clean data, a v3 encoder and honest search labels

Date: 2026-10-11. Status: accepted. The user gave the go: "yeah, go ahead, keep working on it and ping me when you
have results and keep working in general". The full plan is in `docs/retrain/plan-v3.md`, which builds on plan-v2,
plan-v1 and the critic's review in `critique-v1.md`. The reasoning is in NOTEBOOK 2026-10-11 07:30, 08:00 and 21:20.

## Context
- The product is base_r3 + hppo search. On 1,488 fresh deals vs ygo-agent it scored 0.512, against 0.410 for the raw
  policy (NOTEBOOK 2026-10-11 00:10).
- Three audits showed what holds it back.
  - **Training data teaches bad habits.** LLM lines fire hand traps on their own turn 1. The verifier's inferred
    chain labels fire Mulcharmy, Ash and Maxx "C" with nothing to answer. sw-v labels activate at 60% of windows,
    where ygo-agent activates at 24%.
  - **The encoder cannot see some of what a decision needs.** It cannot tell which effect an option is (string,
    effect type, once-per-turn), whether an effect was already used this turn, or whether the Normal Summon is spent.
  - **Search overrides are noisy.** Only 22% of hppo's overrides had a paired gap of 2 SE or more. 38.7% of them
    "chose" an identical copy or an equivalent zone.

## Decision
Retrain everything from scratch, in stages with pre-registered gates (plan v3):
- **Search (stage S).** Use honest overrides: merge semantically identical candidates; override only on a paired,
  significant gap (z = 2, at least 0.05), with adaptive samples. The gates are non-inferiority to hppo, overrides at
  least halved, and less own-turn-1 waste.
- **Data (stage D).**
  - The verifier never tries unnamed chain activations (`YGO_STRICT_VERIFY=1`).
  - Inferred labels get weight 0.
  - gold_pass and inferred PASS get weight 0.
  - A line is dropped if it fires a hand trap on its own turn with nothing to answer. The hand-trap class comes from
    reference play, not a hand list (`scripts/audit/handtrap_class.py`).
  - Holdout and new-card decks are asserted out of training.
- **New data.**
  - Stage M: muse plans whole turns from real states, going second, mid-game and end-game. Each line is verified
    strictly and kept only if a play-out shows it is not worse than the policy's own turn.
  - Stage L: honest decisive window labels. A label is emitted only where the best candidate beats the runner-up
    significantly, and only if a replication audit shows a precision of at least 0.85.
- **Model (stage B).** PolicyNetV3 keeps the observations and adds:
  - an embedding of the effect each option names (string vector plus the Lua record of that effect);
  - attention to each card's effect sentences;
  - categorical globals;
  - activation counts per effect;
  - a Normal-Summon-used flag.

  The Lua records are the engine's card definitions, not play rules. The user: "we can do that since we are
  already using the ygopro core as the gym environment".
- **Training after B.** First a data decision on the old architecture (B1), then B-v3, then PPO with a KL anchor
  (P). After that come a value critic (V), the final card on fresh seeds (F), and a search → policy pilot (X).
  The user on X: "ideally ... the policy would be all we need so we dont need to ship policy + search".
- **Efficiency is a goal** (§E). Report $ per pp at every gate, use one recipe, and measure the update paths for new
  cards and decks. The user: "superhuman ygo performance at a budget too".

## Rejected
- **Obs v3**, a lockstep observation change for counters. Only 9 of 175 pool decks (5%) are counter-centric, under
  the 10% rule (NOTEBOOK 2026-10-11 21:20).
- **Hand-encoded play rules, masks or priors.** The user: "No I don't want to hand encode things". The principle
  audits only filter or measure data. The policy never sees a rule.
- **The critic's "sw-v on opponent turns only" ablation.** It would bring back labels that fire Maxx "C" at about 50%
  of windows; hw labels replace it.
- **A new or bigger text-embedding model.** Per-sentence vectors already reach 0.99 AUC on effect properties.

## Consequences
- Budget: about $X committed plus a $X reserve. Every vast job is capped, logged and destroyed.
- `make_policy(emb, run_args)` builds every network, so v1 and v3 checkpoints load everywhere. Run args carry `v3`,
  `desc_strings` and `desc_lua`.
- If B-v3 does not beat B-v1 on the same mix, P starts from B-v1 and the report says the encoder did not help.
