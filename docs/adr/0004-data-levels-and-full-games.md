# ADR 0004 — Demonstration data at all levels: from combo lines to full interactive games

Status: **accepted**, 2026-10-02 (agreed with the user).

## Context
ADR 0003 produces engine-verified turn-1 combo lines (L1): 57.6k BC decisions over 8 decks. That covers one slice of
the game. Real games are decided by going second, interaction (hand traps, negates, when to use your own), later turns,
the battle phase and resources, and only full games give outcomes (value labels). A policy cloned only on expert
trajectories also drifts into states the demonstrations never cover (covariate shift), and its errors compound over a
game.

## Decision
1. **Data levels, all kept and mixed in training** (each sample tagged with level, source and verification status):
   - L1: turn-1 combo lines (ADR 0003, plus hand-variant expansion).
   - L2: going second against a set-up board (break it, then build or attack).
   - L3: combos under interruption: the opponent uses hand traps or negates; the turn player replans from the interrupted state.
   - L4: full multi-turn games between two decks, with the winner recorded, which gives value labels.
2. **Full-game generator (L2–L4).** Two pool decks play a normal duel. Each main phase is planned from the *current
   state* by the planner (muse, or Claude; cheaper models only once gated on quality), verified by the engine search from
   that state, then executed. Response decisions (chain windows, attacks, prompts outside a plan) come first from a
   rule-based responder (hand traps at their trigger, attack into weaker or empty boards) with sampling for diversity,
   later from the policy itself. When a response breaks a plan, the turn player replans from the new state, which is
   the interaction data. Everything is replayable (config + choices + labels + actor + winner).
3. **Policy in the loop (DAgger / expert iteration).** Once BC plays legal games, it drives most decisions. The
   planner, search and verifier label the states it actually reaches (hard or uncertain spots first), and we retrain.
   This is cheaper per sample than planning everything and fixes covariate shift.
4. **Diversity over volume.** Grow the deck pool (30–50 archetypes across eras), keep a fixed set of held-out decks,
   and rebalance per deck and per level so easy decks do not dominate.
5. **Evaluation.** BC accuracy (val lines, held-out decks) is a proxy. The real metric is win rate against the fixed
   gauntlet, including held-out decks.

## Expectations
Zero-shot expert play on unseen decks is unlikely. The target is rules-correct, competent zero-shot play plus strong play
after a short self-play fine-tune on the new deck. Card identity is text-only (no ID table), which makes both possible.
