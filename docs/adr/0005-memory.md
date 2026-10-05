# ADR 0005 — Policy memory: longer salient history, card-knowledge memory, recurrence (measured, not assumed)

Status: **accepted**, 2026-10-05 (user requirement: "past cards encode a lot").

## Context
Obs v1 (docs/obs_spec.md) gives the policy the current public board plus the **last 32 decisions**, passes and zone
picks included, so one busy turn (30-100 decisions) already fills it. Cards revealed and then hidden again (a searched
card shown, then held; a card bounced to the hand) are **not remembered**. Expert play depends on that information:
what the opponent searched or still holds, which once-per-turn effects are spent, what was used earlier in the duel.
The current model (PolicyNet v1, ~4M params, 4-layer transformer over board + chain + history tokens) has no value head
and no recurrence. Everything is stored as replays, so a new observation spec costs no data: shards are rebuilt from
the records.

## Decision
Memory is an experiment axis, like data size (gold-game saturation curve) and model size (mix vs mix-l, ~4M vs ~12M):
1. **Salient history, length H ∈ {32, 128, 256, 512}** (obs v2). The env records decisions as now but the history
   skips non-informative steps: PASS answers, zone picks (MSG_SELECT_PLACE / DISFIELD), and (already) auto-taken single
   options. Obs carries up to HISTORY_MAX = 512 entries (most recent first). Shards store history ragged
   (flat rows + lengths) so short histories cost nothing; the trainer pads each batch to the model's H (`--hist H`,
   the H most recent entries). v1 (32, unfiltered) stays the reference point.
2. **Card-knowledge memory** (obs v2, card rows): the env follows informational core messages (moves, confirms,
   hand/deck shuffles, draws) and keeps, per player, what each player knows about the other's hidden cards (hand
   contents first). A hidden opponent hand row whose identity is known is filled with its code and flagged
   `remembered`; own hand rows get a `revealed to opponent` flag. On/off switch in the model (`--no-known`) to measure it.
3. **Recurrence**: a variant with an LSTM/GRU over each seat's sequence of decision encodings (ygo-agent uses an
   LSTM), trained on per-game sequences (truncated BPTT). Built after 1-2 so it is compared against the best
   non-recurrent memory, i.e. does recurrence add anything once history and card knowledge are explicit?
4. **Measurement**: the same gold BC job (`scripts/vast_bc_gold.sh`) with the memory variant as a parameter: gold
   agreement on held-out games (seen / unseen decks), old-data accuracy, the playtest (head-to-head vs base, vs
   ygo-agent, vs greedy/random, combo test, stuck rate) and the muse review of self-play games (errors / 100 decisions).
   Expect long-memory gains on mid/late-game decisions; report accuracy by turn bucket (turn 1, 2-3, 4+).
5. **Cost guard**: attention over 1 + 160 + 16 + H tokens grows quickly with H (H=512 ≈ 3x tokens of v1). If H ≥ 256 is
   too slow to train on the cheap GPUs, compress history with a small latent cross-attention (Perceiver-style, e.g.
   32 latents) and compare compressed vs raw at H=128.

## Consequences
Obs spec v2, C++ env + Python reference env change together (lockstep test must stay at 0 mismatches); every
consumer of `history` (model, ygo-agent adapter, policies) is checked. Old checkpoints stay usable only with v1 obs:
keep the v1 path until v2 wins on the measurements. Value head is a separate, later addition (needed for RL/search).
