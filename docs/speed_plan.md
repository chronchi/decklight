# Speed plan: cheaper and faster RL rounds (2026-10-06)

User (2026-10-06): once ppo1 finishes, and before more rounds, find the bottlenecks and make the whole process faster and
more efficient. This plan ranks the work by time saved per round against effort. Every number below is measured, with
its source given.

## Where a round's time goes today (ppo1 + one report card)

| Stage | Time | Source |
|---|---|---|
| Host acquisition | 2–14 min; 16 failed hosts in 48 jobs | launcher logs (avg ready 217 s; ppo1 ≈ 20 min, 2 bad hosts) |
| Setup (uv sync + torch from PyPI + core/ext build) | 1–3.5 min | `runs/*/torch.txt` "deps install" |
| Smoke stage (3 iterations + 2 evals) | 6.6 min | ppo1 smoke.txt |
| Training iteration | 13 s: rollout 3.5–4 + update 9 | ppo1 train.txt |
| In-run eval (vs BC 224 + ygo-agent 248 games) | ≈ 126 s every 40 iterations, about 20% of training wall time | ppo1 train.txt |
| Report card job for an existing checkpoint | 33 min, of which 15 min build shards and run a 0-epoch train before the 5–10 min of games | r2-zero timing.txt |
| Eval noise | n = 224 gives ±6.5 pp, so most evals cannot tell a 55% model from 50% | ppo1 evals |

The GPU is the training bottleneck: 80–100% busy, while load is 5 of 24–32 cores, so the CPU is idle. Eval and
report-card play still use the synchronous `play_match`, so they still pay the barrier on slow `fusion_check` steps.

## Plan (in order)

0. **Measure first** (no new spend):
   - wall-clock accounting of ppo1 from its logs;
   - `torch.profiler` over 20 update minibatches, added to the next job as a 2-minute stage. It tells P3 where the
     9 s go (attention, the trim/pad copies, the optimizer).
1. **One job per round** (muse, plumbing): train → pick checkpoint → report card on the same box.
   - No second host acquisition.
   - Card a given checkpoint directly, without shards or the 0-epoch train.
   - Saves ≈ 35–50 min and ≈ $X per round.
2. **Async, concurrent eval with more games** (muse + review):
   - Port `play_match` to the async VecEnv.
   - train_rl saves a checkpoint and a separate eval process on the idle cores (plus a small GPU share) scores it,
     while training continues.
   - Removes the ≈ 20% eval stall.
   - Raises n vs BC to ≈ 600 (±4 pp) at no wall-time cost.
   - The async `play_match` also speeds every report card and the data generators.
3. **Faster update** (Claude, from the P0 profile). Candidates:
   - SDPA/flash attention path, `torch.compile` and fused AdamW;
   - fewer host↔device syncs (`.item()` in every minibatch);
   - 1 vs 2 PPO epochs. This changes the algorithm, so it needs an A/B on the eval, not just a speed measurement.
   - Target 1.3–2× on the 9 s.
4. **Overlap rollout with the update** (actor thread on the previous weights; PPO's ratio already handles
   off-policy samples): iteration ≈ max(rollout, update) instead of the sum, about 1.4×.
5. **Job overhead** (muse):
   - a CUDA image with torch preinstalled (saves 1–3.5 min a job);
   - drop the separate smoke stage: the main run's iteration-0 eval and first-iteration fail-fast checks cover it
     (saves 6.6 min);
   - relay eval lines to the launcher log, so progress shows up without ssh;
   - steer offers away from machines in `vast_bad_machines.json` / with failures.
6. **GPU price/performance** (one bench job, ≈ $X): since training is GPU-bound, measure update samples/s per $
   on 3090 / 4080 / 4090 / 5090 offers and rent the best ratio.

## Expected effect
- **Per round today:** ≈ 3.5–4 h wall time (acquisition + setup + smoke + 2.5 h training + 20% eval stall +
  33 min card job) for ≈ 22M samples and ±6.5 pp evals.
- **After P1–P5:** the same samples in ≈ 2–2.3 h on one box, with ±4 pp evals. Or, in the same time, ≈ 2× the samples.
- P6 may cut $/sample further.

## Rules carried over
- Box at full planned capacity (cores for eval/env, GPU for training).
- Fail-fast checks in every new stage.
- No ML on the laptop: tests use tiny nets.
- Laptop steps go through `taskset -c 0-3`.
