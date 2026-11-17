# How decklight was built

One person directed the project; AI coding agents did most of the engineering. This page describes the setup and the
rules that kept results trustworthy.

## Roles
- **The author** set goals and made decisions: what to try next, which gates count, what ships.
- **Claude (Claude Code)** planned experiments, wrote specs, did the subtle work (engine and observation code,
  search, RL debugging), reviewed every change and kept the lab log.
- **A second coding model (via OpenCode)** implemented well-specified tasks from written briefs: parsers, launchers,
  refactors, tests, much of the web client.
  - Each brief lists the files the agent may touch and an acceptance command.
  - The agent ends with a fixed report block.
  - Claude reviewed the diff and the acceptance output before anything was committed.

## Machines
- **A 4-core laptop** for everything except GPU training: builds, tests, simulation, data generation, analysis and
  monitoring.
- **Rented cloud GPUs** for training and large evaluations.
  - Every job is time-capped and checks its own outputs as it runs.
  - It copies results back to the laptop, then the machine is destroyed.
  - Jobs are sized to use the whole machine: process count is set from measured CPU, GPU-memory and RAM use.

## Rules that mattered
- **Pre-register, then measure.**
  - Every experiment's question, arms, gate and decision rule are written down before launch.
  - A result that fails its gate is reported as a failure, even when it looks promising. The decision to override a
    gate is the author's and is recorded as such.
- **Paired evaluation on fixed deals.**
  - Arms play the same deals (decks, shuffle seed, seat), and differences are computed per deal.
  - Win rates against a self-play pool are never reported as progress.
- **Fail fast and loud.**
  - Jobs append problems to an alert file that the monitor relays at once.
  - A long job is checked at its first milestone (first epoch, first 10% of games).
  - Scripts exit non-zero on results that cannot be right: identical checkpoints, empty outputs, NaN losses, a
    fine-tune whose best checkpoint is its initialization.
- **Look at the games, not just the numbers.** Each step includes reading rendered games and decision-level
  statistics. That is how the hand-trap timing habits were found, and how search's noisy overrides were traced
  (`scripts/audit/maxx_search.py`).
- **No hand-written play rules.** Play flaws are fixed through data, rewards, the value function or the search's
  statistics, never by masking moves.
- **Everything reproducible from git.** Launch scripts, analysis scripts and decisions live in the repository; large
  artifacts (games, checkpoints) are regenerated or shipped as release assets.

## What the agents were good at, and not
- **Good at:**
  - mechanical, well-specified work (one pass, few fixes);
  - forensics on file formats;
  - large refactors with a test suite as the oracle;
  - UI built from a mockup.
- **Needed close review:**
  - anything touching the engine message format;
  - subtle statistics (the per-opportunity vs per-game trap shows up more than once in the history);
  - claims made in summaries, which were checked against the raw outputs before being recorded.
