# Workflow: where time went (2026-10-01/02) and how to go faster

Measured on the session that built Phase 0 + the demo pipeline. Read this before planning work or writing a muse brief.

## Where we were slow

| # | Bottleneck | Evidence | Fix |
|---|---|---|---|
| 1 | **Verification is serial and CPU-bound** | Salamangreat 8 lines × 3 seeds = 5–30 min; expansion 72 variants = 14 min; 3 of 4 cores idle | Run lines in parallel (`multiprocessing.Pool`, one process per line×seed, 4 workers on the laptop). Expected ~3x. |
| 2 | **3 seeds triple verify time** | ~146 s per seed for Salamangreat on the laptop. CORRECTION (vast smoke 2026-10-02): seeds DO change the verdict for some lines (Salamangreat: 3 verified with seed 0 alone vs 6 with seeds 0–2), so they raise yield, not just obs diversity | Keep several seeds but run them **in parallel** (one process per line×seed), and stop at the first seed that verifies when only one record is needed. |
| 3 | **Backtracking replays the whole prefix** | `verify._search.restore()` rebuilds the duel (~15 ms) and replays every choice on each backtrack: O(depth) per node, and long lines are 60–130 choices deep | Keep env checkpoints at step boundaries (restore from the nearest), or a small LRU of envs by prefix length. |
| 4 | **Laptop throttles** | Same benchmark 1.3k → 0.66k steps/s after an hour; 1.3 GHz under load | Bulk verify/expand runs → vast CPU box (9+ effective cores ≈ 3x, ~$X/h). The launcher is ready (`scripts/vast_job.py`). |
| 5 | **Muse runs that also execute long jobs** | scale-gold-decks 91 min, extract-v2 79 min, pool-decks 61 min: most of it was muse waiting on verify runs / LLM calls inside its own session | Muse produces code and artifacts; Claude launches the long runs (background, parallel). Briefs say "do not run verification for more than one line; Claude runs the full sweep". |
| 6 | **One deck at a time, serially** | 4-deck run is one muse session going deck by deck | One muse session per deck (they write disjoint `data/guides/<deck>/`, `decks/pool/<deck>.ydk`), each with its own report file, run in parallel (2–3 at once; LLM throttle is per call). |
| 7 | **Bad wait loops** | `pgrep -f <brief>` matched the waiting shell itself (waited forever); 600 s tool timeouts killed waits repeatedly; one loop ran 40 min doing nothing | Wait on the PID: `opencode run ... & echo $! > data/cache/briefs/<task>.pid`, then `while kill -0 $(cat …pid) 2>/dev/null; do sleep 15; done` with `run_in_background`. Never `pgrep -f` a string that also appears in the wait command. |
| 8 | **The opencode arg-order bug** | `-f brief.md "message"`: `-f` swallows the message as a 2nd file → "File not found", muse did nothing | The message goes **before** `-f` (fixed in AGENTS.md). |
| 9 | **Prompt edits invalidate the whole LLM cache** | extract-v2 re-ran all sections (+25 calls) after a prompt tweak | Version prompts (`PROMPT_VERSION`) and re-extract only the decks/sources that need it. |
| 10 | **Reading long logs to review muse** | Each review meant scrolling 3–5k chars of log tail | Briefs demand a fixed final block (below), so reviews are a `grep -A40 "## RESULT"`. |
| 11 | **Concurrent edits** | Muse ran verify on code I was editing (results depended on timing); `git add -A` once swept muse's in-progress files into a commit | Commit before launching muse. Don't edit files a running muse job executes. Always `git add <paths>`, never `-A`, while muse runs. |
| 12 | **Stage-by-stage manual chaining** | Every change meant re-running discover → extract → verify → expand by hand per deck | One `scripts/demo_pipeline.py --decks ... --stages extract,verify,expand` with per-stage caching (skip when inputs are unchanged by hash). |

## Muse brief template (what worked, what to add)

What worked: exact file allow-list, explicit "do NOT edit STATUS/NOTEBOOK; put suggested notebook text in your reply",
a concrete acceptance command, a primary target to report on ("the White Stone line: what happens, and the options
at the failure point"). Muse is good at: parsers/plumbing (Lua effects: one pass, 0 fixes), web discovery and
reports, careful forensics (recovered the tflite card-id permutation). Muse overreaches when allowed: it edited
STATUS.md, and it runs long jobs inline.

Add to every brief:
```
## Limits
- Time budget: stop after ~30 min of work and report what is done/left.
- Do not run any job expected to take > 5 min; write the command in your report instead.
- Files you may create/modify: <list>. Anything else: describe the change in the report.
## Report format (end your reply with exactly this)
## RESULT
status: done | partial | blocked
changed_files: [...]
acceptance: <command> → <one-line result>
numbers: {...key metrics...}
issues: [...]
suggested_notebook: |
  ...
```

## Cheap wins to do first (next session)
1. Fix #7 + #8 in practice (pid files; arg order) — done in AGENTS.md.
2. Parallel `verify_lines.py` / `expand_lines.py` (#1) and default to 1 seed (#2). Brief for muse: mechanical.
3. Env checkpoints in `_search.restore` (#3). Claude: it is search code.
4. `demo_pipeline.py` orchestrator with stage caching (#12). Muse.

## Making muse and the LLM stages fast (measured 2026-10-02)

- **Per-call latency:** 159 cached LLM calls: median **38 s**, mean 48 s, p90 102 s, all **serial** (`demos/llm.py`
  spawns `opencode run --pure` per call; it also tries the HTTP endpoint first, which always fails here).
  One deck's extraction ≈ 25 calls ≈ 15–20 min of waiting.
  1. **Concurrency:** run calls in a thread pool (4–8 at once; each is a separate CLI process). ~5x on extraction.
  2. **Right model per job:** muse (reasoning) only for hard parsing. Try `deepseek-v4.1-flash` for section
     extraction and the nickname layer, and gate it on the gold set (recall must not drop) before switching.
  3. **Fewer, bigger calls:** one call per guide (all combo sections at once) instead of one per section.
  4. **Skip the dead HTTP attempt** (process-level flag after the first failure).
- **Muse agent sessions** are sequential model turns of ~30–60 s each, so turns are the cost, not tokens:
  1. Put the context in the brief (exact paths, function signatures, the command lines to run) so muse doesn't spend
     turns exploring the repo.
  2. One deliverable per brief; split per deck and run 2–3 sessions in parallel.
  3. Never let muse wait on a long run (verify sweeps, expansion): it writes the command, Claude runs it.
- **vast for CPU sweeps: latency vs throughput.** A cold job costs ~2–5 min of overhead (offer, boot, image pull, uv sync,
  core + extension build) plus seconds of transfer, so it only pays when the laptop run would take > ~10 min.
  Rules:
  - Interactive debugging (one line, `explain_line.py`) and anything < 10 min: laptop, parallel (4 procs).
  - Sweeps (all decks verify + expand, big searches): a **warm session box**. Rent one CPU box at the start of a
    working session (≈ $X/h, so even 8 idle hours ≈ $X), push code deltas with tar/rsync (seconds), run sweeps
    there, pull `data/demos/` back, destroy it at the end (launcher `--keep` mode + `--cleanup`; TODO).
  - Also cache the built `.venv` + `build/` on the box between pushes, so only changed Python files move.

## Model choice: cost vs latency vs quality (benchmarked 2026-10-02, `scripts/bench_llm.py`)

OpenCode Go ($X/month) limits are **monthly dollar budgets per model** (5 h window = 20%, week = 50%): muse $X/month
(~45k requests per 5 h), deepseek-v4.1-flash $X (~26k / 5 h; off-peak price, **doubled** Mon–Fri 01–04 and 06–10 UTC),
longcat-2.5-preview-free unlimited and not counted (limited-time preview). Our whole usage in 14 days: $X notional
($X of it muse). **Cost is not a constraint for any of them; latency and quality are.**

| model | short answer (~2.3k chars) | long answer (~8–10k chars) | quality on our extraction prompts |
|---|---|---|---|
| muse-spark-1.3 | 26–30 s | 67–87 s | **4/4 correct** |
| deepseek-v4.1-flash | **10–12 s** | 89 s | 3/5: twice silently returned `{"lines": []}` on the harder section |
| longcat-2.5-free | 20–30 s (one outlier 345 s) | n/a | 2/4: **always** `{"lines": []}` on the harder section |

Parallel calls work for all three (4 muse calls: 85 s wall vs ~4× serial). Latency scales with **output length**,
so the fast models only win on short answers, and their failure mode (an empty, valid-looking answer) silently
drops data.

Decision:
- **Extraction and anything where missing data is costly: muse, with 4–8 concurrent calls** (the real speed lever).
- **Short, checkable outputs** (classification/tags, yes/no, nickname → one of N candidates): deepseek-v4.1-flash
  off-peak with validation; retry on muse when the answer is empty/invalid.
- LongCat: only for throwaway or easily verified tasks (free, but slow and unreliable on our prompts).
- Agentic coding (briefs): muse.

## vast CPU box, measured (smoke 2026-10-02, Quadro P4000 host, $X/h, cgroup ≈ 17 cores of 72 Xeon E5-2686 v4)
- Boot to ssh: 45–90 s; setup (uv + core + extension build): ~3–4 min, so a cold job has ~5 min overhead.
- VecEnv: 1 thread 1.7k, 8 threads 8.2k, all cores 9.6k agent steps/s (laptop: 2.4k at 8 threads) → ~4×.
- `verify_lines.py` (serial) Salamangreat seed 0: 130 s vs ~146 s per seed on the laptop. **Single-process speed is
  about equal; the box only wins once verification is parallel.**
- Some hosts drop ssh in the first minutes ("closed by remote host", tar exit 141/255); the launcher retries transfers
  and runs jobs detached. Smoke attempts total ≈ $X.

## Muse acceptance checks are smoke-sized (2026-10-05)
Muse cannot run on vast (the OpenCode Go key never leaves the laptop), and its sessions are mostly API wait, so the
laptop's 4-core cap only bites when a brief's acceptance check is really a data job (the scenarios brief asked for a
window search over 60 games: ~90 s/game, ~1.5 h). Rule: acceptance checks use a handful of games / tiny inputs and
finish in < 5 min; the full run is a command in the report that Claude launches on vast.
