# Workflow: where time went (2026-10-01/02) and how to go faster

Measured on the session that built Phase 0 + the demo pipeline. Read this before planning work or writing a muse brief.

## Where we were slow

| # | Bottleneck | Evidence | Fix |
|---|---|---|---|
| 1 | **Verification is serial and CPU-bound** | Salamangreat 8 lines × 3 seeds = 5–30 min; expansion 72 variants = 14 min; 3 of 4 cores idle | Run lines in parallel (`multiprocessing.Pool`, one process per line×seed, 4 workers on the laptop). Expected ~3x. |
| 2 | **3 seeds per line add almost nothing** | On turn 1 the obs shows the own deck as an unordered multiset, so other seeds only change draws/random effects. They tripled verify time | Verify with 1 seed; spot-check other seeds only for lines with random effects. Diversity comes from hand variants (`expand_lines.py`), not seeds. |
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
