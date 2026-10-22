# ADR 0001 — Repo layout, machine roles, ML framework

Status: accepted (2026-10-01; amended same day: homelab dropped, laptop does all non-GPU work; 50 GB storage cap)

## Context
- The code mixes a performance-critical C++ part (engine + batched gym env) with Python (cards, data,
  search, model, training, evaluation).
- There are two machines: this laptop (always on, docked) and vast.ai (rented GPU, ephemeral, ~$X of credit).
- Outside data changes over time (card scripts, card DB, banlists), so versions must be pinned and recorded.
- Work is done by coding agents across context resets, so boundaries and contracts must be explicit.

## Decisions

### Machines
| Machine | Role | Holds |
|---|---|---|
| **Laptop** (i7-10510U 4c/8t, 15 GiB, ~115 GB free) | Everything except GPU training: git, builds, tests, simulation, search, LLM data generation, monitoring vast jobs | **All** of `data/` and `runs/` (prune old checkpoints) |
| **vast.ai** | GPU training only, short jobs with a time cap and auto-destroy | Nothing; checkpoints/metrics get pulled to the laptop before destroy |

### ML framework
**PyTorch.** The network is new (effect tokens, no per-ID table), so there are no JAX weights to reuse.
Using the old ygo-agent JAX checkpoint as a benchmark opponent, if ever needed, stays an isolated
optional extra in `eval/`.

### Build
- The engine (`third_party/ygopro-core`) builds with its native **meson** (`scripts/build_core.sh`).
- The C++ env + pybind11 extension also uses **meson / meson-python**, so there's one build system and
  `uv sync` builds it the same way on the laptop and on vast. meson/ninja come in as uv dev deps, so no sudo is needed.
- **No custom docker image** (the laptop user isn't in the docker group). vast jobs use a stock PyTorch
  image and build the extension on the instance (a few minutes).

### Layout
```
ygo-zero/
├── AGENTS.md CLAUDE.md STATUS.md PLAN.md NOTEBOOK.md
├── pyproject.toml uv.lock .gitignore .gitmodules
├── third_party/        # pinned submodules (never edited): ygopro-core, CardScripts, BabelCDB
├── cpp/ygoenv/         # C++ env: core wrapper, message parsing, featurizer, batched env,
│                       #   replay-clone, puzzle start; pybind11 → decklight._ygoenv
├── src/decklight/
│   ├── engine/         # ctypes reference driver (correctness oracle) + wrappers over the extension
│   ├── cards/          # cdb access, Lua effect parser, PSCT segmentation, card tokens
│   ├── decks/          # .ydk I/O, pool / hold-out registry
│   ├── llm/            # OpenCode client, response cache, prompts/*.yaml
│   ├── data/           # demo dataset format, .yrpX import, synthetic line generation
│   ├── search/         # turn-1 combo search, MCTS
│   ├── model/          # PyTorch networks
│   ├── train/          # BC, PPO/VRPO
│   ├── eval/           # gauntlet, Elo, end-board metrics
│   ├── infra/          # vast launcher (nothing else imports this)
│   └── cli.py
├── configs/{env,model,train,eval}/*.yaml
├── decks/{pool,holdout}/*.ydk  (+ gauntlet.yaml)   # small, versioned
├── tests/{unit,engine,gpu}/                        # pytest markers per tier
├── scripts/            # thin shell entry points only
├── docs/               # obs_spec.md, action_spec.md, card_tokens.md, adr/
├── data/               # gitignored: card features, demos, llm_cache
└── runs/               # gitignored: one immutable dir per run
```
Create subpackages when they get their first real code; don't scaffold them empty.

### Rules that follow
- **Dependencies point one way:** engine → cards → model → train → eval. `infra/` and `llm/` are leaves;
  the core library never imports them.
- **Logic lives in the package; scripts are thin wrappers.**
- **Experiments are config-driven.** A run = config + git sha + data version → `runs/<date>-<name>-<sha>/`
  {config, metrics, ckpt}. No hyperparameters in shell scripts, and no scripts rewriting themselves.
- **Contracts are tested:** obs/action spec and card-token docs, a determinism golden test
  (same seed + responses → same trace), and C++ env vs ctypes oracle on the same seed.
- **Test tiers:** `unit` (laptop, seconds), `engine` (needs built core), `gpu` (vast only).
- **Pinned data:** record submodule SHAs in `data/MANIFEST` alongside every derived artifact.
- **LLM calls:** prompts are versioned files; the cache is keyed by (model, prompt hash) and lives under `data/`.

### Storage (hard cap: 50 GB for the `ygo-zero/` folder)
The global uv cache (`~/.cache/uv`) is outside the cap; prune it with `uv cache prune`.

| Item | Budget | Rule |
|---|---|---|
| Code, submodules, git, `build/` | < 0.5 GB | — |
| `.venv` | ~1.5 GB | **CPU-only PyTorch** on the laptop (CUDA wheels would be ~6–7 GB; ygorl's JAX-CUDA venv was 7.3 GB) |
| Embedding/model weights | ≤ 2 GB | `HF_HOME=data/cache/hf` (inside the project, so it counts); one embedding model |
| Card features + LLM cache | < 1 GB | float16 vectors; LLM answers as compressed text |
| Game data (demos, self-play, human replays) | 1–3 GB | **Stored as replays: seed + decks + response list (~1 KB/game).** Observations are regenerated by the engine on load. Raw obs would be ~1 MB/game. |
| Temporary training shards | ≤ 10 GB | compressed; deleted when the run ends |
| Checkpoints (`runs/`) | ≤ 10 GB | per run: last 2 full (with optimizer) + `best` + milestones as fp16 weights-only; others pruned automatically |
| Logs, metrics, eval results | < 1 GB | — |
| **Expected total** | **~20–28 GB** | ~20 GB headroom |

Enforcement: a storage check runs before every data-generation or training job. It warns at 40 GB and refuses to
start at 45 GB. Retention and shard cleanup are done by code (the run manager), not by hand.

