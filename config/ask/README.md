# Ask tab: tools and eval set (v1)

Plan: `docs/plans/ask-tab.md`. Prompts, tools and evals are versioned files here; every chat message records the
versions it ran with.

- `tools.v1.json`: the five v1 tools (`search_cards`, `card`, `deck`, `deck_odds`, `my_stats`), their JSON schemas
  (OpenAI function format), what existing server call each maps to, and the shape of `{summary, ref, block}`.
  `limits.tool_calls_per_answer` is the cost guard (8).
- `eval/v1.jsonl`: 20 questions (user, 2026-11-14: "tab first with ~20, grow the set from use"; Claude drafts, the
  user reviews a sample). One JSON object per line:
  - `q`: the question; `context`: the id of an earlier question in the same chat (follow-ups).
  - `tools`: expected calls, in order. `args` is a subset: every key given must match; text args match
    case-insensitively as substrings; missing keys are free; `null` means the key must be absent.
  - `answer.must`: each entry must appear in the reply; `|` separates alternatives.
  - `answer.numbers`: a percentage in the reply within `tol` points (values computed exactly, hypergeometric).
  - `answer.numbers_from_fixture`: the expected number comes from the fixture's records.
  - `answer.must_not_invent`: every card name in the reply must be one the tools returned this turn.
  - `answer.language`: the reply's language; `answer.max_tool_calls`: overrides the default 8.
  - `fixture`: `starters` = the two starter decks (`blue-eyes-83b783`, `branded-b27239`, copied in
    `eval/fixtures/decks/`); `records-small` = 12 bot games in the record index layout (`eval/fixtures/records-small.sql`;
    going first 5–2, going second 1–3 plus one undecided, all 54.5%). Every run gets a fresh copy of both.
- Scoring per question: tool score (expected calls made, in order, no call with a bad schema) and answer score
  (all checks pass). A model passes the v1 bar at ≥ 90% on both; local models are measured against the same bar.

## Running it
From `ygo-zero/`, after `cargo build --release -p decklight-server`:

    rust/target/release/ask_eval [--only s01,o01] [--model M] [--base-url URL] [--auth opencode|opencode-go | --key-env VAR]

The runner (`rust/decklight-server/src/bin/ask_eval.rs`, scoring in `src/ask/eval.rs`) builds the real app on the
fixtures, runs each question through the real chat loop and tools, prints `[i/N] id: tools ok|FAIL answer ok|FAIL` and
writes one JSON line per question (answer, calls with results, checks, usage) to `data/cache/ask_eval/`. The key is read
from opencode's `auth.json` (or the variable) and never printed. Defaults: OpenCode Zen, `deepseek-v4.1-flash-free`.
OpenCode Go: `--base-url https://opencode.ai/inference/go/openai/v1 --auth opencode-go --model deepseek-v4.1-flash`.
Exit 2 when every answer is empty, 3 when the provider refuses for good.
`--preset opencode-go` (providers.v1.json) takes the base URL, model and key as the app does, the key from this
machine's credential store: `rust/target/release/ask_key set opencode-go` (no echo), or
`ask_key import-opencode opencode-go` to copy opencode's key without showing it; `ask_key status` shows
hints only.
