# ADR 0002 — Demonstration (behaviour-cloning) data pipeline

Status: accepted, 2026-10-02. Decided with the user: the agent collects combo knowledge from the internet itself
(go-ahead given); dataset quality is the priority, because cloning is the main bootstrap for the policy.

## Pipeline (each stage cached, re-runnable, and measured)
1. **Discover + fetch** (agent with web access): for each pool deck, find guides / deck profiles / combo
   videos' transcripts and current decklists. Store the raw text plus metadata (URL, site, date fetched,
   publication date if any) in `data/guides/<deck>/`. Fetch politely: cache by URL, throttle, never refetch.
2. **Extract** (LLM, muse for hard parsing): guide text → structured lines: starting hand (+ required deck/
   extra cards), ordered steps (action, card, effect, targets/materials if stated), claimed end board, and the
   source span. Card names are resolved to codes against our card DB, and unresolved names are flagged, not guessed.
   Starts from ygo-architect's step-3 ComboTree schema and prompts.
3. **Ground + verify in the engine** (the quality gate): a line executor starts a puzzle duel with the hand,
   maps each step to an env option, and searches over the choices the guide leaves unsaid (zones, positions,
   unstated targets/materials). A line is **kept only if it executes and reaches its claimed end board**.
   Output = replay records (config + choice indices) tagged with provenance and with how many choices were
   inferred by search vs stated by the guide.
4. **Expand**: replay verified lines under many seeds / hand permutations where they still hold, and add
   search-found lines (Phase 3b) and ygo-agent rollouts as separately-tagged sources.

## Quality rules
- Gold set: the user's 5 hand-curated guides (`../ygo-architect/data/combo_guides/`). Track extraction recall and
  verification rate on them for every pipeline change.
- Nothing unverified ever enters the training set; every record keeps its provenance (source URL → line →
  replay), so bad sources can be removed later.
- Per-deck report: guides found, lines extracted, % verified, failure reasons (name resolution, illegal step,
  end board mismatch).
