# Deck builder: design

Status: proposal, 2026-11-12, waiting for the user's OK.

The user asked for this: "a deck builder so people can build their decks and then also export the ydk either as a
file or copying to paste it somewhere, think about the core ui designs here to implement this, think that as a modular
piece to as well since we might have new features in the future".

## What exists to build on
- The server holds every card (14 792 rows in BabelCDB), their texts and 760 archetype names (`!setname` in
  `edopro-strings.conf`). Card art comes from the art sync (`imgUrl`). The card detail sheet (`#detail`, a lookup) and
  the deck browser's list-and-filter pattern (`web/render/decks.js`) already exist.
- The lobby takes a pasted .ydk (the "Your deck" picker). There is no card search, no saved user decks and no export.

## User flows
1. **Decks**: a new lobby tab, next to Multiplayer · Bot · Puzzles.
   - It lists your decks: cover card, name, "Main 40 · Extra 15 · Side 0", a legality line in text.
   - Actions: New deck · Import (.ydk file or paste) · open a deck.
   - The "Your deck" picker in the Bot and Multiplayer tabs lists your decks first, then the pool.
2. **Editor**:
   - Header: back, editable name, counts.
   - Body: Deck view (Main / Extra / Side as card grids with ×2 / ×3 badges) and Add cards (search).
   - Tapping a search result adds a copy. Fusion, Synchro, Xyz and Link cards go to Extra by themselves. Long-press or
     ⓘ opens the card's details, which have − / + / "To Side".
   - Tapping a card in the deck opens a small action row: − · + · Move to Side/Main · Details.
3. **Export**: one Export button opens a sheet with the deck's .ydk text and three actions:
   - Copy (clipboard);
   - Share or save the .ydk file: the Android share sheet (Files, Drive, Discord…) in the app, a download in a
     browser;
   - Close.
4. **Import**: open a .ydk file or paste text. Unknown card ids are listed by id and kept, never silently dropped.
5. **Legality**, always visible and never colour alone: "Main 38/40 · 2 more", "3 copies max: Ash Blossom ×4". A deck
   that is not legal can still be saved and exported; Start duel says why it can't use it.

## Search: a mode menu, measured rankers
The user asked (2026-11-12): "Are you planning to include some kind of fuzzy search too? Also maybe we could leverage
the embeddings we have for the cards to do semantic search too right … basically we should support: I want a card
that does something like this and that". Then: "Is the qwen model too big for the phone? You sure about that? How much
ram/core does it take? We can make it maybe for semantic search the user has to tap a button to search, like a
drop-down menu with two options: normal search (the two modes) and semantic search".

The search bar has a small mode menu next to it:
1. **Names & text** (default, live as you type):
   - fuzzy names and archetypes: a trigram index with Damerau-Levenshtein re-ranking, prefix boost, and accents and
     punctuation ignored ("ash blosom", "snakeye", "S:P little");
   - BM25 over effect text, with stemming and quoted phrases ("banish face-down").
2. **Describe a card** (runs on the Search tap): "a light tuner that special summons itself and searches".
   - The query is encoded on the phone by Qwen3-Embedding-0.6B (fp16 ONNX, through the app's ONNX Runtime) with the
     query instruction, and compared with the per-effect embeddings already shipped
     (`data/cards/emb/qwen3-0.6b-d512.npz`).
   - "X and Y": the query is split into clauses on "and" and commas. A card's score is the mean over clauses of its
     best-matching effect, so a card needs both.
   - The final list fuses this ranking with BM25 by reciprocal-rank fusion.
   - Each result shows the matched effect sentence, so results can be checked.
   - The model (1.2 GB) is an optional download in Settings, not in the APK. Until it is installed, the mode explains
     that and offers the download.

Filter chips (type, attribute, level…) narrow both modes.

**Measured 2026-11-12** (NOTEBOOK; `scripts/search/semantic_eval.py`, `rust/spikes/embed-bench`). The test set is 60
cards common in decks/pool, with 2 LLM-written player queries each (one property, two properties). The metric is
exact-card recall@10 over 14 606 card names.

| ranker | recall@10, one property | recall@10, two properties |
|---|---|---|
| BM25 | 0.18 | 0.13 |
| fp16 + instruction, clause split | 0.22 | 0.23 |
| **fp16 + instruction, hybrid RRF with BM25** | **0.38** | **0.38** |
| int8 + instruction, clause split | 0.17 | 0.20 |
| q4f16 + instruction, clause split | 0.17 | 0.20 |

- On the S23 Ultra (4 threads), fp16 takes 145 / 239 / 389 ms for 16 / 32 / 64 tokens, with about 0.93 GB peak RSS.
  int8 is about 4x faster but loses quality: only 26 % top-10 overlap with fp16, and 0.82–0.91 cosine on the card
  texts. fp16 reproduces the shipped vectors at cosine 1.00.
- A tap-to-search mode can afford fp16. The static distilled encoder stays a later option, gated on beating this
  hybrid.
- The model stays loaded only while the builder is open and is dropped when the builder closes.

## Search and cards as shared services (user, 2026-11-13)
"the search could be also another tab not just within deck builder, I'm thinking here in the future where we bundle
another llm where you can directly ask for cards and we provide all the necessary tools to get information, for example
you talk to the llm and it fetches the relevant cards, it does some calculations on game positions and all of that. So
it would be a fifth tab right? Like deck builder and a brainstorming or something like that. But let's start with the
deck builder and then we can think about this other part".

So:
- **Card search and card lookup are server services, not deck-builder internals.** The messages
  (`cards_search`, `card_get`, `deck_*`) are request/response JSON with a `req` id echoed back, documented in
  `docs/design/api/cards-and-decks.md`. A future in-app LLM ("Ask" or brainstorming tab) calls the same messages as its tools,
  with no second API to keep in sync.
- **The search UI is one component** (`web/features/cards/search.js`): the search bar, the mode menu, filter chips and
  the result list. The deck builder embeds it with an "add" action. A standalone **Cards** tab can later embed it with a
  "details" action and no deck.
- The lobby tabs are planned as Multiplayer · Bot · Puzzles · **Decks**. Cards and Ask come later, and the tab strip
  must fit 6 tabs at 360 px (icons + short labels, or a "More" overflow).
- Split of work (user): Claude does the visual parts; DeepSeek does the plumbing (formats, storage, search engine,
  server messages, tests).
- One search bar for both editor views (user, 2026-11-13: "we should also have a search for the deck itself right? ...
  For example the search is for both deck and add cards tabs"). Query and kind chips carry over between the views. In
  **Deck** it filters the deck on the client (`localMatcher` in `web/features/cards/search.js`: every word must appear
  in name, type line or text; the status reads "N of M cards match", counted in copies; there is no mode menu). In
  **Add cards** it runs `cards_search` over every card. Both views have a clear (×) button.

## Layout
Phone (one column, thumb-first):
```
┌ ‹  My Snake-Eye           ⋯ ┐   ⋯ = Export · Duplicate · Delete
│ Main 40 · Extra 15 · Side 3 │   legality line under the counts
├─────────────────────────────┤
│ [Deck]  [Add cards]         │   segmented switch; the deck keeps its scroll
│ ▢▢▢▢▢  ▢▢▢▢▢  ▢▢▢▢▢ …        │   Main grid, then Extra, then Side (headers stick)
│ ×3 badges, tap → − + Side ⓘ  │
├─────────────────────────────┤
│ 🔍 Search cards…    Filters │   the search bar sits at the bottom (reach);
└─────────────────────────────┘   typing slides the results up as a sheet
```
Results sheet: a list of rows (thumb, name, one stat line such as "Level 4 · LIGHT · Spellcaster · 1500/1000"), a +
button per row (44 px) and the count already in the deck. Filter chips: Monster / Spell / Trap, Attribute, Level/Rank,
Type, Archetype (searchable), Extra types. Text search matches name and effect text.

Wide screen (≥ 900 px): three columns: deck · results · card detail.

Night design and WCAG 2.2 AA as everywhere: targets ≥ 44 px; counts as text; focus kept when a sheet opens;
reduced motion respected.

## Modular structure
The builder is a feature module with no special cases in `app.js`. Future features plug in the same way (a
collection, side-decking between games, a "cards" browser, banlist formats).
- **Client**
  - `web/lib/ydk.js`: parse and serialize .ydk (pure). The lobby paste box and rooms reuse it.
  - `web/lib/deck.js`: the deck model and `validate(deck, format)` (pure). A format is
    `{main: [40, 60], extra: 15, side: 15, copies: 3, banlist: null}`, so banlists (and Rush or Speed) plug in later.
  - `web/features/deckbuilder/{index,view,actions}.js`: its own state, pure HTML builders and action handlers.
  - A small feature registry in `app.js`:
    `registerFeature({id, tab?, render(state), act(name, el), onMessage(m)})`. Actions are named `db:*` and messages
    `deck_*` / `cards_*`. Overlays still go through the stage (docs/design/ui-attention.md: the editor's sheets are layer 3,
    lookups).
  - Shared view pieces: `web/render/cardgrid.js` (thumbnails + badges) and the existing card detail.
- **Server (Rust, decklight-server)**
  - `decks.rs`: user decks as `.ydk` files in `<root>/decks/user/` plus a small `index.json` (name, updated,
    cover). Messages: `deck_list`, `deck_get`, `deck_save`, `deck_delete`, `deck_import`. User data survives app updates
    (outside every asset unit).
  - `cards_search.rs`: `cards_search {q, filters, offset, limit}` → rows `{code, name, kind, stat line}`, over the
    in-memory card data plus archetype names. It returns in milliseconds and is unit-tested. Any future feature can
    reuse it.
  - New games accept `deck: {user: id}` next to the pool decks.
- **Android**: a `DecklightShare` JavaScript interface, the same pattern as `NetBridge`. It shares or saves the .ydk
  through the system share sheet (FileProvider). The browser build falls back to a Blob download.

## Phases
1. Model and server: `ydk.js`, `deck.js`, `decks.rs`, `cards_search.rs` (fuzzy names + BM25), with tests.
2. The editor UI on mocks, with screenshots for the user before any APK.
3. Export and import, including the Android share bridge.
4. Picker integration (your decks in Bot and Multiplayer).
5. "Describe a card": the optional model download, the on-device fp16 encoder and the RRF hybrid in `cards_search`
   (can run in parallel with 2–4).
