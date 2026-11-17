# Cards and decks API

The lobby WebSocket takes JSON messages with a `t` field. These messages work in the lobby and during a duel.
Every request may carry `req` (any JSON value); every reply echoes it unchanged. Errors are
`{"t": "error", "req": ..., "error": "<plain English>"}`.

`cards_search` and `card_get` are the card services the deck builder and a future in-app LLM ("Ask" tab) share.
`deck_*` manage saved user decks (`decks/user/*.ydk` + `decks/user/index.json`).

## Card search

### `cards_search`

```json
{"t": "cards_search", "req": 7, "q": "ash blosom", "filters": {"kind": "monster"}, "offset": 0, "limit": 40}
```

```json
{"t": "cards", "req": 7, "q": "ash blosom", "total": 1, "rows": [
  {"code": 14558127, "name": "Ash Blossom & Joyous Spring", "kind": "monster",
   "line": "Level 3 · FIRE · Zombie · 0/1800", "extra": false,
   "match": {"by": "name", "snippet": ""}}
]}
```

- `q`: search text (default `""`). Names are fuzzy (accents and punctuation ignored, `"snakeye"` finds
  `Snake-Eye`, `"ash blosom"` finds `Ash Blossom & Joyous Spring`). Effect text is searched with BM25.
  Double-quoted phrases (`"banish face-down"`) are required substrings of the effect text.
- `mode`: `"names_text"` (default) or `"describe"` (semantic search; see below). `"describe"` answers
  `describe mode not installed` until the model is downloaded, and ranks on its own thread (the reply arrives
  later, still echoing `req`).
- `filters` (all optional, applied before ranking):
  - `kind`: `monster` | `spell` | `trap` (string or array).
  - `attribute`: `EARTH` | `WATER` | `FIRE` | `WIND` | `LIGHT` | `DARK` | `DIVINE` (string or array).
  - `race`: race name, e.g. `Spellcaster`, `Beast-Warrior`, `Cyberse` (string or array).
  - `level`: `[min, max]` over Level/Rank (Link rating for Link monsters).
  - `atk`: `[min, max]` inclusive. Monsters only; a `?` ATK (stored as -2) never matches a range.
  - `def`: `[min, max]` inclusive. Monsters only; Link monsters have no DEF and never match, and `?` DEF never
    matches.
  - `st_type`: Spell/Trap subtype: `normal` | `quickplay` | `continuous` | `equip` | `field` | `ritual` | `counter`
    (string or array). Matches Spells and Traps only; `normal` is a Spell/Trap with none of the other subtype bits.
  - `frame`: `normal` | `effect` | `ritual` | `fusion` | `synchro` | `xyz` | `pendulum` | `link` | `token`
    (string or array).
  - `archetype`: one setcode, decimal or `0x` hex.
  - `extra_only`: `true` restricts to Fusion/Synchro/Xyz/Link monsters.
  - Any other key is an error, never ignored. Ranges must be `[min, max]` integers with `min <= max`; a bad range
    or an unknown `st_type` value is an error.
- `offset` (default 0), `limit` (default 40, maximum 200).
- `total` is the number of matches before the page. `match.by` is `name`, `text` or `archetype`; `match.snippet`
  is the matching effect sentence (at most 160 characters) for text hits, else `""`.

### `mode: "describe"` (semantic search)

```json
{"t": "cards_search", "req": 8, "mode": "describe",
 "q": "a light tuner that special summons itself and searches", "limit": 20}
```

The query is encoded with the optional Qwen3-Embedding-0.6B fp16 model (with the query instruction). It is
split into clauses on `and` and commas; a card's score is the mean, over clauses, of its best-matching effect
embedding. That ranking is fused with the BM25 text ranking by reciprocal rank fusion (k = 60). Filters,
`offset`, `limit` and `total` work as in `names_text`; `match.by` is `"describe"` and `match.snippet` is the
best-matching effect sentence for the first clause (or the BM25 snippet when the text cannot be split).

If the model is not installed the reply is still the `names_text` error shape (`describe mode not installed`).

### `card_get`

```json
{"t": "card_get", "req": "c1", "code": 14558127}
```

```json
{"t": "card", "req": "c1", "code": 14558127, "name": "Ash Blossom & Joyous Spring", "kind": "monster",
 "line": "Level 3 · FIRE · Zombie · 0/1800", "text": "When a card or effect is activated ...",
 "archetypes": [], "extra": false, "alias": 14558127}
```

Unknown codes are an error (`unknown card <code>`).

## Describe-a-card model

The semantic model (~1.2 GB) is an optional download, not part of the app bundle. It installs into
`models/qwen3-embed-0.6b-fp16/` (pinned Hugging Face revision `c25a394`, three SHA-256-checked files) and is
downloaded one file at a time, resuming an interrupted `.part` with an HTTP `Range` request. Every reply
echoes `req`.

### `describe_status`

```json
{"t": "describe_status", "installed": false, "downloading": true, "bytes": 421345678,
 "total": 1211544523, "error": null, "rev": "c25a394"}
```

`installed` is true only when `installed.json` is present and all three files have the pinned sizes.

### `describe_download`

Starts (or resumes) the download and replies with the current `describe_status`. While it runs the server
pushes `describe_status` frames to the requesting connection at most every 500 ms and once at the end
(installed, cancelled or error). Only one download runs at a time; a second request while running is a no-op
that still answers with the status. If the disk lacks room the reply is an error
(`Not enough free space: needs X GB`).

### `describe_cancel`

Cancels a running download between chunks and keeps the `.part` file for a later resume. Replies with
`describe_status`.

### `describe_remove`

Deletes the install directory and drops the loaded encoder session. Replies with `describe_status`
(`installed: false`, `bytes: 0`).

## Saved decks

A deck is `{main: [code], extra: [code], side: [code]}`. Legality uses the Standard format (Main 40–60,
Extra 0–15, Side 0–15, 3 copies max counting alternate arts together). `counts` are raw card counts.

### `deck_list`

```json
{"t": "deck_list", "req": 1}
```

```json
{"t": "user_decks", "req": 1, "decks": [
  {"id": "my-snake-eye-a1b2c3", "name": "My Snake-Eye", "updated": "2026-11-13T10:20:31",
   "cover": 9674034, "counts": {"main": 40, "extra": 15, "side": 3},
   "legal": {"ok": true, "problems": []}}
]}
```

Newest first. `updated` is local ISO time (`YYYY-MM-DDTHH:MM:SS`).

### `deck_get`

```json
{"t": "deck_get", "req": 2, "id": "my-snake-eye-a1b2c3"}
```

```json
{"t": "user_deck", "req": 2,
 "deck": {"id": "my-snake-eye-a1b2c3", "name": "My Snake-Eye", "main": [1, 2], "extra": [], "side": [],
          "updated": "2026-11-13T10:20:31", "cover": 9674034},
 "legal": {"ok": false, "problems": [
   {"kind": "main_count", "text": "Main 38/40 · 2 more", "codes": []}
 ]},
 "unknown": [99999999]}
```

`legal.problems` here are full objects (`kind`, `text`, `codes`); `deck_list` uses only their texts.
`unknown` lists codes that are not in the card store (they are kept in the deck).

### `deck_save`

```json
{"t": "deck_save", "req": 3, "id": "my-snake-eye-a1b2c3", "name": "My Snake-Eye",
 "main": [1, 2], "extra": [100], "side": [], "cover": 9674034}
```

```json
{"t": "deck_saved", "req": 3, "id": "my-snake-eye-a1b2c3", "updated": "2026-11-13T10:22:05",
 "legal": {"ok": true, "problems": []}}
```

Without `id` a new id is generated (`<slug>-<6 hex>`). More than 200 cards, or a code that is not a positive
integer, is rejected. A deck that is not legal can still be saved.

### `deck_delete`

```json
{"t": "deck_delete", "req": 4, "id": "my-snake-eye-a1b2c3"}
```

```json
{"t": "deck_deleted", "req": 4, "id": "my-snake-eye-a1b2c3"}
```

The `.ydk` file is moved to `decks/user/.trash/`, never hard-deleted.

### `deck_import`

```json
{"t": "deck_import", "req": 5, "ydk": "#created by decklight\n#main\n14558127\n", "name": "Ash"}
```

```json
{"t": "deck_saved", "req": 5, "id": "ash-d4e5f6", "updated": "2026-11-13T10:23:44",
 "legal": {"ok": false, "problems": ["Main 1/40 · 39 more"]},
 "unknown": [], "unknown_lines": ["some junk line"]}
```

Accepts `.ydk` text or a `ydke://` URL. The deck is stored exactly as given: Extra-Deck monsters found in
`#main` are not rearranged; legality reports the result. `unknown_lines` are non-comment lines that were not
positive card codes.

### `deck_export`

```json
{"t": "deck_export", "req": 6, "id": "my-snake-eye-a1b2c3"}
```

```json
{"t": "deck_ydk", "req": 6, "id": "my-snake-eye-a1b2c3", "name": "My Snake-Eye",
 "ydk": "#created by decklight\n#main\n...", "ydke": "ydke://...!"}
```

### `lobby_decks` (server → client)

```json
{"t": "lobby_decks", "decks": [ /* the lobby's full deck list */ ]}
```

Sent after every successful `deck_save`, `deck_import` or `deck_delete`, so the Bot / Multiplayer / Puzzles
pickers and the Decks tab update without a reload. Rows come in three groups:

- `"group": "saved"`: the user's decks, newest first. These rows come **first**; `"user": true`.
- `"group": "prebuilt"`: the curated bundle set, in `config/app/prebuilt_decks.txt` order.
- `"group": "tuned"`: decks found for fine-tuned models.

```json
{"id": "user:my-snake-eye-a1b2c3", "name": "My Snake-Eye", "group": "saved", "user": true,
 "legal": true, "problems": [],
 "meta": {"arch": "", "style": "", "about": "", "origin": "Built in Decks", "year": 0,
          "mech": ["Fusion", "Link", "Ritual"], "counts": [24, 14, 2], "sig": [9674034, 14558127, 10000000],
          "main": [[14558127, 3], [9674034, 1], /* ... */], "extra": [[10000000, 1], /* ... */]}}
```

`legal` and `problems` use the same validation as `deck_list` (`problems` are only their texts). A deck pasted in
this session (not saved) still appears with `"group": "user"`.

Saved rows carry a `meta` object in the same shape as a `deck_index.json` entry, so the client renders them with the
same row and deck-list view as the prebuilt decks:

- `main` / `extra`: `[code, copies]` pairs in first-appearance order. The Side deck is not listed.
- `counts`: `[monsters, spells, traps]`, copies counted, from the card DB's type bits (unknown cards count nowhere).
- `mech`: summon mechanics present in the deck, in this order and each at most once: `"Fusion"`, `"Synchro"`,
  `"Xyz"`, `"Link"` (any Extra card with that frame), `"Ritual"` (a Main-deck Ritual monster), `"Pendulum"`
  (any Main or Extra Pendulum card).
- `sig`: up to 3 cards for the row's art: the saved cover first (if it is in the deck), then Main cards by copies
  (ties: first appearance), then Extra cards; duplicates are dropped.
- `arch`, `style`, `about` are empty, `year` is `0`, and `origin` is `"Built in Decks"` (no primer for saved decks).
- An empty deck has `counts: [0, 0, 0]`, `mech: []` and `sig: []`.

## Starting a game with a saved deck

`new_game` and `room_host` accept a saved deck next to the pool decks:

```json
{"t": "new_game", "deck": {"user": "my-snake-eye-a1b2c3"}}
```

The lobby row id works directly, in its `"user:<id>"` string form (same id validation and errors as
`{"user": "<id>"}`):

```json
{"t": "new_game", "deck": "user:my-snake-eye-a1b2c3"}
```

`{"ydk": "<text>"}` still accepts a pasted deck.
