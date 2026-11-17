# ADR 0010: Match record v1, open game dataset, human vs human in the first app release

Date: 2026-11-01. Status: accepted (user's decisions below; the format details are Claude's design, open to review).
Revised the same day: §5 now uses the EDOPro protocol we already have (user: "did you check the whole multirole
support we have here? This might be one of the solutions right"), and the license is decided.
Plan: `docs/plans/2026-11-01-offline-app.md`. Branch: `offline-app`.

## Context
The user wants one game record that is "interoperable", "easily shared back to improve the models" and used for
"stats and history"; decklight should make fetching and sharing data easy; human vs human "like EDOPro", where "the
goal is not for us to run the server per se, right now at least, but we should facilitate, probably similar to how
edopro does"; and the whole design should be AI-friendly.

Today the web server writes `runs/web/<ts>/game.jsonl` + `.ydk` files (`write_record` in
`src/decklight/play/session.py`): the seed plus our env's **option indices**. The weaknesses:
- the indices depend on our option ordering and the MAX_OPTIONS cut, so they break whenever the env changes;
- there is no engine, script or cdb version pinning, and a card script change can change a replay;
- the record is not self-describing (no schema, no format version).

## User decisions (2026-11-01)
- Shared games go to an **open dataset**: "I think GitHub for now until i figure out the huggingface dataset, that
  sounds very promising actually because we can have an open dataset. Hugging face might be better actually."
  So the target is a public Hugging Face dataset, and v1 submissions arrive through GitHub.
- Human vs human: "Part of the first app release id say."

## Decision

### 1. Match record v1: one self-describing JSON document per game
`format: "decklight.match/1"`, with a JSON Schema in `docs/design/schemas/match-record-v1.json`. On disk it is
`<id>.match.json.gz`. Fields:
- `id`: a UUIDv7 (time-ordered and globally unique, so merging records from many phones never collides);
  `created` (UTC, ISO 8601); `app` (product version).
- `engine`: the ygopro-core commit plus a hash of our patches; `cards`: the card-pack id and the sha256 of its
  manifest (scripts + cdb + strings, ADR 0009 / A5); `rules`: the duel flags, start LP, hand and draw counts.
- `kind`:
  - `"full"`: we ran the engine. Holds the seed and **every** response, so it replays exactly.
  - `"seat"`: a game we only saw from one seat, with no replay from the server. Holds the filtered messages that
    seat received plus its own responses. Good for stats and partial training; it cannot be replayed in the engine.
  Games on EDOPro servers are normally `full`: the server sends every client the replay at the end
  (`STOC_NEW_REPLAY`, an LZMA `.yrpX` holding a `.yrp` with `engine_seed`, the decks and every response,
  `edopro/gframe/generic_duel.cpp:576-609`), which converts losslessly.
- `engine_seed`: the core's 4 × u64 xoshiro state (what EDOPro replays store), each a **decimal string** (JS
  numbers lose integers above 2^53). Our own games also keep `seed`, the u64 that `seed_words` expands. `first`: who
  went first (RPS / choice).
- `decks`: per seat, main/extra/side code lists **in the order they were added to the core** (after any shuffle,
  as `.yrp` stores them), plus an optional name and a hash of the sorted deck for stats.
- `add_order`: the order the teams' cards were added to the core, when not [0, 1]. It changes the duel (NOTEBOOK
  2026-11-01 `.yrpX` import). Multirole does not record it, so the importer finds it by replaying both orders. `seats`: per seat a `type`:
  - `human`, with an anonymous `player` id: a random id per install, never a name or account;
  - `bot`, with `model` (bundle id + sha256) and `mode` (raw / search / honest);
  - `remote`, with the protocol (decklight / edopro).
- `mode` (practice, puzzle, lan, online, ...) and `assists`: which learning features were on, per seat (hints, win
  chance, search), so assisted games can be separated from unassisted ones in the data.
- `responses`: in order, `{seat, r: hex of the raw OCG response bytes, opt: our option index list or null, ms}`.
  Raw bytes are the engine-level truth: they are what EDOPro clients send and what EDOPro replays store, and they do
  not depend on our option ordering. `opt` is a cross-check and a convenience for training. Both directions already
  exist (`Decision.choose` gives bytes; `engine/inverse.py` maps bytes back to indices).
- `result`: winner, reason, turns, surrendered, disconnected.
- Optional derived sections, all regenerable and dropped on sharing unless asked for: `events` (canonical messages,
  the A1b `canon` form), `annotations` (hints shown, win chance per turn), `transcript` (the plain-text game,
  readable by people and LLMs).

The format is additive within v1: readers ignore unknown fields, and a breaking change becomes v2 with a converter.

### 2. Replays never rot
Card packs are archived forever (versioned GitHub release assets, A5), and a record pins its pack by sha256. The
replayer loads that pack, not the newest one. A record whose pack is missing still feeds stats but is flagged as
not replayable.

### 3. History and stats on the device
One file per game plus a SQLite index on the device: id, date, decks, opponent, mode, result, turns, assists and
shared flag. The history screen, stats (win rate per deck, matchup, going first or second) and review/what-if all
read from it. The laptop server writes the same records and index.

### 4. Sharing: opt-in, anonymous, open
- Off by default; it can be turned on per game or always. Only `full`/`seat` cores are shared: no derived sections,
  no names, no device data. The player id is the random install id.
- v1 channel (GitHub, per the user): the app exports the selected games as one zip (Android share sheet / file),
  and a GitHub issue form "Share games" takes the zip as an attachment. No tokens live in the app.
- Intake job on the laptop:
  - validate against the schema;
  - **replay every `full` record in the engine with its pinned pack and require the stored result**; records that
    disagree are rejected and logged (fail loud: a record we cannot reproduce must not reach training);
  - dedupe by id;
  - publish to the Hugging Face dataset (Parquet shards per month, with a dataset card).
- Later: direct upload, e.g. through a small Hugging Face Space that holds the write token. The record format does
  not change for that.
- **Data license: CC BY 4.0** (user, 2026-11-01: "Cc by 4.0").

### 5. Human vs human in the first release: the EDOPro protocol, which we already speak
The wire protocol between players is **EDOPro's LAN protocol** (TCP), not a new one. The Python pieces exist and
are tested (plan `docs/plans/2026-10-24-edopro-client.md`):
- `edopro/protocol.py`: framing and structs. The struct padding was checked against a real Multirole.
- `edopro/filter.py`: per-viewer filtering, a port of generic_duel.
- `edopro/host.py`: an EDOPro-compatible host; EDOPro clients join it. Today it has one remote seat plus a local
  bot.
- `edopro/client.py` + `play/stream.py` (`StreamView`): join any EDOPro/Multirole room and play from the filtered
  stream. Verified on a local Multirole: 5/5 duels, and 8/8 with custom cards (NOTEBOOK 2026-10-26).

Ported to Rust, those give the app every human-vs-human path through one protocol:
- **Join** a room on public EDOPro lobbies, a friend's or community Multirole (`scripts/multirole_local.sh` already
  builds one), or an EDOPro desktop host on the LAN. The server runs the engine and the scripts, so the app needs
  only the cdb/strings for display. A card newer than our pack shows as its code until the pack updates, and must
  never crash the app.
- **Host** a room on the phone: `host.py` extended to two remote seats. The other player joins from the app or from
  EDOPro desktop, over LAN, a phone hotspot (works with no internet, which suits travel) or Tailscale.
- So we "facilitate, similar to how edopro does" without running servers. Multirole is the server for online
  play, and the app hosts for offline play.

Rules:
- **Records**: as host, the app writes `full`. As client, it converts the server's `.yrpX` into `full`, and falls
  back to `seat` only if no replay arrives.
- **Fair play**: in human vs human, the learning features are off by default; the lobby shows both players'
  settings; `assists` records them. As a client there is no local engine anyway, so no search or win chance.
- **Protocol version**: servers reject a client whose version differs. `CLIENT_VERSION` must follow EDOPro
  releases; it goes in the card-pack manifest (A5), so a version bump is a pack update, not an app release.
- **Seats inside the app**: the in-app UI ↔ local server link stays our WebSocket protocol. Bots and outside AI
  agents are seats too: locally over WebSocket, or as EDOPro clients like WindBot (`scripts/edopro_join.py`). Every
  game, whoever plays it, produces the same record.
- **Open, before advertising public-lobby play**: whether ProjectIgnis is fine with third-party clients and bots on
  its public servers. Bots must say they are bots in their names (WindBot convention). Private rooms on friends'
  Multirole servers and LAN rooms do not depend on this.
- **Replays from Multirole (verified in its source, f258e00, `Room/State/Dueling.cpp` `SendReplay`)**: at the end of
  each duel it sends the replay with `SendToAll`, to both duelists and spectators. The replay holds the seed, both
  players' main and extra decks, and every response. The side deck is not in it. The replay goes out only after the
  duel, so nothing hidden leaks during play. A replay over the u16 packet limit (~64 KB) is not sent (the server
  sends a "replay too big" chat line instead); that game becomes a `seat` record.
- **Licenses**: EDOPro, ygopro-core, CardScripts, BabelCDB and Multirole are all AGPL-3.0, and so is decklight, so
  shipping the core and scripts in the app is consistent. The app must offer its source (it is public) and keep
  their notices. The dataset holds only card codes and moves (CC BY 4.0).
- **ProjectIgnis policy (checked 2026-11-01)**: the EDOPro README, the site and its FAQ say nothing on third-party
  clients or bots on their servers. The FAQ mentions bans appealed through their Discord. So there is no written
  rule either way: ask them on Discord before the app advertises their public lobbies.
- **Decisions (user, 2026-11-01)**:
  - Games against EDOPro players are shared anonymously like any other, with a line in our room notes ("games may
    be shared anonymously for research"): "I prefer the first option, we can put a line in our room."
  - Public lobbies: "let's no advertise the edopro's community lobbies, but let's support already, so it will just
    be turned off for now." The join code supports any Multirole lobby. Joining ProjectIgnis' public lobby list sits
    behind a setting that is **off by default** and hidden in the UI until they agree. LAN, hotspot, Tailscale and
    private Multirole servers (joined by address) are on.

## Alternatives rejected
- Keep option indices as the replay key: they break when the env's option list changes.
- EDOPro's `.yrpX` replay as our native format: it is binary, has no schema, no pack pinning, no seats/assists, and
  it is not friendly to datasets or AI. An export to `.yrpX` stays a possible interop feature.
- One shared store per player (accounts, a server we run): out of scope. The user does not want to run a server
  now.
- A new decklight-only wire protocol for human vs human: the EDOPro protocol already works (host and client are
  tested against Multirole), and it lets our players meet EDOPro players.
- Raw observations in records: large, and they depend on the encoder version; regenerating them from replays is
  the project's rule (ADR 0001).

## Consequences
- A2 writes match record v1 from the Rust server; `scripts/app/to_match_record.py` converts the existing
  `runs/web` records by replaying them in PyEnv. The Python server switches to v1 at A7.
- A2b becomes the Rust port of the EDOPro stack (protocol, filter, host with two remote seats, client, StreamView,
  `.yrpX` import). Oracles: the Python host and client (stream parity), and a local Multirole (`multirole_check`).
- New intake job (schema, replay check, dedupe, Hugging Face publish), with an ALERT on every rejected record.
- Pack archiving becomes a hard requirement of A5.
