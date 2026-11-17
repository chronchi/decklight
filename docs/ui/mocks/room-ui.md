# Room UI (A2b-3): play a person, over LAN, hotspot, Tailscale or a Multirole server

Status: approved (user, 2026-11-02: "The UI looks good"); the open questions keep their defaults: tab "People", a
user-typed duelist name (default "Duelist"). Implementation follows this spec and the visual
mockup `docs/ui/mocks/room-ui.html` (same tokens as `web/styles.css`). Decisions it rests on: ADR 0010 §5.

## Principles
- Same night style as the lobby (sumi background, washi text, kincha accent; memory "web client night design"), WCAG
  2.2 AA (contrast ≥ 4.5:1 text, ≥ 3:1 marks, 44 px targets, focus visible, no colour-only state, reduced motion).
- One protocol, three verbs: **Host**, **Join**, **Nearby**. No accounts, no names beyond a duelist name the user
  types once (stored on the device; default "Duelist").
- People first: a friend's room is joined by **address** (`host:port`, a Tailscale name works) or picked from the LAN
  list. The public EDOPro lobby list does not exist in the UI until the hidden setting is on (ADR 0010: off by
  default, not advertised).
- Fair play is visible: in a room, learning features (hints, win chance, undo, what-if) are **off** and the room card
  says so. Nothing to configure.
- Sharing is honest: the room note "Games may be shared anonymously for research" is shown on our room card and sent
  as the EDOPro room note; the game is shared only if the user's app-wide sharing setting is on (opt-in, ADR 0010).

## Where it lives
The lobby tabs become **Play · People · Puzzles**. "Play" stays the bot duel. "People" holds the room flow; a game
against a person reuses the duel screen unchanged (A2b-3 builds the state frames from the received stream).

## Screens

### P1 People (tab)
```
decklight
[ Play ]  [ People ]  [ Puzzles 12 ]

Your deck              Mitsurugi            >     (same picker as Play)
Your name              Duelist              >

┌ Host a room ─────────────────────────────────┐
│ Others join you on this Wi-Fi, your hotspot   │
│ or Tailscale.                                 │
│                              [ Host ]         │
└───────────────────────────────────────────────┘

Join
  Address   [ 192.0.2.20:7911            ]
  Password  [ optional                     ]
                                   [ Join ]

Nearby rooms                              ↻
  Kai's room · 1/2 · TCG · 192.0.2.20    Join
  (none found: "No rooms on this network yet.")
```
- **Nearby** scans on open and on ↻ (EDOPro LAN discovery). Each row: room name, seats taken, rule, address; the
  whole row is the target (44 px). Rooms from EDOPro desktop show the same way.
- Address field accepts `host`, `host:port` (default port 7911) or a Multirole room link; invalid input gets an
  inline message under the field ("Use an address like 192.0.2.20 or name:7911"), never a toast only.
- Password is optional; shown as a normal text field with a "show" toggle off by default.

### P2 Hosting (waiting room)
```
Your room                                       Close
┌───────────────────────────────────────────────┐
│ Kai's room                       Waiting…  ◌  │
│ Join from decklight or EDOPro at:             │
│   192.0.2.20 : 7911            [ Copy ]     │
│   kai-phone : 7911 [ Copy ]   │
│ Rules  TCG · 8000 LP · Master Rule 5          │
│ Hints, win chance and undo are off in rooms.  │
│ Games may be shared anonymously for research. │
└───────────────────────────────────────────────┘

Seats
  You      Mitsurugi            Ready ✓
  Open     waiting for a player

                              [ Start duel ]  (disabled until both ready)
```
- Every address the device can be reached on (Wi-Fi, hotspot, Tailscale) is listed with a Copy button; the waiting
  state uses text ("Waiting…") plus a slow 2 s breathing ring that stops with `prefers-reduced-motion`.
- When a player joins: their row shows name, client ("decklight" / "EDOPro"), and "Choosing a deck" → "Ready ✓".
  The host's Start becomes enabled; the joiner sees "Waiting for the host to start".
- Close asks once ("Close the room? Kai will be disconnected.") when a player is in it.

### P3 Joining / joined
```
Kai's room                                      Leave
Connecting to 192.0.2.20:7911…           (spinner text, then:)
Seats
  Kai      host                  Ready ✓
  You      Mitsurugi            [ Ready ]
Hints, win chance and undo are off in rooms.
Games may be shared anonymously for research.
```
- Errors replace the spinner with a plain sentence and a Back button: "Wrong password", "That room is full",
  "Your app and the room use different EDOPro versions (room 41.0, app 40.1). Update the card pack.", "Your deck was
  refused: <card> is not allowed (banlist)", "Could not reach 192.0.2.20:7911".

### P4 Rock-paper-scissors and turn order
Reuses the duel screen's existing RPS/first prompts (the stream carries STOC_HAND_RESULT / TP).

### P5 In the duel
- The opponent's name replaces "decklight" in the header; the thinking ticker reads "Kai is choosing…".
- The game menu loses Hint / Undo / What-if (greyed with the reason "Off in rooms") and gains "Surrender".
- Connection lost: a non-blocking bar "Connection lost. Reconnecting…" (EDOPro has no reconnect: after 30 s it becomes
  "Kai left the duel" and the duel ends as a disconnect, recorded as such).

### P6 After the duel
The existing game-over sheet, plus: "Rematch" (sends the EDOPro rematch answer; both must accept) and "Back to
room". The record line says where the game went ("Saved · will be shared anonymously" or "Saved on this device").

## Settings (game menu → Settings → Online)
- **Your name** (also editable from P1).
- **Share my games anonymously** (app-wide, off by default; ADR 0010).
- Hidden until ProjectIgnis agrees: **Show public EDOPro lobbies** (off). When on, People gains a "Public" list read
  from the Multirole room listing. No code path reaches public servers while it is off.

## Server messages (WebSocket, in-app server ↔ web client; additive to the bot protocol)
Client → server:
- `{"t":"room_host","deck":…, "name":"Kai", "password":"" }`
- `{"t":"room_join","address":"192.0.2.20:7911","password":"","deck":…,"name":"Kai"}`
- `{"t":"room_scan"}` → `{"t":"rooms","rooms":[{"name","address","seats":[taken,total],"rule","client"}]}`
- `{"t":"room_ready","ready":true}`, `{"t":"room_start"}` (host only), `{"t":"room_leave"}`, `{"t":"rematch","yes":true}`
Server → client:
- `{"t":"room","role":"host|guest","state":"waiting|ready|starting|dueling|ended","name","addresses":[…],
  "rules":{…},"note":"Games may be shared anonymously for research","seats":[{"name","client","deck_name"|null,
  "ready","you":bool}],"error":null|"text"}`
- then the usual `state` frames for the duel (A2b-3: built from the received stream), `result`, and the record line.

### Added at implementation (A2b-3c, 2026-11-02)
- There were no RPS / turn-order prompts on the duel screen to reuse (bot duels pick "rps" server-side), so the
  room frame carries them: `"ask": null|"rps"|"first"|"rematch"` and `"hand_result": [ours, theirs]|null`;
  answers are `{"t":"room_rps","hand":1|2|3}` (1 scissors, 2 rock, 3 paper: EDOPro's encoding),
  `{"t":"room_first","first":true}`, `{"t":"rematch","yes":true}`.
- `{"t":"room_back"}` after a duel: the host's app rejoins its still-open room, a guest rejoins the same address.
- Room frame extras: `"state"` also takes `"connecting"`; `"server"` (`host:port`), `"password"` (bool),
  `"rules"` `{text, start_lp, …}` (from STOC_JOIN_GAME), seat `"pos"` and `"host"`; `"field":"address"` with a
  `"closed"` state for an address the server cannot parse.
- In-duel `state` frames carry `"room": {"opponent", "role"}`; the ticker frame's `thinking` has `"who"`.
- Join addresses: `host`, `host:port`, `[v6]:port`, optionally `#<room id>` for a Multirole room.
- A guest whose host never answers (a full room: our host accepts only between duels) gets "That room is full or
  has closed" after 10 s.
- Not done: the seat's `"client"` (decklight / EDOPro) is always null: the protocol does not tell them apart.

## Accessibility checklist (measured at implementation)
- Tab order: tabs → deck → name → Host → address → password → Join → Nearby rows.
- `aria-live="polite"` on the room status line (joins, ready, errors); seats are a list with text states (no colour
  only); every button ≥ 44×44 px; contrast measured with the existing token pairs; reduced motion stops the ring.
