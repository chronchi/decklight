# ADR 0011: One Rust product server, one place for new features

Date: 2026-11-02. Status: accepted (user's answers quoted below). Plan:
`docs/plans/2026-11-01-offline-app.md`. Branch: `offline-app`.

## Context
Two servers can play the product today: the Python web server (the oracle) and the Rust server (A2c, A3b, A4). The
Rust server is what the phone app, the browser and the desktop app will ship (ADR 0009, D2/D3); Python stays for
training, data and as the reference implementation. Once A7 merges the Rust work, every client will speak to the
same server, so a feature must exist there once. The user asked:

> Once we finish all of this, every new feature or extension we make will be available across all clients, right?

and answered the proposal to make Rust the only product server with:

> Yes

The Python web server still has features Rust does not (A4 part 2 left these open), so the switch cannot happen
before they are ported. The user, on porting them:

> We need to port them too

EDOPro clients only receive the EDOPro protocol (room duels), so learning features cannot reach them there. The user:

> Understandable

## Decision
1. **After A7, the Rust server is the only product server.** New features are implemented in the Rust server and
   `web/` once, and every client gets them: Android (Tauri), browser and desktop. The Python web server remains for
   training, data and as the test reference; it gets no new product features.
2. **Before A7, port everything the Python web server still has that Rust lacks:**
   - puzzles, `puzzle_solution`, primer and study;
   - record meta: `hints_used`, `undos`, `what_if`, and the review written to disk.

   This is milestone **A4b**, before A5 (weekly card packs).
3. **EDOPro clients get only what the EDOPro protocol carries** (room duels). Learning features stay off in rooms.

## Alternatives rejected
- **Keep both servers.** Every product feature would be written twice and every one would need its own parity gate;
  the drift is exactly what ADR 0009 removes. The Python server keeps its role as the oracle, not as a product.
- **Port only the features used by the current clients.** The phone app, browser and desktop are one client surface
  in practice; a feature missing in Rust is missing on every client after A7.

## Consequences
- New product work has a single implementation and a single web client; parity gates are only needed while both
  servers exist (until A7).
- A4b is a hard prerequisite for A7 and sits before A5 in the plan.
- `web/` gains no client-specific forks: Android, browser and desktop all load it.
- Rooms stay learning-free by protocol, not by a per-client switch.
