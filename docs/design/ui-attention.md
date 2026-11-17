# One thing at a time: the duel screen's attention rules

Written 2026-11-12 after the user's phone test. The user: "the bottom [sheet] stays on the image while the card is being
revealed"; "Don't just fix this one, look for the principle behind it so everything is consistent". Earlier the same day:
the card actions were clipped under the hand, and the options slid up over a bot effect before it could be read.

All of these bugs have the same cause. Each surface decides on its own when to appear, so two of them end up on screen
together and one covers the other.

## The surfaces

| Layer | What | Examples (ids today) | Started by |
|---|---|---|---|
| 0 Board | the field, hands, bars, phase strip, log line | `#board`, `#hand`, bars | — |
| 1 Narration | game events the user should read: activation, reveal, target, negate, attack, chain resolve, turn recap | `#ov-cues`, `#ov-reveal`, `.recap` | the game |
| 2 Decision | what the user must answer: the prompt sheet, the card-actions dialog, zone picking on the board, RPS, first/second | `#sheet`, `#menu`, `.slot.zpick`, `#rps` | the game |
| 3 Lookup | what the user opened to read or change: card detail, keyword gloss, log/stats, game menu, settings, deck sheet, opponent's deck | `#detail`, `#kwpop`, `#logdrawer`, `#gamemenu`, `#settings`, `#decksheet`, `#primersheet`, `#reviewframe` | the user |
| 4 System | pause banner, connection/error toasts, game over, boot | `#pausebanner`, `#banner`, `#gameover`, `#puzzleresult`, `#boot` | the app |

## Rules

1. **One stage.** Layers 1, 2 and 3 never share the screen. At any moment at most one of them is visible: the *active*
   surface. The others wait. They are not shown dimmed underneath, because a dimmed surface still covers part of the board.
2. **What the user opened goes first.** A lookup (3) is on stage until the user closes it. Narration and decisions that
   arrive in the meantime queue up behind it. The bot's actions are held, just as Pause holds them.
3. **Read before deciding.** Narration (1) is shown before a decision (2). A prompt that arrives during narration waits.
   If narration starts while a decision is up, the decision steps aside fully (it slides out and becomes inert) and comes
   back when the narration ends. It keeps its state: the selection so far, scroll position and focus.
4. **Nothing is lost.** A surface that steps aside is restored exactly as it was. Focus returns to where it was before.
5. **Never stuck, never skipped.** Narration plays every scene at the chosen Pacing, even when a decision waits behind it.
   The fallback is a stall watchdog: only if no new scene starts for 4 s is the rest applied at once. (A flat 2.5 s
   ceiling from the prompt's arrival, the first version, skipped a whole bot turn at once.) Instant pacing means narration takes no time. The Pause button and "Pause after each bot action"
   hold the queue and never drop items from it.
6. **The active surface gets the space.** It is laid out against the full viewport, minus the bars. It is never clipped
   by the hand or by a sheet: a revealed card shows its whole text box at 360×780.
7. **System messages sit above everything but block nothing.** Layer 4 never hides the active surface's controls. Game
   over replaces the stage.
8. **One mechanism.** A single controller in `web/app.js` (the "stage") owns the queue and decides which surface is
   active. Every surface asks the stage to show itself and tells it when it is done. No surface sets its own z-index
   race or timeout to get around another. The z-index values follow the layer table and nothing else.

## Checks
`scripts/app/duel_ui_check.py` asserts the rules in mock mode:
- for every pair of surfaces that can collide, their visible boxes never overlap;
- the active surface's box is inside the viewport at 360×780 and 412×915;
- a surface that stepped aside comes back with its state;
- the queue always drains (no surface waits longer than its fallback).
