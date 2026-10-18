# Web client: pacing and learning features (plan, 2026-10-17)

The user's request, quoted: "things happen too fast with the bot, for example master duel has a bunch of cues right for the
battle, for activating effects, for chains, which makes it easier to follow the game." After Claude proposed the list below:
"I really like the pacing ... I liked all other suggestions as well so everything you said could be implemented".

## 1. Event cues and pacing (the base for everything else)
The client queues `log` events and plays each one as a short scene, applying the board only as each scene finishes;
today a whole bot turn arrives as snapshots and pieces jump into place. The same timeline drives `scripts/game_video.py`.
- **Activations:** the card rises to the centre (~1 s) with its name and the effect text, over a dimmed board.
  Opponent activations hold slightly longer.
- **Chains:** links stack visibly as Chain 1/2/3, then resolve one at a time from the top down. A negated link is
  shown crossed out (the user's Ash confusion).
- **Summons:** the card slides into its zone from where it came (hand, Extra Deck or GY), with the summon type shown
  in small text.
- **Targets:** a thin line from the source card to its target.
- **Battle:** a brush-stroke line from attacker to target, the ATK/DEF numbers facing each other, the destroyed card
  fading into the GY, damage floating up and the LP counter ticking down.
- **Turns and phases:** a quiet banner such as "Turn 3 · Wiccanthrope · Battle Phase".
- **Card movement:** cards sent to the GY, banished or added to hand move to their pile instead of vanishing.
- **Controls:**
  - speed 1× / 2× / instant;
  - "pause after each bot action" (tap to step);
  - a "what happened" summary at the start of your turn, each line tappable for card details.
- **Night design** (memory web-client-night-design): no flashes, no white, no screen shake. Slow fades and slides with
  one accent line.
- **Data the server must add:** chain-link numbers, the location each summon came from, attack targets, LP deltas.
- **Who builds it:** the cue design and client work go to an Opus subagent (the user prefers Claude for UI). Muse does
  the server-side event data.

## 2. Learning features (in priority order)
1. **Win-chance bar** from value_p2, updated after every action. Calibrate before showing it: a shown 60% must win about
   60% of the time, measured on recorded games.
2. **Post-game review** (like chess.com): each of your decisions next to the agent's choice and the win chance lost.
   Decisions are marked good play / mistake / blunder / missed response, and tapping one opens the board at that point.
3. **Missed-response alerts:** windows where you had a legal hand trap or quick effect, the agent would have used it,
   and auto-pass skipped the window.
4. **"Ask the bot" hints:** the agent's top 3 options with its confidence in each. Practice mode only.
5. **What-if:** from any earlier decision in the review, choose another option and the bot plays the game out from there.
6. **Rules in context:** tappable keywords (chain, spell speed, quick effect, once per turn, damage step, banish vs
   send) and a one-line rule note attached to cues.
7. **Opponent deck primer:** archetype, key cards and which of your hand traps hit them, taken from bot games.
8. **Puzzles** from recorded positions, checked by search, including combo practice for your own deck's openings.
9. **Undo** in practice mode.

**Rejected:**
- written "why" explanations from an LLM (paid per request and can invent reasons; win-chance swings and the agent's
  choices explain plays without that risk);
- sound on by default (the user plays at night).

## Order
Cues first, then items 1–3, then 4–6, then 7–9. The user's further comments (due 2026-10-17) get folded in first.

## 3. Accessibility (the user, 2026-10-17: "we have to implement this following accessibility standards as well, that's important in general")
The whole client targets WCAG 2.2 AA:
- Contrast: 4.5:1 for normal text; 3:1 for large text and for meaningful marks. Measured by script, not by eye.
- No meaning carried by colour alone.
- Touch targets at least 24 px, 44 px for primary buttons.
- `prefers-reduced-motion` is honoured, and the pause and speed controls stay.
- Accessible names on controls and cards; an `aria-live` region announces cues and prompts.
- Dialogs follow the dialog pattern: focus moves in and returns, and Escape closes.
- Everything works by keyboard, with a visible focus ring, and stays usable at 200% zoom.
- axe-core runs in the screenshot script and fails on serious or critical issues.

The ratios are met with desaturated colours, never with brightness or glow (the night design comes first).

Negation cues (the user: "for negation it's too faint"):
- a shape, a label naming the cause, and motion together;
- at least 3:1 contrast;
- the negated link stays marked until the chain ends;
- held about 1.6 s at 1×.
Cue lines are about 1.5–2× heavier, at least 2 px at 412 px width.

## 4. Decisions, 2026-10-17 (after the stats mockups in docs/mockups/stats/)
- The user, on the mockups: "I think a mixture of a to c actually, I like the option of the duel log, stats and other
  things at the top".
  - The log sheet gets tabs at the top (Log | Stats | Cards), taken from C.
  - The Stats tab mixes C's tiles, B's LP chart and key moments, and A's ledger table.
- Win chance gets a "Show win chance" switch in the game menu, off by default. While it's off, the chance appears
  nowhere: no bar, no chart line, no percentages in key moments or the review. It ships only after a calibration check
  on recorded games.
- Order confirmed ("Yeah order is right"):
  1. stats and tabs;
  2. the win-chance switch, after calibration;
  3. post-game review and missed-response alerts;
  4. hints and what-if;
  5. rules, primer, puzzles and undo.
- Delegation (the user): "delegate as much possible to muse and then verify yourself the work, only delegate to opus
  visual tasks but try to delegate more to muse". When a mockup already fixes the design, muse implements it and Claude
  verifies the screenshots.
