# Tab icons (Play, Puzzles, Decks, Ask)

Drawn for the bottom tab bar (user, 2026-11-14: "create 4 icons that follow our brand guidelines ... using SVG so it's
easily editable"). Built from the decklight mark (`docs/ui/brand/decklight`, `web/assets/decklight-icon.svg`): cards with
rounded corners, the gold line of play with its nodes.

- 24×24 grid, 1.2 stroke (was 1.6; user 2026-11-15 found it too heavy and picked 1.2 from 1.35/1.2/1.0 mocks), the `.node` r 1.65, round caps and joins, `currentColor`: they take the tab's text colour (`--muted` at rest,
  `--text` when selected).
- One accent per icon: the `.node` circle. The app colours it `--accent` on the selected tab
  (`.sel .node { fill: var(--accent) }`), so the selected tab is shown by colour, the label colour and the bar above it
  (never by colour alone, WCAG 1.4.1).
- Shapes that sit over other strokes are filled with `var(--icon-fill, #141310)` (the bar background) so lines do not
  cross; set `--icon-fill` where the bar background differs.
- Inline them (fetch or paste) so CSS can reach `.node`; as `<img>` they render in one colour.
