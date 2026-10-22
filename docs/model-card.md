# Model card: Decklight v0.1

## What it is
A Yu-Gi-Oh! TCG duelist for the EDOPro engine. Given any legal deck it plays both its own turns
and the opponent's turn (hand traps, chain responses). It has three parts, shipped as the release bundle
`decklight-models-v0.1.tar.gz`:

| part | file | size | role |
|---|---|---|---|
| policy **P** | `data/cache/P/best.pt` | 8.6 MB | transformer over the board, hand, chain, history and legal options; scores each legal option |
| value net **value_p2** | `data/cache/value_p2/best.pt` | 8.8 MB | P's network with a value head trained further; estimates the win chance of a position |
| aux heads | `data/cache/aux_p2/aux.pt` | 0.1 MB | life-point / card / turns-to-end estimates for the web client's reports (not used in play) |
| card embeddings | `data/cards/emb/*.npz`, `data/cards/lua_desc.npz` | 50 MB | per-card vectors from card text (Qwen3-0.6B) and script features, so unseen cards get a representation |

At play time, **honest search** runs on top of P:
- it samples worlds consistent only with what the player has seen;
- it plays each of the policy's top 4 moves to the end of the turn;
- it scores the end positions with value_p2;
- it replaces the policy's move only when the gap is statistically significant (the honest override, `--ho`).

Settings: `scripts/product.sh`.

## How it was trained
1. **Behaviour cloning** on:
   - ygo-agent's self-play games (most of the labels);
   - verified combo lines written from public deck guides with an LLM's help, then checked move by move in the
     engine;
   - games where an LLM played turns.
2. **PPO self-play** on a pool of ~180 deck lists, KL-anchored to the BC policy (Stage P).
3. **Value-only RL** for value_p2, with the policy frozen.

Card text and scripts come from Project Ignis' BabelCDB and CardScripts. No card art is used anywhere.

## How it was evaluated
See [results.md](results.md).
- 0.623 [0.598, 0.647] against ygo-agent on its own 31 decks (1,488 paired games).
- Search gains carry to deck lists and archetypes the policy never trained on.
- No human-strength evaluation exists yet.

## Limits and known weaknesses
- **Not shown to be superhuman.** The only external benchmark is one RL agent.
- Weaker going first than going second.
- Some hand-trap timing habits come from the teacher data: it fires Maxx "C" on its own turn (mostly Draw/Standby) in
  ~10% of games without search and ~16% with search.
- Decks far from the training pool (new mechanics, heavy counters) are less reliable. Card embeddings give unseen cards
  a representation, but the policy has not seen how they are played.
- Search costs ~10–15 s per searched decision on one laptop CPU core (much less on a GPU). A full game on a laptop
  takes 10+ minutes.
- One game per match: no side decking or match strategy.

## Intended use
Practice and study (the web client's hints, primers and puzzles), and research on game-playing agents under hidden
information. Not for online play against people without their consent.

## License
AGPL-3.0, like the code (see LICENSE and NOTICE).
