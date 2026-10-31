# Model card: decklight v0.2.0

## What it is
A Yu-Gi-Oh! TCG duelist for the EDOPro engine. Given any legal deck it plays both its own turns
and the opponent's turn (hand traps, chain responses). It has three parts, shipped as the release bundle
`decklight-models-v0.2.0.tar.gz`:

| part | file | size | role |
|---|---|---|---|
| policy **r1** | `data/cache/forge_r1/best.pt` | 9.0 MB | transformer over the board, hand, chain, history and legal options; scores each legal option. The Forge release: P plus 16 archetype specialists distilled in |
| policy **P** | `data/cache/P/best.pt` | 8.6 MB | the previous product policy (r1's base); kept for reproducing earlier results |
| value net **value_p2** | `data/cache/value_p2/best.pt` | 8.8 MB | P's network with a value head trained further; estimates the win chance of a position |
| aux heads | `data/cache/aux_p2/aux.pt` | 0.1 MB | life-point / card / turns-to-end estimates for the web client's reports (not used in play) |
| card embeddings | `data/cards/emb/*.npz`, `data/cards/lua_desc.npz` | 50 MB | per-card vectors from card text (Qwen3-0.6B) and script features, so unseen cards get a representation |

At play time, **honest search** runs on top of r1:
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
3. **Value-only RL** for value_p2, with the policy (P) frozen.
4. **The Forge** (r1): per-archetype fine-tunes of the current release (5 from P → r0, then 11 more from r0 → r1), distilled back into one policy by
   behaviour cloning (KL to each specialist on its own decks, KL to the release itself on ~140 other decks). See
   [results.md](results.md#the-forge).

Card text and scripts come from Project Ignis' BabelCDB and CardScripts. No card art is used anywhere.

## How it was evaluated
See [results.md](results.md).
- 0.624 [0.599, 0.648] against ygo-agent on its own 31 decks (1,488 paired games), level with P (+0.1 pp paired).
- +11.0 pp [+7.3, +14.7] over P (both with search) on the 11 archetypes added in r1.
- Search gains carry to deck lists and archetypes the policy never trained on.
- No human-strength evaluation exists yet.

## Limits and known weaknesses
- **Not shown to be superhuman.** The only external benchmark is one RL agent.
- Weaker going first than going second.
- Some hand-trap timing habits come from the teacher data: it fires Maxx "C" on its own turn (mostly Draw/Standby) in
  ~10% of games without search and ~16% with search (P and r1 alike).
- The value net was trained on P's games, not r1's: on the Forge archetypes it judges positions r1 reaches more often
  than P did.
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
