# Model card: decklight v0.3.0

## What it is
A Yu-Gi-Oh! TCG duelist for the EDOPro engine. Given any legal deck it plays both its own turns
and the opponent's turn (hand traps, chain responses). It has three parts, shipped as the release bundle
`decklight-models-v0.3.0.tar.gz`:

| part | file | size | role |
|---|---|---|---|
| policy **Y** | `data/cache/y_league/best.pt` | 9.0 MB | transformer over the board, hand, chain, the history the player can observe and the legal options; scores each legal option. The product since v0.3.0 (r1 → A → X → X2 → Y, [results.md](results.md#the-product-today)) |
| policy Y for the app | `data/cache/onnx/y/` | 31 MB | Y and value_p2 exported to ONNX (the app's Rust runtime; parity 9.5e-6 with PyTorch) |
| policy **r1** | `data/cache/forge_r1/best.pt` | 9.0 MB | the v0.2.0 product: P plus 16 archetype specialists distilled in. Fine-tunes and Forge packs start from it |
| policy **P** | `data/cache/P/best.pt` | 8.6 MB | the previous product policy (r1's base); kept for reproducing earlier results |
| value net **value_p2** | `data/cache/value_p2/best.pt` | 8.8 MB | P's network with a value head trained further; estimates the win chance of a position |
| aux heads | `data/cache/aux_p2/aux.pt` | 0.1 MB | life-point / card / turns-to-end estimates for the web client's reports (not used in play) |
| card embeddings | `data/cards/emb/*.npz`, `data/cards/lua_desc.npz` | 50 MB | per-card vectors from card text (Qwen3-0.6B) and script features, so unseen cards get a representation |

At play time, **honest search** runs on top of Y:
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
5. **From r1 to Y** (v0.3.0): league PPO with a second search bot among the opponents (A), PPO focused on the
   opponent's turn (X), a second Forge round of 25 specialists trained on the observable history (X2), and 80 more
   iterations of league PPO (Y). Every step passed a paired no-loss gate. See
   [results.md](results.md#the-product-today).

Card text and scripts come from Project Ignis' BabelCDB and CardScripts. No card art is used anywhere.

## How it was evaluated
See [results.md](results.md).
- Y: 0.630 [0.595, 0.664] against ygo-agent on its own 31 decks (744 games), +0.7 pp over X and X +2.3 pp over r1
  (paired, with search).
- r1: 0.624 [0.599, 0.648] (1,488 paired games), level with P; +11.0 pp [+7.3, +14.7] over P (both with search) on the
  11 archetypes added in r1.
- Search gains carry to deck lists and archetypes the policy never trained on.
- No human-strength evaluation exists yet.

## Limits and known weaknesses
- **Not shown to be superhuman.** The external benchmarks are two bots (ygo-agent and a search bot).
- Weaker going first than going second.
- Some hand-trap timing habits come from the teacher data: it fires Maxx "C" on its own turn (mostly Draw/Standby) in
  ~10% of games without search and ~16% with search (P and r1 alike).
- The value net was trained on P's games, not Y's: on the Forge archetypes it judges positions Y reaches more often
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
