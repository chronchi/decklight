# Train it further: fine-tunes and the Forge

The bot plays every deck with one shared model. You can make it better in two ways: fine-tune it on one deck, or fold
many fine-tunes back into the shared model (the Forge). Both need a GPU; a 16 GB card is enough (~9 GB used).

Set up a source checkout first (README "Install", steps 1–2), then install the CUDA build of PyTorch (the default wheel
is CPU-only):
```bash
uv sync --group dev --group ml --group web
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Fine-tune on one deck
```bash
uv run --no-sync python scripts/finetune_deck.py --deck my.ydk --out runs/ft-my --hours 2
uv run --no-sync python scripts/web_play.py --ckpt runs/ft-my/policy/best.pt --value runs/ft-my/value/best.pt
```
- It continues the bot's self-play training with your deck in half the seats, and updates both the policy and the
  value head. It starts from r1, the Forge base (`--init` to change it). The product policy Y adds steps
  after r1 ([results.md](../model/results.md#the-product-today)) and is not yet a fine-tune base.
- 2 h is ~180 training iterations on an RTX 4060 Ti. On Blue-Eyes, 2 h gained about twice as much as 0.75 h (raw policy
  +7.9 pp vs +3.8 pp).
- Every fine-tune writes a bot bundle (a folder with `bot.toml`). Copy it into `bots/` to play it in the lobby, from
  EDOPro or in a room ([bots.md](bots.md)). `web_play.py` also lists fine-tunes under `runs/` (or `--models-dir`).
- **What to expect** (0.75 h fine-tunes, piloted against ygo-agent's other 30 benchmark decks): Branded 0.61 → 0.77
  (+16 pp, 95% CI +9 to +23), Blue-Eyes 0.37 → 0.44 (+8 pp, CI +1 to +15). The fine-tuned model is for that deck: use
  the base model for other decks. Details: [results.md](../model/results.md#fine-tuning-on-one-deck).
- Measure your own deck with `scripts/deck_card.py` (needs the benchmark setup in
  [results.md](../model/results.md#reproduce-the-benchmark)). It plays the base and fine-tuned models on the same deals
  against every other benchmark deck, in both seats, and prints the paired difference.

## The Forge: many specialists, one model
Fine-tuning helps one deck. The Forge folds many fine-tunes back into the shared model, so every deck gains without
picking a model per deck:
1. **Specialists:** a short fine-tune per archetype (`finetune_deck.py`), each saved as a **pack**: weights
   (safetensors), the deck lists, the training config and metrics (`scripts/forge.py pack`).
2. **Verify:** each pack must beat the current release on its own decks, on held-out deals (a sequential test).
3. **Distill:** the release learns every accepted pack's moves on its own decks (KL to the pack teacher), while staying
   close to itself elsewhere. The result is the next release.
4. **Gate:** the new release must keep the specialists' gain without losing anything on other decks or against
   ygo-agent.

Releases so far (raw policy):
- **r0** = the base model P + 5 specialists (Branded, Invoked, Radiant-Typhoon, Sacred Beasts, Atlantean-Mermail):
  +7 to +22 pp on those archetypes, no loss elsewhere.
- **r1** = r0 + 11 more: +16 pp on those 11 (85% of the specialists' gain kept), no loss on r0's 5, on unrelated lists
  or against ygo-agent. r1 was the product policy in v0.2.0. Details:
  [results.md](../model/results.md#the-forge).
- **Second round** (v0.3.0): 25 more specialists distilled into X2, on the history the player can observe; with league
  PPO on top it became Y, the product policy since v0.3.0
  ([results.md](../model/results.md#the-product-today)).

## Contribute a pack
Send deck lists, or train a pack on your own GPU (about 1 h on a consumer card) and submit it through the "Submit a
deck or pack" issue form. Packs are data only (no code runs). They are checked automatically and verified on
undisclosed deals before distillation. Releases credit their contributors (CC BY 4.0). See
[contributing-packs.md](contributing-packs.md).
