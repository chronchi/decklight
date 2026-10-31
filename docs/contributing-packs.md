# Contributing Forge packs

There are two ways to help the bot pilot your favourite deck:

**(A) Send deck lists.** Open a [pack submission issue](https://github.com/chronchi/decklight/issues/new?template=submit-pack.yml)
with kind "Deck lists only" and paste each `.ydk`, with its source (event, placing,
date) or `casual`. The maintainers review deck lists by hand.

**(B) Train a pack on your GPU.** You need a CUDA GPU (~9 GB used; about 1 h on a
consumer card) and the release bundle (`bash scripts/fetch_release.sh`, v0.2.0 or later), which holds the Forge base
release r1 in `data/cache/forge_r1/`. Fine-tune from r1:

```bash
uv run python scripts/finetune_deck.py --deck my.ydk --out runs/ft-my --hours 1 --init data/cache/forge_r1/best.pt
uv run python scripts/forge.py pack --ft runs/ft-my --release data/cache/forge_r1/best.pt \
    --release-name r1 --archetype <a> --train-decks decks/*.ydk \
    --heldout-decks heldout/*.ydk --out data/forge/packs/<a> --contributor "<name>"
uv run python scripts/forge.py validate data/forge/packs/<a> --release data/cache/forge_r1/best.pt
```

Then write the file to submit (`<pack_id>.tar.gz`, named after the pack's weights hash):

```bash
uv run python scripts/forge.py archive data/forge/packs/<a> --out-dir .
```

Then attach that `.tar.gz` to the issue form.

**What happens next:** a maintainer adds the `validate` label; a bot checks the pack's
format and comments the result. Valid packs are then verified by the maintainers on
their own undisclosed seeds: a pack is accepted only if it beats the current release
on its decks. Accepted packs are distilled into the next release, which credits the
contributor (see `scripts/forge.py credits` and `NOTICE`).

**Licence:** packs and deck lists are licensed CC BY 4.0 by their contributors; the
release itself is AGPL-3.0 (CC BY 4.0 material may be included in GPLv3-family works).

**Safety:** packs are data only (safetensors weights, deck lists, JSON); nothing in a
pack runs. The checker never unpickles uploads and validates names, paths and sizes
before reading anything else.
