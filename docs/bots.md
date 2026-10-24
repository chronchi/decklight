# Bot bundles

A bot is one folder: `bot.toml` plus the files it names. No env vars, no CLI
flags at play time — every front end (EDOPro host, web lobby, network client)
resolves a bundle the same way.

## The `bot.toml` format

```toml
name = "wiccan-mine"                       # unique across bots/ + config/bots/
description = "Fine-tuned on Wiccanthrope." # shown in --list-bots and the lobby
agent = "product"                          # random | raw | product
deck = "deck.ydk"                          # bundle-relative, repo-relative, or absolute
ckpt = "../policy/best.pt"                 # optional (raw, product)
value = "../value/best.pt"                 # optional (product)
emb = "../emb.npz"                         # optional (merged card embeddings)
cards = ["../cards"]                       # optional custom card dirs
device = "cpu"                             # optional
seed = 0                                   # optional RNG seed (0 = random)
```

Rules:

- `path` may be a dir holding `bot.toml`, or a bare `.toml` (the shipped
  examples in `config/bots/` are bare files). Relative paths resolve against
  the bundle dir first, then the repo root, and may point outside the bundle
  (the shipped bots reference the stock checkpoints in place).
- `cards` entries are laid out like `YGO_EXTRA_CARDS` dirs: `*.cdb` at the top,
  Lua scripts under `script/`. A bundle's engine is the official card set plus
  exactly these dirs, so two bundles with clashing custom codes never share an
  engine (the web server caches one engine per card set, created on first use).
- Loading validates everything and fails with one message listing every
  problem: files exist, the deck parses, every deck code is in the bundle's
  card set, every deck code has an embedding row when the agent reads a model
  (or `emb` is set), and the agent name is registered.

## Three ways to play a bot

**1. EDOPro host** (you join the bot from EDOPro):

```bash
uv run python scripts/edopro_host.py --bot wiccan-mine --port 7911
```

`--bot` takes a bundle name or path. Deck, agent, checkpoints and engine
(custom cards) all come from the bundle. `--deck/--opponent/--ckpt/--value`
keep working for the old flag-based setup. With `agent = "product"` the full
search runs; with `raw` the policy plays through the host engine; with
`random` a uniform random player.

**2. Web lobby** (phone or desktop browser):

Bundles in `bots/` and `config/bots/` appear in the lobby next to the base
model and the fine-tunes. The duel runs on the bundle's engine, so the human
may also play the bundle's custom cards; legality is checked against the
bundle's card set and embeddings.

**3. Network client / WindBot serve** (the bot joins a Multirole room):

```bash
uv run python scripts/edopro_join.py Name=bot Deck=wiccan-mine Host=127.0.0.1 Port=7911
uv run python scripts/edopro_join.py --bot path/to/bot --host 127.0.0.1 --port 7911
uv run python scripts/edopro_join.py --list-bots
```

`Deck=` is a bundle name, a bundle path, or a `.ydk` path (a bare deck gets an
on-the-fly random bundle). `--agent/--ckpt/--emb/--device/--seed` override the
bundle's files. Only `random` and `raw` run here: a client never sees the
opponent's hidden choices, and the product search needs the true engine state,
so it stays on the host and web sides. `--serve PORT` exposes the
WindBot `ServerMode` HTTP endpoint; bundles are re-discovered on each request,
so a bundle dropped into `bots/` works without restart.

`--bot <name|path>` in `edopro_host.py`, `Deck=<name>` in `edopro_join.py`, and
lobby models in `web_play.py` all resolve the same bundle dirs: `bots/` (your
bundles) plus `config/bots/` (shipped examples).

## Fine-tune output

`finetune_deck.py` writes `<out>/bundle/bot.toml` at the end (with a copy of
the `.ydk`, `ckpt`/`value` pointing at `../policy/best.pt` and
`../value/best.pt`, the embeddings npz actually used, and the custom card
dirs actually used via `--extra-cards`), validates it, and prints:

```text
bot bundle: runs/ft-my/bundle (copy it into bots/ to play it anywhere)
```

## Custom cards and embeddings

A bundle's `cards` dirs work with no env vars. The old `YGO_EXTRA_CARDS` /
`YGO_EMB` env vars stay as a fallback for the flag-based commands.

The policy reads card-text embeddings. A stock `data/cards/emb/*.npz` covers
the official cards; custom cards need a merged npz that appends them:

```bash
# 1. check the custom cards play (script errors, activations, summons):
YGO_EXTRA_CARDS=data/cache/custom/zedja uv run python scripts/check_custom.py \
    --deck decks/custom/wiccanthrope.ydk --games 20
# 2. embed what is missing from the stock npz and write the merged file:
uv run python scripts/embed_cards.py --only-missing \
    --base data/cards/emb/qwen3-0.6b-d512.npz --out /tmp/merged.npz
```

Point the bundle's `emb` at the merged file. Validation refuses the bundle
when any deck code lacks an embedding row.
