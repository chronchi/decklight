# bots/

Your bot bundles live here: one folder per bot with a `bot.toml` (or a bare
`.toml`). See `docs/bots.md` for the format. `finetune_deck.py` prints a bundle
path at the end of every run — copy it here to play it anywhere
(`edopro_host.py --bot <name>`, `edopro_join.py Deck=<name>`, the web lobby).

This dir is git-ignored (except this file); the shipped examples live in
`config/bots/`.
