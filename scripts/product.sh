# The product: the one place that names the current best checkpoint, value net and search settings.
# Source it (`. scripts/product.sh`) in jobs that play "the product"; job scripts of past runs keep their own paths as
# a record of what ran. History:
#   2026-10-14 P + hppo + value_p   0.606 vs ygo-agent (guardfix-20261014-1704)
#   2026-10-16 P + hppo + value_p2  0.629 vs ygo-agent, +2.0 pp paired over value_p (value-p2-20261015-2044; user:
#              "let's now just register the new value net as the default")
#   2026-10-21 + --ho 1 (honest override)  0.623 vs ygo-agent, -0.6 pp (SE 1.2) vs without, own-turn Maxx games
#              23% -> 16%, overrides 0.54 -> 0.10 (full-p-ho-20261021-1204; gate FAIL by 0.6 pp, user: "let's switch
#              to ho anyways")
#   2026-10-31 policy P -> Forge release r1 (P + 16 archetype specialists distilled): 0.624 vs ygo-agent, +0.1 pp
#              (SE 1.27) vs P on the 1,488 deals, +11.0 pp [+7.3, +14.7] on the 11 Forge archetypes (full-r1-20261030-1609
#              + full-p-fresh-20261031-0608; gate PASS, NOTEBOOK 2026-10-31 15:00; user: "Yes, make the changes")
PRODUCT_CKPT=data/cache/forge_r1/best.pt
PRODUCT_VALUE=data/cache/value_p2/best.pt
PRODUCT_VALUE_DIR=data/cache/value_p2  # what launchers ship to the box (--data)
# search_match args of the product's search (honest k4 s4 h1, det-tries 32, skip, conf-skip 0.9, ho)
PRODUCT_SEARCH="--honest 1 --det-tries 32 --det-fallback skip --horizon 1 --k 4 --samples 4 --conf-skip 0.9 --ho 1"
# Custom cards: the product reads these env vars when set (merged npz files covering the extra
# cards; see scripts/embed_cards.py --only-missing). Defaults = the stock files above.
#   YGO_EXTRA_CARDS : extra card dirs for the engine/DBs (engine/resources.py)
#   YGO_EMB         : merged card embeddings (search/product.py, search_match --emb, edopro host)
#   YGO_STRINGS     : merged effect-string table (model/policy_v3.py)
#   YGO_LUA         : merged lua_desc table (model/policy_v3.py)
