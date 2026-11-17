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
#   2026-11-08 policy r1 -> X (window PPO, runs/window-ppo-20261108-0940 export-X): no-loss card vs P 0.516; product card
#              X + value_p2 0.624 vs ygo-agent, +2.3 pp [-1.3, +5.9] paired vs r1 on 744 of the 1,488 deals; X's own critic
#              -0.7 pp [-3.2, +1.9] vs value_p2, so value_p2 stays (product-x-20261108-2059, NOTEBOOK 2026-11-09 00:15;
#              user: "Let's go with option 1, it might be enough")
#   2026-11-11 policy X -> Y (cycle 1: 25 specialists distilled into X2, + 80 it league PPO; c1-league-20261111-0821
#              export-Y, observable-history net): segment gate FAIL on tsoj (+1.28 pp [-1.28, +4.08], need +3), no-loss vs
#              X 0.527 [0.509, 0.544], retention +7.01 pp, old archetypes -0.83 pp (c1-league-20261111-1057; prereg
#              amendment 8; user: "Let's adopt Y actually"). Search with Y and ONNX export are not yet validated.
PRODUCT_CKPT=data/cache/y_league/best.pt
PRODUCT_CKPT_DIR=data/cache/y_league  # what launchers ship to the box (--data)
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
