"""Embed card texts with Qwen3-Embedding-0.6B -> data/cards/emb/<tag>.npz (see ygo_zero.cards.embed).

uv run python scripts/embed_cards.py --dim 512            # all cards
uv run python scripts/embed_cards.py --limit 200 --tag smoke
uv run python scripts/embed_cards.py --strings --dim 512   # card str1..16 + system strings (ygo_zero.cards.effect_strings)
"""

import argparse
from pathlib import Path

from ygo_zero.cards.effect_strings import build_strings
from ygo_zero.cards.embed import build

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=512)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--out-dir", type=Path, default=None)
ap.add_argument("--strings", action="store_true", help="embed effect strings instead of card texts")
args = ap.parse_args()
if args.strings:
    build_strings(dim=args.dim, batch=args.batch, limit=args.limit, tag=args.tag, out_dir=args.out_dir)
else:
    build(dim=args.dim, batch=args.batch, limit=args.limit, tag=args.tag, out_dir=args.out_dir)
