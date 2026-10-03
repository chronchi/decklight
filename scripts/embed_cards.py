"""Embed card texts with Qwen3-Embedding-0.6B -> data/cards/emb/<tag>.npz (see ygo_zero.cards.embed).

uv run python scripts/embed_cards.py --dim 512            # all cards
uv run python scripts/embed_cards.py --limit 200 --tag smoke
"""

import argparse
from pathlib import Path

from ygo_zero.cards.embed import build

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=512)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--out-dir", type=Path, default=None)
args = ap.parse_args()
build(dim=args.dim, batch=args.batch, limit=args.limit, tag=args.tag, out_dir=args.out_dir)
