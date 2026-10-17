"""Embed card texts with Qwen3-Embedding-0.6B -> data/cards/emb/<tag>.npz (see ygo_zero.cards.embed).

uv run python scripts/embed_cards.py --dim 512            # all cards
uv run python scripts/embed_cards.py --limit 200 --tag smoke
uv run python scripts/embed_cards.py --strings --dim 512   # card str1..16 + system strings (ygo_zero.cards.effect_strings)

Incremental (custom cards: YGO_EXTRA_CARDS set, base npz shipped to the box):

uv run python scripts/embed_cards.py --only-missing --base data/cards/emb/qwen3-0.6b-d512.npz --out /tmp/merged.npz
uv run python scripts/embed_cards.py --only-missing --strings --base data/cards/emb/strings-qwen3-0.6b-d512.npz --out /tmp/merged.npz
uv run python scripts/embed_cards.py --only-missing --dry-run --base ... --out /tmp/x.npz   # no model load
"""

import argparse
import sys
from pathlib import Path

from ygo_zero.cards.effect_strings import append_string_embeddings, build_strings, missing_string_keys
from ygo_zero.cards.embed import append_card_embeddings, build, missing_card_codes

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=512)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--out-dir", type=Path, default=None)
ap.add_argument("--out", type=Path, default=None, help="--only-missing: merged output npz")
ap.add_argument("--strings", action="store_true", help="embed effect strings instead of card texts")
ap.add_argument("--only-missing", action="store_true",
                help="embed only codes/keys absent from --base and write the merged npz to --out")
ap.add_argument("--base", type=Path, default=None, help="--only-missing: base npz to extend")
ap.add_argument("--dry-run", action="store_true",
                help="--only-missing: list what would be embedded (count and names), no model load")
args = ap.parse_args()

if args.only_missing:
    if args.base is None or args.out is None:
        ap.error("--only-missing needs --base and --out")
    if args.strings:
        missing, texts = missing_string_keys(args.base)
        print(f"{len(missing)} string keys missing from {args.base}", flush=True)
        for k in missing:
            print(f"  {k} {texts[k][:120]}", flush=True)
        if not args.dry_run:
            append_string_embeddings(args.base, args.out, dim=args.dim, batch=args.batch)
    else:
        missing, names = missing_card_codes(args.base)
        print(f"{len(missing)} card codes missing from {args.base}", flush=True)
        for c in missing:
            print(f"  {c} {names[c]}", flush=True)
        if not args.dry_run:
            append_card_embeddings(args.base, args.out, dim=args.dim, batch=args.batch)
    sys.exit(0)
if args.dry_run:
    ap.error("--dry-run needs --only-missing")
if args.out is not None:
    ap.error("--out needs --only-missing")
if args.strings:
    build_strings(dim=args.dim, batch=args.batch, limit=args.limit, tag=args.tag, out_dir=args.out_dir)
else:
    build(dim=args.dim, batch=args.batch, limit=args.limit, tag=args.tag, out_dir=args.out_dir)
