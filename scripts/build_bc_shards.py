"""Replay every demo record source into BC shards (see ygo_zero.model.bc_data).

uv run python scripts/build_bc_shards.py --out data/cache/bc_shards --jobs 2
"""

import argparse
from pathlib import Path

import numpy as np

from ygo_zero.model.bc_data import build_shards, eval_only_decks

SOURCES = {  # tag -> globs of record files (verified lines and their hand-variant expansions)
    # strict-d: retrain stage D strict re-verification of the lines strict_junk.py flagged (scripts/vast_stage_d.sh);
    # exclude the old records of those lines with --exclude '<glob>@data/cache/audit/strict_junk.json'
    "guides": ["data/demos/verified/*.jsonl", "data/demos/expanded/*.jsonl", "data/demos/guides-strict-d/*/*.jsonl"],
    "claude": ["data/demos/claude/verified/*.jsonl", "data/demos/claude/expanded*/*.jsonl",
               "data/demos/claude/strict-d/*/*.jsonl"],
    "muse": ["data/demos/planned*/verified/*.jsonl", "data/demos/planned*/expanded*/*.jsonl",
             "data/demos/planned*/improved/verified/*.jsonl", "data/demos/planned*/improved/expanded*/*.jsonl",
             "data/demos/planned*/strict-d/*/*.jsonl"],
    "games": ["data/demos/games/*.jsonl"],  # ADR 0004 L2-L4 (deck label = actor's deck)
    "gold": ["data/demos/gold/*.jsonl"],  # hand-played full games (scripts/hand_play.py; kinds gold / gold_pass)
    "search": ["data/demos/search/*.jsonl"],  # turn-1 search teacher (scripts/gen_search.py; kind search)
    "selfimit": ["data/demos/selfimit/*/*.jsonl"],  # turn-1 self-imitation (scripts/select_turn1_lines.py; kind selfimit)
    "teacher": ["data/demos/teacher/*.jsonl"],  # ygo-agent self-play (scripts/gen_teacher.py; kinds teacher / teacher_pass)
    "dagger": ["data/demos/dagger/*.jsonl"],  # DAgger on our policy's states (scripts/gen_dagger.py; kinds dagger / dagger_pass)
    "window": ["data/demos/window/*.jsonl"],  # mid-game search teacher (scripts/gen_window.py; kinds window / window_own)
    # self-play window search, one dir per labelling (scripts/relabel_window.py; kinds sw-<name> / sw-<name>_own)
    "sw": ["data/demos/sw-*/*.jsonl"],
    "turns": ["data/demos/turns/*/*.jsonl"],  # LLM-planned mid-game turns (scripts/plan_turns.py; kind turn_<bucket>)
}

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("data/cache/bc_shards"))
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--only", nargs="*", default=None, choices=sorted(SOURCES), help="build only these sources")
ap.add_argument("--glob", nargs="*", default=None,
                help="build only these record globs (one source 'custom'); kinds still come from the file paths")
ap.add_argument("--strict-labels", action="store_true",
                help="inferred labels weigh 0 except zone picks / identical copies (retrain plan v2)")
ap.add_argument("--exclude", nargs="*", default=[], metavar="GLOB@FILE",
                help="drop records listed in FILE from record files matching GLOB (retrain stage D: strict_junk.json "
                     "for lines replaced by strict re-verification, drops-*.final.jsonl for hand-trap drops)")
args = ap.parse_args()
sources = {"custom": args.glob} if args.glob else SOURCES
src = {
    tag: sorted(f for g in globs for f in Path(".").glob(g) if f.stat().st_size)
    for tag, globs in sources.items()
    if args.glob or args.only is None or tag in args.only
}
print({k: len(v) for k, v in src.items()}, "files")
print("samples:", build_shards(src, args.out, args.jobs, strict=args.strict_labels, exclude=args.exclude))
# newcards decks never enter training data (retrain plan v2): stop loud if any shard carries one
_new = eval_only_decks()[1]
_bad = sorted({str(d) for f in args.out.glob("*.npz") for d in np.unique(np.load(f)["deck"]) if str(d) in _new})
if _bad:
    print(f"ALERT build_bc_shards: newcards decks in shards: {_bad}", flush=True)
    raise SystemExit(3)
