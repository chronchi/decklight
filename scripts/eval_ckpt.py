"""Evaluate one RL checkpoint concurrently with training (see train_rl --eval-mode async).

Appends one JSON record to --out (the same record train_rl logs to metrics.jsonl)
and prints the same ``[eval it N] ...`` line. Exit 2 when a section has n == 0,
non-zero on any other failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _effective_cores() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 4
    return max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True, help="frozen BC checkpoint (the eval baseline)")
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--eval-names", required=True, help="comma-separated pool deck names (the fixed eval decks)")
    ap.add_argument("--gauntlet-decks", type=Path, default=ROOT / "decks/gauntlet")
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--ga-seeds", type=int, default=1)
    ap.add_argument("--ga-opps", type=int, default=4)
    ap.add_argument("--no-ygoagent", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=max(1, _effective_cores() // 2))
    ap.add_argument("--out", type=Path, required=True, help="jsonl file the record is appended to")
    ap.add_argument("--games-dir", type=Path, default=None, help="save every game here (rl.evaluate.save_games)")
    args = ap.parse_args()

    from ygo_zero.rl.evaluate import format_eval_line, run_eval

    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    ev_names = [n for n in args.eval_names.split(",") if n]
    if not ev_names:
        print("eval_ckpt: empty --eval-names", flush=True)
        raise SystemExit(1)
    res = run_eval(args.ckpt, args.init, args.emb, ev_names, hold, args.gauntlet_decks,
                   args.eval_seeds, args.ga_seeds, args.ga_opps, args.no_ygoagent,
                   args.device, args.threads, it=args.iter, games_dir=args.games_dir)  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(format_eval_line(res), flush=True)
    if any(v.get("n", 0) == 0 for v in res["eval"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())
