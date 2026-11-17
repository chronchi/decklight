#!/usr/bin/env python3
"""Export a train_rl run's last.pt into the product layout (tsoj PPO step 3; reuses finetune_deck.export).

    uv run python scripts/tsoj/export_rl.py RL_OUT DEST [--init data/cache/forge_r1/best.pt] [--value-init ...] \
        [--ckpt RL_OUT/snap-0020.pt] [--value-only]

Writes DEST/policy/best.pt and DEST/value/best.pt; exits non-zero (finetune_deck invariants) when the policy equals
the init exactly (with --value-only: when it differs from the init at all), the value head equals value_p2's, or the files do not load through the product loaders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from finetune_deck import export, fail, load_state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rl_out", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--init", type=Path, default=ROOT / "data/cache/forge_r1/best.pt")
    ap.add_argument("--value-init", type=Path, default=ROOT / "data/cache/value_p2/best.pt")
    ap.add_argument("--ckpt", type=Path, default=None, help="a --save-every snapshot instead of RL_OUT/last.pt")
    ap.add_argument("--value-only", action="store_true", help="a value-head-only run: the policy must equal --init")
    a = ap.parse_args()
    last = a.ckpt or a.rl_out / "last.pt"
    if not last.is_file():
        fail(f"{last} missing (train_rl wrote no checkpoint)")
    pol, val = export(load_state(last), a.dest, a.init, a.value_init, a.value_only)
    print(f"exported {last} -> {pol}, {val}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
