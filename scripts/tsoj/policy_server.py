#!/usr/bin/env python3
"""The PPO learner's policy server for remote clients (decklight.rl.remote; NOTEBOOK 2026-11-07 tsoj PPO step 2).

    uv run python -u scripts/tsoj/policy_server.py --sock /tmp/p.sock --weights DIR/weights.pt \
        [--init data/cache/forge_r1/best.pt] [--emb ...] [--device cuda] [--ready FILE]

Without a weights file, `--init` is published there as version 0 (value head from `--value-init` when given, as
train_rl does), so actors can start before the learner. `train_rl.py --ext-dir` overwrites it after every update.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sock", required=True)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--init", type=Path, default=ROOT / "data/cache/forge_r1/best.pt")
    ap.add_argument("--value-init", type=Path, default=None)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ready", default=None, help="file written (with the pid) once listening")
    a = ap.parse_args()

    import numpy as np
    import torch

    from decklight.eval.bc_policy import _run_args
    from decklight.model.ckpt import load_state
    from decklight.model.policy_v3 import make_policy
    from decklight.rl.remote import PolicyServer, publish_weights

    if not a.weights.exists():
        net = make_policy(dict(np.load(a.emb)), _run_args(a.init), dropout=0.0, value=True)
        state = {k: v.float() if v.is_floating_point() else v for k, v in load_state(a.init).items()}
        if a.value_init is not None:
            vs = torch.load(a.value_init, map_location="cpu")
            vs = vs.get("model", vs)
            state.update({k: v.float() for k, v in vs.items() if k.startswith("value_head.")})
        missing, unexpected = net.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("value_head.") for k in missing):
            print(f"ALERT policy_server: init does not fit ({missing[:3]}, {unexpected[:3]})", flush=True)
            return 3
        publish_weights(net, a.weights, 0)
        print(f"[policy-server] published {a.init} as version 0 -> {a.weights}", flush=True)
    srv = PolicyServer(a.weights, a.init, a.emb, device=a.device, max_batch=a.max_batch, seed=a.seed,
                       threads=a.threads)
    srv.serve(a.sock, ready_file=a.ready)
    return 0


if __name__ == "__main__":
    sys.exit(main())
