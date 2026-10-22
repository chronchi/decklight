"""Paired per-label comparison of two policies on held-out labels (the decisive-label learnability test, NOTEBOOK
2026-10-20). For every held-out sample of the given kinds, `hit` = the policy's argmax is in the label's equivalence set.
Reports, per split (val = unseen games of trained decks, holdout = unseen decks) and kind:
  base acc, student acc, fixes (base wrong -> student right), breaks (base right -> student wrong), one-sided exact
  McNemar p (fixes vs breaks).
The split is recomputed exactly as train_bc.py does it (same --holdout and --val-frac-kind).

    python scripts/eval_flips.py --shards data/cache/bc_shards --emb <npz> --base P/best.pt --student run/final.pt \
        --kinds sw-hw sw-hw_own --holdout <decks> --val-frac-kind sw-hw=0.25 sw-hw_own=0.25 [--out flips.json]
Exits 2 when there are 0 held-out samples, 3 when base and student give identical predictions everywhere
(an untrained student). GPU box only for real runs; a laptop check uses --limit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bc import _run_args, batches

from decklight.model.bc_data import load_shards, split_masks
from decklight.model.policy_v3 import make_policy


def mcnemar_one_sided(fixes: int, breaks: int) -> float:
    """P(X >= fixes) for X ~ Bin(fixes + breaks, 0.5): the chance of this many fixes if the change were random."""
    n = fixes + breaks
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(fixes, n + 1)) / 2**n


def load_policy(ckpt: Path, emb: Path, device):
    a = {"d": 256, "layers": 4, "heads": 4, "hist": 0, "known": False, "v3": False}
    a.update(
        {
            k: v
            for k, v in _run_args(ckpt).items()
            if k in ("d", "layers", "heads", "hist", "known", "v3", "desc_strings", "desc_lua")
        }
    )
    a["emb"] = emb
    model = make_policy(emb, a, dropout=0.0, card_dropout=0.0).to(device)
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.eval(), a


def hits(model, a, data, idx, device) -> np.ndarray:
    out = np.zeros(len(idx), bool)
    pos = 0
    with torch.no_grad():
        for obs, _act, _w, eq, _si, _sq, j in batches(
            data, idx, 256, device, False, None, a["hist"], True, a["known"]
        ):
            pred = model(obs).float().argmax(-1)
            out[pos : pos + len(j)] = eq[torch.arange(len(j), device=device), pred].cpu().numpy()
            pos += len(j)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--kinds", nargs="+", required=True)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--val-frac-kind", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    device = torch.device(a.device)
    data = load_shards(a.shards)
    kvf = {k: float(v) for k, v in (s.split("=") for s in a.val_frac_kind)}
    masks = split_masks(data, holdout_decks=tuple(a.holdout), kind_val_frac=kvf)
    kind_ok = np.isin(data["kind"], a.kinds)
    base, ba = load_policy(a.base, a.emb, device)
    stud, sa = load_policy(a.student, a.emb, device)
    rep: dict = {}
    any_diff = False
    total = 0
    for split in ("val", "holdout"):
        idx = np.flatnonzero(masks[split] & kind_ok)
        if a.limit:
            idx = idx[: a.limit]
        total += len(idx)
        if len(idx) == 0:
            continue
        hb, hs = hits(base, ba, data, idx, device), hits(stud, sa, data, idx, device)
        any_diff |= bool((hb != hs).any())
        kinds = data["kind"][idx]
        for kd in [*a.kinds, "all"]:
            m = np.ones(len(idx), bool) if kd == "all" else kinds == kd
            if not m.any():
                continue
            fx, br = int((~hb[m] & hs[m]).sum()), int((hb[m] & ~hs[m]).sum())
            r = {
                "n": int(m.sum()),
                "base_acc": round(float(hb[m].mean()), 4),
                "student_acc": round(float(hs[m].mean()), 4),
                "base_wrong": int((~hb[m]).sum()),
                "fixes": fx,
                "breaks": br,
                "p_one_sided": round(mcnemar_one_sided(fx, br), 4),
            }
            rep[f"{split}/{kd}"] = r
            print(
                f"{split:8s} {kd:12s} n {r['n']:5d}  base {r['base_acc']:.3f} -> student {r['student_acc']:.3f}  "
                f"fixes {fx} / breaks {br} (of {r['base_wrong']} base-wrong)  p {r['p_one_sided']:.4f}",
                flush=True,
            )
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))
    if total == 0:
        print("ALERT eval_flips: 0 held-out samples of the requested kinds", flush=True)
        return 2
    if not any_diff:
        print("ALERT eval_flips: base and student predict identically on every held-out sample", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
