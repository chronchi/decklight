"""Behaviour cloning on the demo records (PLAN Phase 4, ADR 0003 data).

uv run python scripts/train_bc.py --shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz \
    --out runs/bc-smoke --epochs 1 --limit 2000 --threads 2          # laptop smoke (CPU, <= 2 threads)
On vast: scripts/vast_bc.sh builds shards + embeddings and calls this with --device cuda.

Metrics per epoch (runs/<name>/metrics.jsonl): loss, top-1 accuracy over legal options (all / stated-only) on train,
val (unseen lines of trained decks) and holdout (unseen decks). Checkpoints: last.pt (with optimizer), best.pt
(fp16 weights, by val accuracy).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ygo_zero.model.bc_data import OBS_KEYS, load_shards, split_masks
from ygo_zero.model.policy import PolicyNet


def batches(data, idx, bs, device, shuffle, rng):
    if shuffle:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        obs = {k: torch.from_numpy(data[k][j]).to(device, non_blocking=True) for k in OBS_KEYS}
        yield (
            obs,
            torch.from_numpy(data["action"][j]).to(device),
            torch.from_numpy(data["weight"][j]).to(device),
            j,
        )


def evaluate(model, data, idx, bs, device, amp):
    model.eval()
    tot = {"n": 0, "loss": 0.0, "acc": 0.0, "n_stated": 0, "acc_stated": 0.0, "chance": 0.0}
    per_deck: dict[str, list[int]] = {}
    with torch.no_grad():
        for obs, act, w, j in batches(data, idx, bs, device, False, None):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(obs)
            logits = logits.float()
            loss = F.cross_entropy(logits, act, reduction="none")
            hit = (logits.argmax(-1) == act).float()
            tot["chance"] += (1.0 / obs["action_mask"].sum(-1).clamp(min=1).float()).sum().item()
            stated = w >= 1.0
            tot["n"] += len(j)
            tot["loss"] += loss.sum().item()
            tot["acc"] += hit.sum().item()
            tot["n_stated"] += stated.sum().item()
            tot["acc_stated"] += hit[stated].sum().item()
            for d, h in zip(data["deck"][j], hit.tolist(), strict=True):
                per_deck.setdefault(str(d), [0, 0])
                per_deck[str(d)][0] += 1
                per_deck[str(d)][1] += int(h)
    model.train()
    n = max(tot["n"], 1)
    return {
        "n": tot["n"],
        "loss": tot["loss"] / n,
        "acc": tot["acc"] / n,
        "acc_stated": tot["acc_stated"] / max(tot["n_stated"], 1),
        "chance": tot["chance"] / n,  # expected accuracy of a uniform pick among the legal options
        "per_deck": {d: round(h / c, 4) for d, (c, h) in sorted(per_deck.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--holdout", nargs="*", default=["kewl-tune"], help="decks never trained on")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N samples (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (laptop: <= 2)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load_shards(args.shards)
    if args.limit:
        data = {k: v[: args.limit] for k, v in data.items()}
    masks = split_masks(data, holdout_decks=tuple(args.holdout))
    idx = {k: np.flatnonzero(m) for k, m in masks.items()}
    print({k: len(v) for k, v in idx.items()}, "decks:", sorted(set(map(str, data["deck"]))), flush=True)

    emb = dict(np.load(args.emb))
    model = PolicyNet(emb, d=args.d, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params {n_params / 1e6:.2f}M device {device} amp {amp}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(idx["train"]) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.05
    )
    best = -1.0
    log = (args.out / "metrics.jsonl").open("a")
    t0 = time.time()
    for ep in range(args.epochs):
        run = {"loss": 0.0, "n": 0}
        for obs, act, w, j in batches(data, idx["train"], args.batch, device, True, rng):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(obs)
            loss = (F.cross_entropy(logits.float(), act, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run["loss"] += loss.item() * len(j)
            run["n"] += len(j)
        rec = {"epoch": ep, "time": round(time.time() - t0, 1), "train_loss": run["loss"] / max(run["n"], 1)}
        for split in ("val", "holdout"):
            if len(idx[split]):
                rec[split] = evaluate(model, data, idx[split], args.batch * 2, device, amp)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "args": vars(args)},
            args.out / "last.pt",
        )
        score = rec.get("val", {}).get("acc", -rec["train_loss"])
        if score > best:
            best = score
            torch.save(
                {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                args.out / "best.pt",
            )
    print(f"done in {time.time() - t0:.0f}s, best val acc {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
