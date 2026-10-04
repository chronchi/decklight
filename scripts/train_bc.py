"""Behaviour cloning on the demo records (PLAN Phase 4, ADR 0003 data).

uv run python scripts/train_bc.py --shards data/cache/bc_shards --emb data/cards/emb/qwen3-0.6b-d512.npz \
    --out runs/bc-smoke --epochs 1 --limit 2000 --threads 2          # laptop smoke (CPU, <= 2 threads)
On vast: scripts/vast_bc.sh builds shards + embeddings and calls this with --device cuda.

Metrics per epoch (runs/<name>/metrics.jsonl): loss, top-1 accuracy over legal options (all / stated-only) on train,
val (unseen lines of trained decks) and holdout (unseen decks); with --set-target (default) the loss is
-log(sum of softmax over the equiv set) and acc counts argmax in the equiv set (acc_exact = old definition).
Checkpoints: last.pt (with optimizer), best.pt (fp16 weights, by val accuracy).
Mixing: --kind-frac line_expanded=0.2 game_plan=3 redraws each kind's train share every epoch (<1 subsample, >1 repeat).
Merging: --merge-last K also evaluates the uniform average of the last K epochs' weights (Kolibri tech report §2.2:
checkpoint merges add points); saved as merged.pt when it beats best.pt on --select-on.
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
    has_equiv = "equiv" in data
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        obs = {k: torch.from_numpy(data[k][j]).to(device, non_blocking=True) for k in OBS_KEYS}
        act = torch.from_numpy(data["action"][j]).to(device)
        if has_equiv:
            eq = torch.from_numpy(data["equiv"][j]).to(device, non_blocking=True)
        else:  # old shards: equiv = one-hot of the action
            eq = torch.zeros(len(j), obs["actions"].shape[1], dtype=torch.bool, device=device)
            eq[torch.arange(len(j), device=device), act] = True
        yield (
            obs,
            act,
            torch.from_numpy(data["weight"][j]).to(device),
            eq,
            j,
        )


def set_loss(logits: torch.Tensor, equiv: torch.Tensor) -> torch.Tensor:
    """Per-sample -log(sum of softmax over the equiv set): lse(all) - lse(equiv)."""
    return torch.logsumexp(logits, -1) - torch.logsumexp(
        logits.masked_fill(~equiv, float("-inf")), -1
    )


def evaluate(model, data, idx, bs, device, amp, set_target=True):
    model.eval()
    tot = {
        "n": 0, "loss": 0.0, "acc": 0.0, "acc_exact": 0.0,
        "n_stated": 0, "acc_stated": 0.0, "chance": 0.0,
    }
    per_deck: dict[str, list[int]] = {}
    per_kind: dict[str, list[int]] = {}
    has_kind = "kind" in data
    with torch.no_grad():
        for obs, act, w, eq, j in batches(data, idx, bs, device, False, None):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(obs)
            logits = logits.float()
            if set_target:
                loss = set_loss(logits, eq)
            else:
                loss = F.cross_entropy(logits, act, reduction="none")
            pred = logits.argmax(-1)
            hit = eq[torch.arange(len(j), device=logits.device), pred].float()
            exact = (pred == act).float()
            tot["chance"] += (1.0 / obs["action_mask"].sum(-1).clamp(min=1).float()).sum().item()
            stated = w >= 1.0
            tot["n"] += len(j)
            tot["loss"] += loss.sum().item()
            tot["acc"] += hit.sum().item()
            tot["acc_exact"] += exact.sum().item()
            tot["n_stated"] += stated.sum().item()
            tot["acc_stated"] += hit[stated].sum().item()
            for d, h in zip(data["deck"][j], hit.tolist(), strict=True):
                per_deck.setdefault(str(d), [0, 0])
                per_deck[str(d)][0] += 1
                per_deck[str(d)][1] += int(h)
            if has_kind:
                for kd, h in zip(data["kind"][j], hit.tolist(), strict=True):
                    per_kind.setdefault(str(kd), [0, 0])
                    per_kind[str(kd)][0] += 1
                    per_kind[str(kd)][1] += int(h)
    model.train()
    n = max(tot["n"], 1)
    return {
        "n": tot["n"],
        "loss": tot["loss"] / n,
        "acc": tot["acc"] / n,
        "acc_exact": tot["acc_exact"] / n,
        "acc_stated": tot["acc_stated"] / max(tot["n_stated"], 1),
        "chance": tot["chance"] / n,  # expected accuracy of a uniform pick among the legal options
        "per_deck": {d: round(h / c, 4) for d, (c, h) in sorted(per_deck.items())},
        "per_kind": {k: round(h / c, 4) for k, (c, h) in sorted(per_kind.items())},
    }


def select_train_idx(data, masks, kinds):
    """Index sets with optional --kinds filtering: train keeps only those kinds, val/holdout keep all."""
    idx = {k: np.flatnonzero(m) for k, m in masks.items()}
    if kinds is not None:
        kind_arr = data["kind"] if "kind" in data else np.array(["unknown"] * len(data["action"]))
        keep = np.isin(kind_arr, list(kinds))
        idx["train"] = idx["train"][keep[idx["train"]]]
    return idx


def parse_kind_frac(items: list[str] | None) -> dict[str, float]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = float(v)
    return out


def epoch_train_idx(data, train_idx: np.ndarray, kind_frac: dict[str, float], rng) -> np.ndarray:
    """Per-epoch training indices: each kind in kind_frac keeps frac x its samples (fresh draw every epoch; frac > 1
    repeats the kind floor(frac) times plus a random remainder). Kinds not listed keep all their samples once."""
    if not kind_frac or "kind" not in data:
        return train_idx
    kinds = data["kind"][train_idx]
    parts = [train_idx[~np.isin(kinds, list(kind_frac))]]
    for k, f in kind_frac.items():
        pool = train_idx[kinds == k]
        if not len(pool) or f <= 0:
            continue
        reps, rem = int(f), f - int(f)
        parts += [pool] * reps
        if rem > 0:
            parts.append(rng.choice(pool, size=round(rem * len(pool)), replace=False))
    return np.concatenate(parts)


def average_states(states: list[dict]) -> dict:
    return {k: (sum(s[k].float() for s in states) / len(states)) if states[0][k].is_floating_point() else states[-1][k]
            for k in states[0]}


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
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument(
        "--set-target", dest="set_target", action=argparse.BooleanOptionalAction, default=True,
        help="loss = -log(sum of softmax over equiv options); off = plain CE on the action",
    )
    ap.add_argument("--select-on", choices=["val", "holdout"], default="val")
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--card-dropout", type=float, default=0.0)
    ap.add_argument("--holdout", nargs="*", default=["kewl-tune"], help="decks never trained on")
    ap.add_argument("--kinds", nargs="*", default=None, help="train only on these sample kinds (default: all)")
    ap.add_argument("--kind-frac", nargs="*", default=None, metavar="KIND=FRAC",
                    help="per-epoch share of each kind's train samples (<1 subsample, >1 repeat)")
    ap.add_argument("--merge-last", type=int, default=0, help="also evaluate the average of the last K epochs")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N samples (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (laptop: <= 2)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.set_target and args.label_smoothing:
        ap.error("--set-target and --label-smoothing are mutually exclusive")
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
    idx = select_train_idx(data, masks, args.kinds)
    print({k: len(v) for k, v in idx.items()}, "decks:", sorted(set(map(str, data["deck"]))), flush=True)

    emb = dict(np.load(args.emb))
    model = PolicyNet(
        emb, d=args.d, layers=args.layers, heads=args.heads, dropout=args.dropout,
        card_dropout=args.card_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params {n_params / 1e6:.2f}M device {device} amp {amp}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    kind_frac = parse_kind_frac(args.kind_frac)
    n_epoch = len(epoch_train_idx(data, idx["train"], kind_frac, np.random.default_rng(0)))
    print(f"train samples per epoch {n_epoch} (kind_frac {kind_frac or 'none'})", flush=True)
    steps = args.epochs * math.ceil(n_epoch / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.05
    )
    best = -1.0
    bad = 0
    log = (args.out / "metrics.jsonl").open("a")
    t0 = time.time()
    args_rec = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    recent: list[dict] = []
    for ep in range(args.epochs):
        run = {"loss": 0.0, "n": 0}
        ep_idx = epoch_train_idx(data, idx["train"], kind_frac, rng)
        for obs, act, w, eq, j in batches(data, ep_idx, args.batch, device, True, rng):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(obs)
            if args.set_target:
                per = set_loss(logits.float(), eq)
            else:
                per = F.cross_entropy(
                    logits.float(), act, reduction="none", label_smoothing=args.label_smoothing
                )
            loss = (per * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run["loss"] += loss.item() * len(j)
            run["n"] += len(j)
        if args.merge_last > 1:
            recent = (recent + [{k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}])[-args.merge_last:]
        rec = {"epoch": ep, "time": round(time.time() - t0, 1), "n_train": len(ep_idx), "train_loss": run["loss"] / max(run["n"], 1)}
        if ep == 0:
            rec["args"] = args_rec
        for split in ("val", "holdout"):
            if len(idx[split]):
                rec[split] = evaluate(
                    model, data, idx[split], args.batch * 2, device, amp, args.set_target
                )
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "args": vars(args)},
            args.out / "last.pt",
        )
        score = rec.get(args.select_on, {}).get("acc", -rec["train_loss"])
        if score > best:
            best = score
            bad = 0
            torch.save(
                {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                args.out / "best.pt",
            )
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                print(f"early stop at epoch {ep} (no {args.select_on} improvement for {bad} epochs)", flush=True)
                break
    if args.merge_last > 1 and len(recent) > 1:
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(average_states(recent))
        rec = {"merged_last": len(recent)}
        for split in ("val", "holdout"):
            if len(idx[split]):
                rec[split] = evaluate(model, data, idx[split], args.batch * 2, device, amp, args.set_target)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        score = rec.get(args.select_on, {}).get("acc", -1.0)
        if score > best:
            print(f"merge of last {len(recent)} beats best ({score:.4f} > {best:.4f}): merged.pt + best.pt", flush=True)
            best = score
            half = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save(half, args.out / "merged.pt")
            torch.save(half, args.out / "best.pt")
        model.load_state_dict(live)
    print(f"done in {time.time() - t0:.0f}s, best {args.select_on} acc {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
