"""Learned value v2: P(the acting player wins | state) from finished games (ADR 0007).

Trains PolicyNet(value=True) with MSE(value, z) on the outcome shards from
scripts/build_value_shards.py. Splits are by GAME (never by state): holdout = games
with either deck in --holdout, val = 10% of the remaining games (seeded hash).

    uv run python scripts/train_value.py --shards data/cache/value_shards \\
        --emb data/cards/emb/qwen3-0.6b-d512.npz --init <policy best.pt> \\
        --out runs/<name>/value --epochs 4 --device cuda

Metrics per epoch (metrics.jsonl) for train/val/holdout: MSE, sign accuracy, sign
accuracy by turn bucket (1, 2, 3-4, 5-8, 9+), calibration (10 bins of p=(value+1)/2
with mean p vs actual win share, plus ECE) and the heuristic's sign accuracy (hacc).
Checkpoints: last.pt (with optimizer), best.pt (fp16 weights, by val MSE).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from decklight.model.bc_data import OBS_KEYS
from decklight.model.policy_v3 import make_policy

TURN_BUCKETS = ("1", "2", "3-4", "5-8", "9+")


def turn_bucket(t: int) -> str:
    if t <= 1:
        return "1"
    if t == 2:
        return "2"
    if t <= 4:
        return "3-4"
    if t <= 8:
        return "5-8"
    return "9+"


def load_value_shards(shard_dir: Path) -> dict[str, np.ndarray]:
    """Concatenate one directory of value shards. Game ids are global (assigned at
    build time), so plain concatenation keeps games distinct; the game_ids/game_row
    side tables are merged with dedup."""
    files = sorted(shard_dir.glob("*.npz"))
    parts = [dict(np.load(f)) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in ("game_ids", "game_row")}
    seen: dict[int, str] = {}
    for p in parts:
        for g, s in zip(p["game_row"].tolist(), [str(x) for x in p["game_ids"].tolist()], strict=True):
            seen.setdefault(int(g), s)
    order = sorted(seen)
    out["game_ids"] = np.array([seen[g] for g in order])
    out["game_row"] = np.array(order, dtype=np.int32)
    return out


def default_holdout() -> list[str]:
    decks = ["kewl-tune"]
    p = Path("decks/pool/HOLDOUT")
    if p.is_file():
        decks += [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return decks


def split_games(
    data: dict[str, np.ndarray], holdout_decks: tuple[str, ...], val_frac: float, seed: int
) -> dict[str, np.ndarray]:
    """Boolean masks train / val / holdout over state rows. Holdout = games with either
    deck in holdout_decks; val = a seeded hash of the game id over the rest."""
    games = data["game"]
    uniq = np.unique(games)
    d0 = {g: str(d) for g, d in zip(games, data["deck0"], strict=True)}
    d1 = {g: str(d) for g, d in zip(games, data["deck1"], strict=True)}
    hold_games = {g for g in uniq if d0[g] in holdout_decks or d1[g] in holdout_decks}
    hold = np.isin(games, list(hold_games))
    h = np.array([int(hashlib.md5(f"{seed}:{g}".encode()).hexdigest()[:8], 16) % 1000 for g in games])
    val = (h < int(val_frac * 1000)) & ~hold
    return {"train": ~val & ~hold, "val": val, "holdout": hold}


def train_batches(data, idx, bs, device, rng):
    idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        obs = {k: torch.from_numpy(np.ascontiguousarray(data[k][j])).to(device) for k in OBS_KEYS}
        yield obs, torch.from_numpy(np.ascontiguousarray(data["z"][j])).to(device).float()


@torch.no_grad()
def evaluate(model, data, idx, bs, device) -> dict:
    """Split metrics over idx in row order (no shuffle)."""
    model.eval()
    n = se = hit = 0
    bkt_n: dict[str, int] = {}
    bkt_hit: dict[str, int] = {}
    h_hit = 0.0
    bins_p = np.zeros(10)
    bins_w = np.zeros(10)
    bins_n = np.zeros(10, dtype=np.int64)
    z_all = data["z"][idx]
    turns = data["turn"][idx]
    hs = data["hscore"][idx]
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        obs = {k: torch.from_numpy(np.ascontiguousarray(data[k][j])).to(device) for k in OBS_KEYS}
        _, v = model.forward_value(obs)
        v = v.float().cpu().numpy()
        z = z_all[i : i + len(j)]
        n += len(j)
        se += float(((v - z) ** 2).sum())
        pred, act = v >= 0, z > 0
        hit += int((pred == act).sum())
        for t, p, a in zip(turns[i : i + len(j)], pred, act, strict=True):
            b = turn_bucket(int(t))
            bkt_n[b] = bkt_n.get(b, 0) + 1
            bkt_hit[b] = bkt_hit.get(b, 0) + int(p == a)
        h = hs[i : i + len(j)]
        h_hit += float((((h > 0) == act).astype(float) * (h != 0) + (h == 0) * 0.5).sum())
        p = np.clip((v + 1) / 2, 0, 1)
        b = np.clip((p * 10).astype(int), 0, 9)
        for k in range(10):
            m = b == k
            bins_n[k] += int(m.sum())
            bins_p[k] += float(p[m].sum())
            bins_w[k] += float(act[m].sum())
    model.train()
    rec = {
        "n": n,
        "mse": se / max(n, 1),
        "acc": hit / max(n, 1),
        "acc_turn": {b: round(bkt_hit[b] / bkt_n[b], 4) for b in TURN_BUCKETS if b in bkt_n},
        "hacc": h_hit / max(n, 1),
    }
    bins = []
    for k in range(10):
        c = int(bins_n[k])
        bins.append([round(bins_p[k] / c, 4) if c else None, round(bins_w[k] / c, 4) if c else None, c])
    with np.errstate(invalid="ignore", divide="ignore"):
        ece = float(
            (bins_n / max(n, 1) * np.abs(bins_p / np.maximum(bins_n, 1) - bins_w / np.maximum(bins_n, 1))).sum()
        )
    rec["cal"] = {"bins": bins, "ece": round(ece, 4)}
    return rec


def _run_args(ckpt: Path) -> dict:
    m = ckpt.parent / "metrics.jsonl"
    for raw in m.read_text().splitlines() if m.is_file() else []:
        rec = json.loads(raw)
        if "args" in rec:
            return rec["args"]
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True, help="policy checkpoint (value head is new)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--holdout", nargs="*", default=None, help="decks never trained on (default: HOLDOUT + kewl-tune)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N states (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (laptop: <= 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-init", type=int, default=1,
                    help="1: score the --init value (when it has a value head) on every split first (epoch 'init'), "
                         "the baseline a new value must beat on the same games")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    holdout = tuple(args.holdout) if args.holdout is not None else tuple(default_holdout())
    args.out.mkdir(parents=True, exist_ok=True)

    data = load_value_shards(args.shards)
    if args.limit:
        data = {k: v[: args.limit] for k, v in data.items() if k not in ("game_ids", "game_row")}
    masks = split_games(data, holdout, args.val_frac, args.seed)
    idx = {k: np.flatnonzero(m) for k, m in masks.items()}
    print({k: len(v) for k, v in idx.items()}, "decks:", sorted(set(map(str, data["deck"]))), flush=True)

    src_args = _run_args(args.init)
    d = src_args.get("d", args.d)
    layers = src_args.get("layers", args.layers)
    heads = src_args.get("heads", args.heads)
    hist = src_args.get("hist", 0) or 0
    known = src_args.get("known", False) or False
    if hist or known:
        raise ValueError("value shards carry the stored history window only: --init must have hist=0, known=False")
    v3 = {k: src_args[k] for k in ("v3", "desc_strings", "desc_lua") if k in src_args}
    model = make_policy(args.emb, {"d": d, "layers": layers, "heads": heads, **v3}, dropout=args.dropout,
                        value=True).to(device)  # fmt: skip
    state = torch.load(args.init, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(
        {k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=False
    )
    assert not unexpected and all(k.startswith("value_head.") for k in missing), f"bad init load: {missing=} {unexpected=}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params {n_params / 1e6:.2f}M device {device}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(max(len(idx["train"]), 1) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.05)
    log = (args.out / "metrics.jsonl").open("a")
    t0 = time.time()
    args_rec = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args_rec.update({"d": d, "layers": layers, "heads": heads, "value": True, **v3})
    log.write(json.dumps({"epoch": -1, "init": str(args.init), "value": True, "args": args_rec}) + "\n")
    log.flush()
    best = float("inf")
    if args.eval_init and not missing:  # --init carries a trained value head (e.g. value_ppo3): its baseline
        rec0: dict = {"epoch": "init", "init": str(args.init)}
        for split in ("train", "val", "holdout"):
            if len(idx[split]):
                rec0[split] = evaluate(model, data, idx[split], args.batch * 2, device)
        print(json.dumps(rec0), flush=True)
        log.write(json.dumps(rec0) + "\n")
        log.flush()
    for ep in range(args.epochs):
        run_loss, run_n = 0.0, 0
        model.train()
        for obs, z in train_batches(data, idx["train"], args.batch, device, rng):
            _, v = model.forward_value(obs)
            loss = ((v - z) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(z)
            run_n += len(z)
        rec: dict = {"epoch": ep, "time": round(time.time() - t0, 1), "train_loss": run_loss / max(run_n, 1)}
        if ep == 0:
            rec["args"] = args_rec
        for split in ("train", "val", "holdout"):
            if len(idx[split]):
                rec[split] = evaluate(model, data, idx[split], args.batch * 2, device)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "args": vars(args)},
            args.out / "last.pt",
        )
        score = rec["val"]["mse"] if "val" in rec else rec.get("train", {}).get("mse", float("inf"))
        if score < best:
            best = score
            torch.save(
                {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                args.out / "best.pt",
            )
    print(f"done in {time.time() - t0:.0f}s, best val mse {best:.4f}", flush=True)
    print_calibration_table(args.out / "metrics.jsonl")


def print_calibration_table(metrics: Path) -> None:
    """The last epoch's calibration table (bin: mean p vs actual win share)."""
    recs = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    recs = [r for r in recs if "val" in r or "train" in r]
    if not recs:
        return
    for split in ("train", "val", "holdout"):
        if split not in recs[-1]:
            continue
        m = recs[-1][split]
        print(f"calibration {split} (n={m['n']} mse={m['mse']:.4f} acc={m['acc']:.4f} "
              f"ece={m['cal']['ece']:.4f} hacc={m['hacc']:.4f}):", flush=True)
        for i, (p, w, c) in enumerate(m["cal"]["bins"]):
            ps = f"{p:.3f}" if p is not None else "  - "
            ws = f"{w:.3f}" if w is not None else "  - "
            print(f"  p{i / 10:.1f}-{(i + 1) / 10:.1f}: mean_p {ps} win_share {ws} n {c}", flush=True)


if __name__ == "__main__":
    main()
