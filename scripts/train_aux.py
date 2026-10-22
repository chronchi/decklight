"""Auxiliary evaluator heads on the frozen value trunk (engine plan N3).

Loads value_p2 (`--value`), freezes everything, and trains `AuxHeads`
(`src/decklight/eval/aux.py`) on trunk features (computed under `torch.no_grad()`
per batch) with AdamW and early stopping on the holdout loss.

    uv run python scripts/train_aux.py --shards data/cache/aux_shards \\
        --emb data/cards/emb/qwen3-0.6b-d512.npz --value data/cache/value_p2/best.pt \\
        --out runs/<name>/aux --epochs 10 --device cuda

Writes `<out>/aux.pt` (fp16 weights + AUX_SPEC + value ckpt sha256) and
`<out>/metrics.json`: per-target R2 vs the persistence baseline (predict 0 change
for d_lp/d_cards; base rate for ends: Brier and AUC; train mean for turns_left),
plus the value-head outputs checked bit-identical before/after on a fixed batch.

Invariants: exit 2 on 0 samples, NaN loss, or a target with 0 unmasked samples;
exit 3 if no head beats its baseline; exit 4 if the value head moved.
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

from decklight.eval.aux import AUX_SPEC, AuxHeads, aux_losses
from decklight.model.bc_data import OBS_KEYS
from decklight.model.policy_v3 import make_policy

TARGET_KEYS = ("d_lp", "d_cards", "ends", "turns_left")


def fail(code: int, msg: str) -> None:
    print(f"ALERT train_aux: {msg}", flush=True)
    raise SystemExit(code)


def load_aux_shards(shard_dir: Path) -> dict[str, np.ndarray]:
    """Concatenate one directory of aux shards. Game ids are global (assigned at
    build time), so plain concatenation keeps games distinct; the game_ids/game_row
    side tables are merged with dedup."""
    files = sorted(shard_dir.glob("*.npz"))
    if not files:
        fail(2, f"no .npz shards in {shard_dir}")
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


def split_games(games: np.ndarray, val_frac: float, seed: int) -> dict[str, np.ndarray]:
    """Boolean masks train / holdout over state rows, split by game id (never by
    state). Guarantees a non-empty holdout whenever there are >= 2 games."""
    uniq = np.unique(games)
    h = np.array([int(hashlib.md5(f"{seed}:{g}".encode()).hexdigest()[:8], 16) % 1000 for g in games])
    hold = h < int(val_frac * 1000)
    if not hold.any() and len(uniq) >= 2:
        hold = games == uniq.min()
    return {"train": ~hold, "holdout": hold}


def _run_args(ckpt: Path) -> dict:
    m = ckpt.parent / "metrics.jsonl"
    for raw in m.read_text().splitlines() if m.is_file() else []:
        rec = json.loads(raw)
        if "args" in rec:
            return rec["args"]
    return {}


def _obs_batch(data: dict, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(data[k][idx])).to(device) for k in OBS_KEYS}


def _tgt_batch(data: dict, idx: np.ndarray, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    t = {k: torch.from_numpy(np.ascontiguousarray(data[k][idx])).to(device).float() for k in TARGET_KEYS}
    return t, torch.from_numpy(np.ascontiguousarray(data["mask"][idx])).to(device)


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (no sklearn dependency); NaN when one class is absent."""
    y = np.asarray(y).ravel()
    s = np.asarray(s).ravel()
    p = s[y == 1]
    q = s[y == 0]
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([p, q]), kind="stable")).astype(np.float64) + 1
    return float((ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))


@torch.no_grad()
def evaluate(aux: AuxHeads, model, data: dict, idx: np.ndarray, bs: int,
             device: torch.device, base: dict[str, float]) -> tuple[dict, float]:
    """Holdout/train metrics over idx in row order. Returns (per-target rec, total loss)."""
    aux.eval()
    model.eval()
    agg = {k: {"se": 0.0, "se_base": 0.0, "n": 0} for k in TARGET_KEYS}
    ends_y, ends_p = [], []
    run_loss, run_n = 0.0, 0
    for i in range(0, len(idx), bs):
        j = idx[i: i + bs]
        obs = _obs_batch(data, j, device)
        tgt, mask = _tgt_batch(data, j, device)
        feat = model.features(obs)
        pred = aux(feat)
        loss = aux_losses(pred, tgt, mask)
        run_loss += float(loss["total"]) * len(j)
        run_n += len(j)
        m = mask.float()
        for k in ("d_lp", "d_cards"):
            se = (((pred[k] - tgt[k]) ** 2).mean(-1) * m).sum().item()
            se0 = ((tgt[k] ** 2).mean(-1) * m).sum().item()
            agg[k]["se"] += se
            agg[k]["se_base"] += se0
            agg[k]["n"] += float(m.sum())
        se = ((pred["turns_left"] - tgt["turns_left"]) ** 2).sum().item()
        se0 = ((tgt["turns_left"] - base["turns_mean"]) ** 2).sum().item()
        agg["turns_left"]["se"] += se
        agg["turns_left"]["se_base"] += se0
        agg["turns_left"]["n"] += len(j)
        p = torch.sigmoid(pred["ends"])
        se = ((p - tgt["ends"]) ** 2).sum().item()
        se0 = ((tgt["ends"] - base["ends_rate"]) ** 2).sum().item()
        agg["ends"]["se"] += se
        agg["ends"]["se_base"] += se0
        agg["ends"]["n"] += len(j)
        ends_y.append(tgt["ends"].cpu().numpy())
        ends_p.append(p.cpu().numpy())
    rec: dict = {}
    for k in TARGET_KEYS:
        n = agg[k]["n"]
        mse = agg[k]["se"] / max(n, 1)
        mse0 = agg[k]["se_base"] / max(n, 1)
        r = {"n": int(n), "mse": mse, "mse_base": mse0,
             "r2": (1 - mse / mse0) if mse0 > 0 else None}
        if k == "ends":
            y = np.concatenate(ends_y) if ends_y else np.zeros(0)
            s = np.concatenate(ends_p) if ends_p else np.zeros(0)
            r["brier"], r["brier_base"] = mse, mse0
            r["auc"] = auc_score(y, s)
            r["base_rate"] = base["ends_rate"]
        if k == "turns_left":
            r["train_mean"] = base["turns_mean"]
        rec[k] = r
    return rec, run_loss / max(run_n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--emb", type=Path, default=Path("data/cards/emb/qwen3-0.6b-d512.npz"))
    ap.add_argument("--value", type=Path, default=Path("data/cache/value_p2/best.pt"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=3, help="early-stop epochs without holdout-loss improvement")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N states (smoke)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (laptop: <= 2)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.value.is_file():
        fail(2, f"missing value checkpoint {args.value}")

    data = load_aux_shards(args.shards)
    if args.limit:
        data = {k: v[: args.limit] for k, v in data.items() if k not in ("game_ids", "game_row")}
    n = len(data["game"])
    if n == 0:
        fail(2, "0 samples in the aux shards")
    if int(data["mask"].sum()) == 0:
        fail(2, "a target with 0 unmasked samples (d_lp/d_cards: no has_next row)")
    masks = split_games(data["game"], args.val_frac, args.seed)
    idx = {k: np.flatnonzero(m) for k, m in masks.items()}
    print({k: len(v) for k, v in idx.items()}, f"unmasked {int(data['mask'].sum())}/{n}", flush=True)

    src_args = _run_args(args.value)
    va = {"d": src_args.get("d", 256), "layers": src_args.get("layers", 4),
          "heads": src_args.get("heads", 4), "hist": int(src_args.get("hist", 0) or 0),
          "known": bool(src_args.get("known", False) or False)}
    if va["hist"] or va["known"]:
        fail(2, "aux shards carry the stored history window only: --value must have hist=0, known=False")
    v3 = {k: src_args[k] for k in ("v3", "desc_strings", "desc_lua") if k in src_args}
    model = make_policy(args.emb, {**va, **v3}, value=True).to(device)
    state = torch.load(args.value, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(
        {k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=False)
    if unexpected or any(not k.startswith("value_head.") for k in missing):
        fail(2, f"bad value load: missing={missing} unexpected={unexpected}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    aux = AuxHeads(va["d"]).to(device)
    opt = torch.optim.AdamW(aux.parameters(), lr=args.lr, weight_decay=args.wd)

    fix = idx["train"][: min(64, len(idx["train"]))]
    with torch.no_grad():
        _, v_before = model.forward_value(_obs_batch(data, fix, device))

    base = {"turns_mean": float(data["turns_left"][idx["train"]].mean()),
            "ends_rate": float(data["ends"][idx["train"]].mean())}
    t0 = time.time()
    best, best_state, bad, history = float("inf"), None, 0, []
    for ep in range(args.epochs):
        aux.train()
        run_loss, run_n = 0.0, 0
        order = rng.permutation(idx["train"])
        for i in range(0, len(order), args.batch):
            j = order[i: i + args.batch]
            obs = _obs_batch(data, j, device)
            tgt, mask = _tgt_batch(data, j, device)
            with torch.no_grad():
                feat = model.features(obs)
            pred = aux(feat)
            loss = aux_losses(pred, tgt, mask)["total"]
            if not math.isfinite(float(loss.detach())):
                fail(2, f"NaN loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aux.parameters(), 1.0)
            opt.step()
            run_loss += float(loss.detach()) * len(j)
            run_n += len(j)
        rec: dict = {"epoch": ep, "time": round(time.time() - t0, 1),
                     "train_loss": run_loss / max(run_n, 1)}
        for split in ("train", "holdout"):
            if len(idx[split]):
                rec[split], _ = evaluate(aux, model, data, idx[split], args.batch * 2, device, base)
        h = rec.get("holdout")
        score = h["d_lp"]["mse"] + h["d_cards"]["mse"] + h["ends"]["mse"] + h["turns_left"]["mse"] \
            if h else rec["train_loss"]
        if score < best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu() for k, v in aux.state_dict().items()}
        else:
            bad += 1
        r2s = {k: (round(h[k]["r2"], 4) if h and h[k]["r2"] is not None else None) for k in TARGET_KEYS} if h else {}
        print(json.dumps({"epoch": ep, "train_loss": rec["train_loss"], "holdout_r2": r2s}), flush=True)
        history.append(rec)
        if bad >= args.patience:
            print(f"early stop at epoch {ep} (best holdout {best:.4f})", flush=True)
            break
    if best_state is None:
        fail(2, "no epoch completed")
    aux.load_state_dict(best_state)

    with torch.no_grad():
        _, v_after = model.forward_value(_obs_batch(data, fix, device))
    identical = bool(torch.equal(v_before, v_after))

    final: dict = {"epochs": len(history), "best_holdout_loss": best, "value_identical": identical,
                   "value_ckpt": str(args.value),
                   "value_sha256": hashlib.sha256(args.value.read_bytes()).hexdigest()}
    for split in ("train", "holdout"):
        if len(idx[split]):
            final[split], _ = evaluate(aux, model, data, idx[split], args.batch * 2, device, base)
    h = final.get("holdout", final.get("train", {}))
    beats = any(h.get(k, {}).get("r2", 0) is not None and h[k]["r2"] > 0 for k in ("d_lp", "d_cards", "turns_left")) \
        or (h.get("ends", {}).get("mse", 1) < h.get("ends", {}).get("mse_base", 0))
    final["beats_baseline"] = bool(beats)
    (args.out / "metrics.json").write_text(json.dumps(final, indent=1))
    torch.save({"model": {k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
                "aux_spec": AUX_SPEC, "value_sha256": final["value_sha256"]}, args.out / "aux.pt")
    print(json.dumps({k: {m: h[k][m] for m in ("r2", "mse", "mse_base") if m in h[k]}
                      for k in TARGET_KEYS if k in h}), flush=True)
    if not identical:
        fail(4, "value-head outputs changed during aux training")
    if not beats:
        fail(3, "no aux head beats its baseline")


if __name__ == "__main__":
    main()
