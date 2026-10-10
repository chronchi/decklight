"""Train the belief head (src/ygo_zero/model/belief.py) on belief shards and report BCE against the hypergeometric
baseline (PLAN §0a "Hidden-information bound" (b); gate: ratio <= 0.9 on held-out games, <= 0.95 on held-out decks).

    uv run python scripts/train_belief.py --shards data/cache/belief_shards --init data/cache/base_r3/best.pt \\
        --out runs/<job>/h128 --hist 128 [--nohist] [--freeze-trunk] [--epochs 3]

Splits (fixed for every variant): holdout = states whose actor or opponent deck is in --holdout (default: the
init checkpoint's BC holdout list, so the trunk never saw them either); val = 10% of the other games by crc32 of the
game id; train = the rest. History variants: --hist 0 = the v1 32-entry window (obs["history"]); --hist N = the
first N salient entries of hist_long (needs a trunk built with hist=N: hist_pos and a 32-bucket hist_age are added
as no-ops); --nohist masks every history token. --freeze-trunk trains only new parameters (known/hist embeddings,
belief head). Writes metrics.jsonl (one line per eval) and final.pt (fp16 weights).

Invariants (exit 3): 0 train / val samples, NaN loss, the untrained head not reproducing the baseline exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from ygo_zero.model.bc_data import KNOWN_KEYS, OBS_KEYS, _npz_headers
from ygo_zero.model.belief import BeliefNet, baseline_bce, bce, init_trunk, scored
from ygo_zero.model.policy import PolicyNet

TARGET_KEYS = ("cand", "cand_n", "y", "base_p")
META_KEYS = ("n_hidden", "turn", "game", "deck_me", "deck_opp")


def alert(out: Path, msg: str) -> None:
    print(f"ALERT {msg}", flush=True)
    with open(out.parent / "ALERT", "a") as f:
        f.write(f"{out.name}: {msg}\n")


def load(shard_dir: Path, hist: int) -> dict[str, np.ndarray]:
    """Every shard, concatenated into preallocated arrays (peak RAM ~ the result). hist_long is cut to `hist`."""
    keys = (*OBS_KEYS, *KNOWN_KEYS, *TARGET_KEYS, *META_KEYS) + (("hist_long", "hist_long_mask") if hist else ())
    files = sorted(shard_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no shards in {shard_dir}")
    heads = [_npz_headers(f) for f in files]
    n = sum(h["y"][0][0] for h in heads)
    out = {}
    for k in keys:
        shape = heads[0][k][0][1:]
        if k in ("hist_long", "hist_long_mask"):
            if shape[0] < hist:
                raise SystemExit(f"shards hold {shape[0]} history entries, --hist {hist} asked")
            shape = (hist, *shape[1:])
        dt = np.result_type(*(h[k][1] for h in heads))
        out[k] = np.empty((n, *shape), dt)
    pos = 0
    for f in files:
        with np.load(f) as z:
            m = len(z["y"])
            for k in keys:
                a = z[k]
                out[k][pos : pos + m] = a[:, :hist] if k in ("hist_long", "hist_long_mask") else a
            pos += m
    return out


def splits(data: dict[str, np.ndarray], holdout: list[str], val_mod: int = 10) -> dict[str, np.ndarray]:
    hold_set = {h.lower() for h in holdout}
    lower = np.vectorize(lambda s: str(s).lower(), otypes=[object])
    hold = np.isin(lower(data["deck_me"]), list(hold_set)) | np.isin(lower(data["deck_opp"]), list(hold_set))
    games = data["game"]
    val = np.array([zlib.crc32(f"game{int(g)}".encode()) % val_mod == 0 for g in games]) & ~hold
    return {"train": np.flatnonzero(~hold & ~val), "val": np.flatnonzero(val), "holdout": np.flatnonzero(hold)}


def batch(data: dict[str, np.ndarray], idx: np.ndarray, hist: int, nohist: bool, device: torch.device
          ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:  # fmt: skip
    obs = {}
    for k in (*OBS_KEYS, *KNOWN_KEYS):
        src = k
        if hist and k == "history":
            src = "hist_long"
        elif hist and k == "history_mask":
            src = "hist_long_mask"
        obs[k] = torch.from_numpy(np.ascontiguousarray(data[src][idx])).to(device, non_blocking=True)
    if nohist:
        obs["history_mask"] = torch.zeros_like(obs["history_mask"])
    tgt = {k: torch.from_numpy(np.ascontiguousarray(data[k][idx])).to(device) for k in TARGET_KEYS}
    tgt["turn"] = torch.from_numpy(data["turn"][idx].astype(np.int64)).to(device)
    return obs, tgt


@torch.no_grad()
def evaluate(model: BeliefNet, data: dict, idx: np.ndarray, args, device, amp: bool) -> dict:
    model.eval()
    tot = {"bce": 0.0, "base": 0.0, "n": 0, "bce_late": 0.0, "base_late": 0.0, "n_late": 0}
    for s in range(0, len(idx), args.eval_batch):
        obs, t = batch(data, idx[s : s + args.eval_batch], args.hist, args.nohist, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg = model(obs, t["cand"], t["cand_n"], t["base_p"])
        lg = lg.float()
        m = scored(t["cand"], t["base_p"])
        late = m & (t["turn"] >= 3).unsqueeze(1)
        tot["bce"] += bce(lg, t["y"], m).item()
        tot["base"] += baseline_bce(t["y"], t["base_p"], m).item()
        tot["n"] += int(m.sum())
        tot["bce_late"] += bce(lg, t["y"], late).item()
        tot["base_late"] += baseline_bce(t["y"], t["base_p"], late).item()
        tot["n_late"] += int(late.sum())
    model.train()
    r = {"states": len(idx), "entries": tot["n"], "bce": tot["bce"] / max(tot["n"], 1),
         "base_bce": tot["base"] / max(tot["n"], 1), "ratio": tot["bce"] / max(tot["base"], 1e-9),
         "ratio_turn3+": tot["bce_late"] / max(tot["base_late"], 1e-9), "entries_turn3+": tot["n_late"]}  # fmt: skip
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True, help="policy checkpoint for the trunk")
    ap.add_argument("--emb", type=Path, default=Path("data/cards/emb/qwen3-0.6b-d512.npz"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hist", type=int, default=0, help="0 = v1 window; N = first N salient hist_long entries")
    ap.add_argument("--nohist", action="store_true")
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--holdout", nargs="*", default=None, help="default: the init checkpoint's BC holdout decks")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4, help="trunk lr (new parameters use 10x)")
    ap.add_argument("--evals-per-epoch", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=None, help="cap train states (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-mod", type=int, default=10, help="val = games with crc32 %% val-mod == 0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    run_args = {}
    mpath = args.init.parent / "metrics.jsonl"
    if mpath.is_file():
        run_args = next((r["args"] for r in map(json.loads, mpath.read_text().splitlines()) if "args" in r), {})
    if args.holdout is None:
        args.holdout = list(run_args.get("holdout", []))
    t0 = time.time()
    data = load(args.shards, args.hist)
    sp = splits(data, args.holdout, args.val_mod)
    rng = np.random.default_rng(args.seed)
    if args.max_train is not None:
        sp["train"] = rng.permutation(sp["train"])[: args.max_train]
        sp["val"], sp["holdout"] = sp["val"][: args.max_train // 4], sp["holdout"][: args.max_train // 4]
    sizes = {k: len(v) for k, v in sp.items()}
    print(f"loaded {len(data['y'])} states in {time.time() - t0:.0f}s; splits {sizes}", flush=True)
    if sizes["train"] == 0 or sizes["val"] == 0:
        alert(args.out, f"empty split {sizes}")
        sys.exit(3)
    if sizes["holdout"] == 0:
        alert(args.out, "no held-out-deck states: the deck generalization gate cannot be read")

    emb = dict(np.load(args.emb))
    trunk = PolicyNet(emb, d=run_args.get("d", 256), layers=run_args.get("layers", 4), heads=run_args.get("heads", 4),
                      hist=args.hist, known=True)  # fmt: skip
    if run_args.get("hist") or run_args.get("known"):
        raise SystemExit("init checkpoint already has hist/known parts: init_trunk assumes a v1 policy")
    state = torch.load(args.init, map_location="cpu")
    fresh = init_trunk(trunk, state.get("model", state))
    print(f"new trunk parameters (no-op init): {fresh}", flush=True)
    model = BeliefNet(trunk).to(device)
    new = [p for n, p in model.named_parameters() if not n.startswith("trunk.") or n[6:] in fresh]
    old = [p for n, p in model.named_parameters() if n.startswith("trunk.") and n[6:] not in fresh]
    if args.freeze_trunk:
        for p in old:
            p.requires_grad_(False)
    groups = [{"params": new, "lr": args.lr * 10}] + ([] if args.freeze_trunk else [{"params": old, "lr": args.lr}])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = math.ceil(sizes["train"] / args.batch)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=max(total, 1),
                                                pct_start=0.05)  # fmt: skip
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"device {device} amp {amp}; trainable {n_train / 1e6:.2f}M; {total} steps", flush=True)
    log = open(args.out / "metrics.jsonl", "a")  # noqa: SIM115 - open for the whole run
    log.write(json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                          "splits": sizes, "fresh": fresh}) + "\n")  # fmt: skip

    # invariant: the zero-initialized head reproduces the baseline (ratio 1.0) before any step
    probe = sp["val"][: args.eval_batch]
    r0 = evaluate(model, data, probe, args, device, amp=False)
    print(f"init probe {r0}", flush=True)
    if abs(r0["ratio"] - 1.0) > 1e-3:
        alert(args.out, f"untrained head ratio {r0['ratio']} != 1.0: the residual is not a no-op")
        sys.exit(3)

    def full_eval(epoch: float, step: int) -> dict:
        rec = {"epoch": round(epoch, 3), "step": step, "time_s": round(time.time() - t0)}
        for split in ("val", "holdout"):
            if len(sp[split]):
                rec[split] = evaluate(model, data, sp[split], args, device, amp)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        short = " ".join(f"{k} {v['ratio']} (t3+ {v['ratio_turn3+']})" for k, v in rec.items() if isinstance(v, dict))
        with open(args.out.parent / "PROGRESS", "a") as f:
            f.write(f"{args.out.name} ep {rec['epoch']} step {step} ratio {short}\n")
        return rec

    every = max(1, steps_per_epoch // args.evals_per_epoch)
    step, first = 0, None
    run_loss = run_n = 0.0
    for ep in range(args.epochs):
        order = rng.permutation(sp["train"])
        for s in range(0, len(order), args.batch):
            obs, t = batch(data, np.sort(order[s : s + args.batch]), args.hist, args.nohist, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = model(obs, t["cand"], t["cand_n"], t["base_p"])
            m = scored(t["cand"], t["base_p"])
            n = int(m.sum())
            if n == 0:
                continue
            loss = bce(lg.float(), t["y"], m) / n
            if not torch.isfinite(loss):
                alert(args.out, f"non-finite loss at step {step}")
                sys.exit(3)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss, run_n = run_loss + loss.item(), run_n + 1
            if step % 100 == 0:
                print(f"step {step}/{total} train bce {run_loss / run_n:.4f}", flush=True)
                run_loss = run_n = 0.0
            if step % every == 0:
                rec = full_eval(ep + s / len(order), step)
                if first is None:  # first milestone: the head must already beat the baseline on val
                    first = rec
                    if rec["val"]["ratio"] >= 1.0:
                        alert(args.out, f"val ratio {rec['val']['ratio']} >= 1 at the first eval (step {step})")
    rec = full_eval(args.epochs, step)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
               args.out / "final.pt")  # fmt: skip
    summary = {"final": True, "val_ratio": rec["val"]["ratio"],
               "holdout_ratio": rec.get("holdout", {}).get("ratio"),
               "val_ratio_turn3+": rec["val"]["ratio_turn3+"]}  # fmt: skip
    print(json.dumps(summary), flush=True)
    log.write(json.dumps(summary) + "\n")
    log.close()


if __name__ == "__main__":
    main()
