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
Fine-tuning: --init runs/<run>/bc/best.pt starts from those weights (architecture args are read from its metrics.jsonl)
and logs an "epoch": -1 record (the starting point's metrics) first; --epochs 0 --init = evaluate only.
--val-frac-kind gold=0.25 gold_pass=0.25 keeps a larger share of a scarce kind's lines in val;
--select-on val:gold picks best.pt by one kind's val accuracy.
--gold-games N (saturation curves): train only on the gold games among the first N by game number (g1, g2, m3 ... u103);
val/holdout stay the same for every N, so points of a curve share one evaluation set.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ygo_zero.model.bc_data import (
    KNOWN_KEYS,
    OBS_KEYS,
    check_eval_decks,
    eval_only_decks,
    history_window,
    load_shards,
    split_masks,
)
from ygo_zero.model.policy_v3 import make_policy


def batches(data, idx, bs, device, shuffle, rng, hist=0, hist_salient=True, known=False):
    if shuffle:
        idx = rng.permutation(idx)
    has_equiv = "equiv" in data
    if hist > 0 and "hlog" not in data:
        raise ValueError("--hist needs shards built with history logs (hlog): rebuild the shards")
    if known and not all(k in data for k in KNOWN_KEYS):
        raise ValueError("--known needs shards built with card knowledge (card_known/known_code): rebuild the shards")
    has_soft = "soft_idx" in data and "soft_q" in data
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        obs = {k: torch.from_numpy(data[k][j]).to(device, non_blocking=True) for k in OBS_KEYS}
        if known:
            for k in KNOWN_KEYS:
                obs[k] = torch.from_numpy(data[k][j]).to(device, non_blocking=True)
        if hist > 0:
            h, m = history_window(
                data["hlog"], data["hlog_start"][j], data["hlog_end"][j],
                data["me"][j], data["turn"][j], hist, salient=hist_salient,
            )
            obs["history"] = torch.from_numpy(h).to(device, non_blocking=True)
            obs["history_mask"] = torch.from_numpy(m).to(device, non_blocking=True)
        act = torch.from_numpy(data["action"][j]).to(device)
        if has_equiv:
            eq = torch.from_numpy(data["equiv"][j]).to(device, non_blocking=True)
        else:  # old shards: equiv = one-hot of the action
            eq = torch.zeros(len(j), obs["actions"].shape[1], dtype=torch.bool, device=device)
            eq[torch.arange(len(j), device=device), act] = True
        if has_soft:
            si = torch.from_numpy(data["soft_idx"][j]).to(device, non_blocking=True)
            sq = torch.from_numpy(data["soft_q"][j]).to(device, non_blocking=True)
        else:
            si, sq = None, None
        yield (
            obs,
            act,
            torch.from_numpy(data["weight"][j]).to(device),
            eq,
            si,
            sq,
            j,
        )


def set_loss(logits: torch.Tensor, equiv: torch.Tensor) -> torch.Tensor:
    """Per-sample -log(sum of softmax over the equiv set): lse(all) - lse(equiv)."""
    return torch.logsumexp(logits, -1) - torch.logsumexp(
        logits.masked_fill(~equiv, float("-inf")), -1
    )


def kl_to_init(logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(pi_init || pi) over legal options only: sum_legal p_ref * (logp_ref - logp).

    All inputs are [B, A]; mask is bool. Computed in float32; illegal entries are masked out of
    the sum so 0 * -inf never produces NaN.
    """
    logp = torch.log_softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)
    logp_ref = torch.log_softmax(ref_logits.float().masked_fill(~mask, float("-inf")), dim=-1)
    p_ref = logp_ref.exp()
    diff = torch.where(mask, logp_ref - logp, torch.zeros((), device=logp.device, dtype=logp.dtype))
    return torch.where(mask, p_ref * diff, torch.zeros((), device=logp.device, dtype=logp.dtype)).sum(-1)


def gumbel_soft_target(
    ref_logits: torch.Tensor, mask: torch.Tensor, soft_idx: torch.Tensor | None,
    soft_q: torch.Tensor | None, scale: float, norm: str = "minmax",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel-MuZero-style improved policy pi' = softmax(prior + c * q_hat) over legal options.

    ref_logits/mask are [B, A] (mask bool, legal options); soft_idx [B, K] holds searched option
    indices (-1 padded) with selection scores soft_q [B, K] aligned to them. The completed Q is the
    search score on searched candidates and v_mix (the prior-weighted mean candidate score) on
    unsearched legal options; q_hat is the min-max normalised Q (all zeros when max == min, so the
    target is then just the prior). Returns (pi_prime [B, A], has_soft [B]): rows without (legal)
    search info carry the prior and False. Illegal entries are exactly 0.
    """
    B, A = ref_logits.shape
    device = ref_logits.device
    logp_prior = torch.log_softmax(ref_logits.float().masked_fill(~mask, float("-inf")), dim=-1)
    prior = logp_prior.exp()
    if soft_idx is None or soft_q is None:
        return prior, torch.zeros(B, dtype=torch.bool, device=device)
    idx = soft_idx.long().clamp(min=0)
    valid = (soft_idx >= 0) & (soft_idx < A)
    # padded slots point at a scratch column (dropped below): scatter_ with duplicate indices is
    # unordered, so pads must never share a column with a real candidate
    idx_safe = torch.where(valid, idx, torch.full((), A, device=device))
    is_cand = torch.zeros((B, A + 1), dtype=torch.bool, device=device)
    is_cand.scatter_(1, idx_safe, valid)
    is_cand = is_cand[:, :A]
    searched = is_cand & mask
    has = searched.any(-1)
    qvals = torch.zeros((B, A + 1), dtype=torch.float32, device=device)
    qvals.scatter_(1, idx_safe, torch.where(valid, soft_q.float(), torch.zeros((), device=device)))
    qvals = qvals[:, :A]
    w = torch.where(searched, prior, torch.zeros((), device=device, dtype=prior.dtype))
    v_mix = (w * torch.where(searched, qvals, torch.zeros((), device=device))).sum(-1, keepdim=True) / w.sum(
        -1, keepdim=True
    ).clamp(min=1e-12)
    q_full = torch.where(searched, qvals, v_mix)
    mn = torch.where(mask, q_full, torch.full((), float("inf"), device=device)).amin(-1, keepdim=True)
    mx = torch.where(mask, q_full, torch.full((), float("-inf"), device=device)).amax(-1, keepdim=True)
    if norm == "minmax":
        q_hat = torch.where(mask, (q_full - mn) / ((mx - mn) + 1e-6), torch.zeros((), device=device))
    else:  # "none": raw value scores (in [-1, 1]); a near-tie stays a near-tie instead of being stretched to c
        q_hat = torch.where(mask, q_full, torch.zeros((), device=device))
    improved = logp_prior + scale * q_hat
    improved = improved.masked_fill(~mask, float("-inf"))
    return torch.softmax(improved.float(), -1), has


def evaluate(model, data, idx, bs, device, amp, set_target=True, hist=0, hist_salient=True, known=False,
             ref=None):
    model.eval()
    tot = {
        "n": 0, "loss": 0.0, "acc": 0.0, "acc_exact": 0.0,
        "n_stated": 0, "acc_stated": 0.0, "chance": 0.0,
    }
    per_deck: dict[str, list[int]] = {}
    per_kind: dict[str, list[int]] = {}
    per_turn: dict[str, list[int]] = {}
    has_kind = "kind" in data
    has_turn = "turn" in data
    with torch.no_grad():
        kl_sum = 0.0
        for obs, act, w, eq, _si, _sq, j in batches(data, idx, bs, device, False, None, hist, hist_salient, known):
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
            if ref is not None:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    ref_logits = ref(obs)
                kl_sum += kl_to_init(logits, ref_logits, obs["action_mask"]).sum().item()
            for d, h in zip(data["deck"][j], hit.tolist(), strict=True):
                per_deck.setdefault(str(d), [0, 0])
                per_deck[str(d)][0] += 1
                per_deck[str(d)][1] += int(h)
            if has_kind:
                for kd, h in zip(data["kind"][j], hit.tolist(), strict=True):
                    per_kind.setdefault(str(kd), [0, 0])
                    per_kind[str(kd)][0] += 1
                    per_kind[str(kd)][1] += int(h)
            if has_turn:
                for t, h in zip(data["turn"][j], hit.tolist(), strict=True):
                    bkt = "t1" if int(t) <= 1 else ("t2-3" if int(t) <= 3 else "t4+")
                    per_turn.setdefault(bkt, [0, 0])
                    per_turn[bkt][0] += 1
                    per_turn[bkt][1] += int(h)
    model.train()
    n = max(tot["n"], 1)
    rec = {
        "n": tot["n"],
        "loss": tot["loss"] / n,
        "acc": tot["acc"] / n,
        "acc_exact": tot["acc_exact"] / n,
        "acc_stated": tot["acc_stated"] / max(tot["n_stated"], 1),
        "chance": tot["chance"] / n,  # expected accuracy of a uniform pick among the legal options
        "per_deck": {d: round(h / c, 4) for d, (c, h) in sorted(per_deck.items())},
        "per_kind": {k: round(h / c, 4) for k, (c, h) in sorted(per_kind.items())},
    }
    if has_turn:
        rec["per_turn"] = {k: round(per_turn[k][1] / per_turn[k][0], 4) for k in ("t1", "t2-3", "t4+") if k in per_turn}
    if ref is not None:
        rec["kl_init"] = kl_sum / n
    return rec


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


def gold_game_number(line: str) -> int:
    """Game number of a gold line id ("<d0>-vs-<d1>/gold-u24" -> 24, "gold-g1" -> 1); -1 when it has none."""
    m = re.search(r"gold-[a-z]*(\d+)$", line)
    return int(m.group(1)) if m else -1


def limit_gold_games(data, train_idx: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Drop training samples of gold kinds whose game is not among the first n gold games (by game number, counted
    over every split so the cut is the same whatever the val/holdout membership). Returns (indices, games kept in
    train)."""
    kind = data["kind"]
    gold = np.isin(kind, ["gold", "gold_pass"])
    nums = sorted({gold_game_number(str(x)) for x in np.unique(data["line"][gold])} - {-1})
    allowed = set(nums[:n])
    tr_gold = gold[train_idx]
    keep_gold = np.array([gold_game_number(str(x)) in allowed for x in data["line"][train_idx[tr_gold]]], bool)
    keep = np.ones(len(train_idx), bool)
    keep[np.flatnonzero(tr_gold)] = keep_gold
    kept_games = {gold_game_number(str(x)) for x in data["line"][train_idx[tr_gold][keep_gold]]}
    return train_idx[keep], len(kept_games)


def limit_train_frac(data, train_idx: np.ndarray, frac: float) -> tuple[np.ndarray, int, int]:
    """Keep the train samples of a fixed share of games (line ids): crc32(line) % 10000 < frac * 10000, so a smaller
    share is a subset of a larger one and val/holdout are untouched (learning curves share one evaluation set).
    Returns (indices, games kept, games in train)."""
    lines = data["line"][train_idx]
    uniq = np.unique(lines)
    keep_lines = {str(x) for x in uniq if zlib.crc32(str(x).encode()) % 10000 < frac * 10000}
    keep = np.array([str(x) in keep_lines for x in lines], bool)
    return train_idx[keep], len(keep_lines), len(uniq)


def select_score(rec: dict, select_on: str, default: float) -> float:
    """--select-on value of a metrics record: "<split>" -> its acc, "<split>:<kind>" -> that kind's acc in the split,
    "last" -> the epoch number (best.pt = the final epoch; never the --init weights)."""
    if select_on == "last":
        return float(rec.get("epoch", -1)) if "epoch" in rec else default
    split, _, kind = select_on.partition(":")
    m = rec.get(split, {})
    v = m.get("per_kind", {}).get(kind) if kind else m.get("acc")
    return default if v is None else float(v)


def _run_args(ckpt: Path) -> dict:
    """Training args of a checkpoint (first metrics.jsonl record with "args" next to it)."""
    m = ckpt.parent / "metrics.jsonl"
    for raw in m.read_text().splitlines() if m.is_file() else []:
        rec = json.loads(raw)
        if "args" in rec:
            return rec["args"]
    return {}


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
    ap.add_argument("--select-on", default="val", help="val | holdout | <split>:<kind> (e.g. val:gold)")
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--card-dropout", type=float, default=0.0)
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="decks never trained on (default: decks/pool/HOLDOUT + kewl-tune; any holdout deck present in "
                    "the data but not listed stops the run)")
    ap.add_argument("--kinds", nargs="*", default=None, help="train only on these sample kinds (default: all)")
    ap.add_argument("--require-kinds", nargs="*", default=None,
                    help="exit 4 with an ALERT when any of these kinds has 0 train samples (the variant's new data)")
    ap.add_argument("--kind-frac", nargs="*", default=None, metavar="KIND=FRAC",
                    help="per-epoch share of each kind's train samples (<1 subsample, >1 repeat)")
    ap.add_argument("--merge-last", type=int, default=0, help="also evaluate the average of the last K epochs")
    ap.add_argument("--init", type=Path, default=None, help="start from this checkpoint (fine-tune)")
    ap.add_argument("--kl-init", type=float, default=0.0,
                    help="KL(pi_init || pi) anchor coef to the --init weights (0 = off)")
    ap.add_argument("--soft-targets", choices=["off", "gumbel"], default="off",
                    help="off = hard targets (default, unchanged); gumbel = distill the search improved "
                    "policy softmax(prior + c*q_hat) on samples with search info (requires --init)")
    ap.add_argument("--soft-scale", type=float, default=5.0,
                    help="c in softmax(prior + c*q_hat) for --soft-targets gumbel")
    ap.add_argument("--soft-norm", choices=["none", "minmax"], default="none",
                    help="q_hat = raw value score in [-1, 1] (none) or min-max over the legal options (minmax, "
                    "Gumbel MuZero; stretches near-ties to the full scale c)")
    ap.add_argument("--gold-games", type=int, default=None, help="train on the first N gold games only (by number)")
    ap.add_argument("--val-frac-kind", nargs="*", default=None, metavar="KIND=FRAC",
                    help="val share for these kinds' lines (default 0.1 for every kind)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="train on a fixed share of the train games (crc32 of line id; nested across shares)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N samples (smoke)")
    ap.add_argument("--hist", type=int, default=0, help="long salient history window H (0 = v1 stored history)")
    ap.add_argument("--hist-all", action="store_true", help="with --hist: use all entries, not just salient ones")
    ap.add_argument("--v3", action="store_true", help="PolicyNetV3 (model/policy_v3.py: effect keys, sentences, counts)")
    ap.add_argument("--desc-strings", default=None, help="v3 effect-string table (default data/cards/emb/strings-*.npz)")
    ap.add_argument("--desc-lua", default=None, help="v3 Lua per-description table (default data/cards/lua_desc.npz)")
    ap.add_argument("--known", action="store_true", help="card-knowledge memory: feed card_known/known_code (model + shards need them)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (laptop: <= 2)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.set_target and args.label_smoothing:
        ap.error("--set-target and --label-smoothing are mutually exclusive")
    if args.kl_init and not args.init:
        print("error: --kl-init requires --init", flush=True)
        raise SystemExit(2)
    if args.soft_targets == "gumbel" and not args.init:
        print("error: --soft-targets gumbel requires --init (the search prior)", flush=True)
        raise SystemExit(2)
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
    if args.holdout is None:
        args.holdout = sorted(eval_only_decks()[0])
    problems = check_eval_decks(data["deck"], args.holdout)
    if problems:
        for msg in problems:
            print(f"ALERT train_bc: {msg}", flush=True)
        sys.exit(3)
    masks = split_masks(data, holdout_decks=tuple(args.holdout), kind_val_frac=parse_kind_frac(args.val_frac_kind))
    idx = select_train_idx(data, masks, args.kinds)
    if args.gold_games is not None:
        idx["train"], kept = limit_gold_games(data, idx["train"], args.gold_games)
        print(f"--gold-games {args.gold_games}: training on {kept} gold games (their train split)", flush=True)
    if args.train_frac < 1.0:
        idx["train"], kg, ng = limit_train_frac(data, idx["train"], args.train_frac)
        print(f"--train-frac {args.train_frac}: training on {kg}/{ng} games ({len(idx['train'])} samples)", flush=True)
        if len(idx["train"]) == 0:
            print("ALERT train_bc: --train-frac left 0 train samples", flush=True)
            sys.exit(3)
    print({k: len(v) for k, v in idx.items()}, "decks:", sorted(set(map(str, data["deck"]))), flush=True)

    if args.init:  # the starting checkpoint fixes the architecture
        src_args = _run_args(args.init)
        args.d, args.layers, args.heads = src_args.get("d", args.d), src_args.get("layers", args.layers), src_args.get("heads", args.heads)
        args.hist = src_args.get("hist", args.hist or 0)
        args.known = src_args.get("known", args.known or False)
        args.v3 = src_args.get("v3", args.v3 or False)
        args.desc_strings = src_args.get("desc_strings", args.desc_strings)
        args.desc_lua = src_args.get("desc_lua", args.desc_lua)
    model = make_policy(args.emb, vars(args), dropout=args.dropout, card_dropout=args.card_dropout).to(device)
    if args.init:
        state = torch.load(args.init, map_location="cpu")
        state = state.get("model", state)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    ref = None
    if args.kl_init or args.soft_targets == "gumbel":
        import copy

        ref = copy.deepcopy(model).eval().requires_grad_(False)
    soft = args.soft_targets == "gumbel"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params {n_params / 1e6:.2f}M device {device} amp {amp}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    kind_frac = parse_kind_frac(args.kind_frac)
    n_epoch = len(epoch_train_idx(data, idx["train"], kind_frac, np.random.default_rng(0)))
    print(f"train samples per epoch {n_epoch} (kind_frac {kind_frac or 'none'})", flush=True)
    if args.require_kinds and "kind" in data:
        # fail loud: a kind this variant exists for with 0 train samples means its data never reached the shards;
        # the variant would silently train as another one
        have = set(np.unique(data["kind"][idx["train"]]).tolist())
        empty = [k for k in args.require_kinds if k not in have]
        if empty:
            print(f"ALERT {args.out.name}: --require-kinds with 0 train samples: {empty}", flush=True)
            raise SystemExit(4)
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
    best_epoch = None  # epoch of best.pt; -1 = the --init weights
    if args.init:
        rec = {"epoch": -1, "init": str(args.init), "args": args_rec}
        for split in ("val", "holdout"):
            if len(idx[split]):
                rec[split] = evaluate(model, data, idx[split], args.batch * 2, device, amp, args.set_target,
                                      args.hist, not args.hist_all, args.known, ref=ref)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        best = select_score(rec, args.select_on, -1.0)
        best_epoch = -1
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, args.out / "best.pt")
    for ep in range(args.epochs):
        run = {"loss": 0.0, "n": 0, "kl": 0.0, "soft_n": 0, "soft_ent": 0.0, "soft_agree": 0.0}
        ep_idx = epoch_train_idx(data, idx["train"], kind_frac, rng)
        for obs, act, w, eq, si, sq, j in batches(data, ep_idx, args.batch, device, True, rng,
                                            args.hist, not args.hist_all, args.known):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(obs)
            if args.set_target:
                per = set_loss(logits.float(), eq)
            else:
                per = F.cross_entropy(
                    logits.float(), act, reduction="none", label_smoothing=args.label_smoothing
                )
            if ref is not None:
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    ref_logits = ref(obs)
                if soft:
                    mask = obs["action_mask"]
                    pi_prime, has_soft = gumbel_soft_target(
                        ref_logits.float(), mask, si, sq, args.soft_scale, args.soft_norm)
                    if not torch.isfinite(pi_prime).all():
                        print(f"ALERT {args.out.name}: non-finite soft target in epoch {ep}", flush=True)
                        raise SystemExit(3)
                    if has_soft.any():
                        # CE(pi', p): legal options only, via zero-filled factors so 0 * -inf never forms
                        logp = torch.log_softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)
                        pp = torch.where(mask, pi_prime, torch.zeros((), device=logits.device))
                        lp = torch.where(mask, logp, torch.zeros((), device=logits.device))
                        per = torch.where(has_soft, -(pp * lp).sum(-1), per)
                        ent = -(pp * torch.where(mask, pi_prime.clamp_min(1e-30).log(),
                                                torch.zeros((), device=logits.device))).sum(-1)
                        run["soft_n"] += int(has_soft.sum())
                        run["soft_ent"] += ent[has_soft].sum().item()
                        run["soft_agree"] += (pi_prime.argmax(-1) == act)[has_soft].sum().item()
                kl = kl_to_init(logits.float(), ref_logits.float(), obs["action_mask"])
                loss = (per * w).sum() / w.sum() + args.kl_init * kl.mean()
            else:
                loss = (per * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run["loss"] += loss.item() * len(j)
            run["n"] += len(j)
            if ref is not None:
                run["kl"] += kl.detach().sum().item()
        if ref is not None:
            train_kl = run["kl"] / max(run["n"], 1)
            if not math.isfinite(train_kl):
                print(f"ALERT {args.out.name}: KL is NaN", flush=True)
                raise SystemExit(3)
        if soft:
            if run["soft_n"] == 0:
                print(f"ALERT {args.out.name}: --soft-targets gumbel with 0 soft samples in epoch {ep}",
                      flush=True)
                raise SystemExit(3)
        if args.merge_last > 1:
            recent = (recent + [{k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}])[-args.merge_last:]
        rec = {"epoch": ep, "time": round(time.time() - t0, 1), "n_train": len(ep_idx), "train_loss": run["loss"] / max(run["n"], 1)}
        if soft:
            rec["soft_n"] = run["soft_n"]
            rec["soft_ent"] = run["soft_ent"] / run["soft_n"]
            rec["soft_argmax_agree"] = run["soft_agree"] / run["soft_n"]
        if ref is not None:
            rec["train_kl"] = train_kl
        if ep == 0:
            rec["args"] = args_rec
        for split in ("val", "holdout"):
            if len(idx[split]):
                rec[split] = evaluate(
                    model, data, idx[split], args.batch * 2, device, amp, args.set_target,
                    args.hist, not args.hist_all, args.known, ref=ref,
                )
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "args": vars(args)},
            args.out / "last.pt",
        )
        if args.select_on == "last":
            # every epoch is a candidate for the report card (fp16, ~9 MB): a later epoch can regress
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                       args.out / f"ep{ep}.pt")
        score = select_score(rec, args.select_on, -rec["train_loss"])
        if score > best:
            best = score
            best_epoch = ep
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
                rec[split] = evaluate(model, data, idx[split], args.batch * 2, device, amp, args.set_target,
                                      args.hist, not args.hist_all, args.known, ref=ref)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        score = select_score(rec, args.select_on, -1.0)
        if score > best:
            print(f"merge of last {len(recent)} beats best ({score:.4f} > {best:.4f}): merged.pt + best.pt", flush=True)
            best = score
            best_epoch = "merged"
            half = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
            torch.save(half, args.out / "merged.pt")
            torch.save(half, args.out / "best.pt")
        model.load_state_dict(live)
    # final weights always kept (fp16, small): best.pt may be an earlier epoch, and jobs delete last.pt
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, args.out / "final.pt")
    print(f"done in {time.time() - t0:.0f}s, best {args.select_on} acc {best:.4f}", flush=True)
    if args.init and args.epochs > 0 and best_epoch == -1:
        # fail loud (mix-20261005-1738): a fine-tune whose best.pt is still the --init weights would be playtested as
        # a copy of its base and waste the whole report card
        msg = (f"ALERT {args.out.name}: best.pt is the --init weights: --select-on {args.select_on} never beat the "
               f"starting point ({best:.4f}); trained weights are in final.pt")
        print(msg, flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
