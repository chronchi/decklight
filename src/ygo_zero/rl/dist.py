"""Data-parallel (DDP) helpers for multi-GPU PPO (train_rl via torchrun).

Single-process behaviour is bit-identical: every helper is a no-op when the
torch.distributed process group is not initialized (WORLD_SIZE unset or 1).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main() -> bool:
    return rank() == 0


def init_from_env(device_type: str) -> tuple[int, int]:
    """Init the process group when torchrun set WORLD_SIZE > 1. Returns (rank, world).

    `nccl` on cuda, `gloo` on cpu. WORLD_SIZE unset or 1 → (0, 1), no process group.
    """
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    if world <= 1:
        return 0, 1
    backend = "nccl" if device_type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def teardown() -> None:
    if is_dist():
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist():
        dist.barrier()


def bcast_flag(flag: bool) -> bool:
    """Broadcast a boolean from rank 0 to all ranks (no-op single-process)."""
    if not is_dist():
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.long)
    dist.broadcast(t, src=0)
    return bool(t.item())


def sum_scalar(x: float, device: torch.device) -> float:
    """All-reduce sum of one float across ranks."""
    if not is_dist():
        return x
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def mean_scalar(x: float, device: torch.device) -> float:
    if not is_dist():
        return x
    return sum_scalar(x, device) / world_size()


def min_scalar(x: float, device: torch.device) -> float:
    if not is_dist():
        return x
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return float(t.item())


def max_scalar(x: float, device: torch.device) -> float:
    if not is_dist():
        return x
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def min_int(x: int, device: torch.device) -> int:
    """All-reduce MIN of one int (for the equal-minibatch-count rule)."""
    if not is_dist():
        return x
    t = torch.tensor([x], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def global_adv_stats(adv: np.ndarray, device: torch.device) -> tuple[float, float, int]:
    """Global mean/std/count of advantages across ranks (all-reduce sum, sumsq, count).

    Single-process → the local mean/std/count (caller normalizes identically).
    """
    adv = np.asarray(adv, dtype=np.float64)
    n = int(adv.size)
    s = float(adv.sum()) if n else 0.0
    ss = float((adv * adv).sum()) if n else 0.0
    if not is_dist():
        mean = s / n if n else 0.0
        var = ss / n - mean * mean if n else 0.0
        return mean, float(np.sqrt(max(var, 0.0))), n
    t = torch.tensor([s, ss, float(n)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    gs, gss, gn = float(t[0].item()), float(t[1].item()), int(t[2].item())
    if gn == 0:
        return 0.0, 0.0, 0
    mean = gs / gn
    var = gss / gn - mean * mean
    return mean, float(np.sqrt(max(var, 0.0))), gn


def global_normalize(adv: np.ndarray, device: torch.device) -> np.ndarray:
    """(adv - global_mean) / (global_std + 1e-8), same formula as train_rl's local norm."""
    adv = np.asarray(adv, dtype=np.float32)
    mean, std, _ = global_adv_stats(adv, device)
    return ((adv - mean) / (std + 1e-8)).astype(np.float32)


def average_grads_(module: torch.nn.Module) -> None:
    """Mean of every parameter gradient across ranks (in place; no-op single-process).

    This is the operative DDP sync for train_rl: the update uses custom forwards
    (`forward_value` / `forward_q` / `_run` + heads) that bypass
    DistributedDataParallel's `forward` hook, so the reducer never engages and the
    average is done explicitly after every backward, before grad clipping.
    Mathematically identical to DDP's gradient averaging.
    """
    if not is_dist() or dist.get_world_size() <= 1:
        return
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    # one collective per call (not one per parameter); which params have a grad is fixed by the code path,
    # so every rank flattens the same list
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(dist.get_world_size())
    for g, f in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads), strict=True):
        g.copy_(f)


def param_checksum(model: torch.nn.Module) -> float:
    """Sum of all policy params in float64 (one scalar per rank for the checksum check)."""
    tot = 0.0
    with torch.no_grad():
        for p in model.parameters():
            tot += float(p.detach().double().sum().item())
    return tot


def checksum_ok(model: torch.nn.Module, device: torch.device, tol: float = 1e-3) -> bool:
    """True when every rank's checksum matches (all-gather + compare, max-min <= tol)."""
    c = param_checksum(model)
    if not is_dist():
        return True
    t = torch.tensor([c], dtype=torch.float64, device=device)
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    vals = [float(g.item()) for g in gathered]
    return max(vals) - min(vals) <= tol


def sync_league_ema(league) -> None:
    """Average the league EMA win rates across ranks so every rank samples with the same weights.

    Roster (snapshots) is identical across ranks — weights are in sync via DDP and
    snapshots are taken at the same iteration — so the EMA key sets match. Keys are
    reduced in sorted order; missing keys are treated as 0.5 (the League default).
    """
    if not is_dist() or league is None:
        return
    # Union of keys across ranks would need another collective; the roster is
    # identical by construction, so the local key set suffices (base + snapshots).
    with league._lock:
        keys = sorted(set(league.ema) | {n for n in league.snaps} | {"base"})
    vals = torch.tensor([float(league.ema.get(k, 0.5)) for k in keys], dtype=torch.float64,
                        device=torch.device("cpu"))
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        vals = vals.to(torch.device("cuda"))
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals = vals / dist.get_world_size()
    with league._lock:
        for k, v in zip(keys, vals.tolist()):
            league.ema[k] = float(v)
