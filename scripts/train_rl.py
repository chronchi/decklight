"""RL v0: PPO self-play from a BC checkpoint, KL-anchored to the frozen BC policy (PLAN Phase 5, first feel).

    uv run python scripts/train_rl.py --init runs/<run>/bc/best.pt --out runs/<name>/rl --hours 1.5 --device cuda

Rollouts: `--envs` C++ envs over the pool decks (holdout excluded) + the gauntlet decks. Each episode is either
self-play (both seats sampled from the learner, both trained) or, with probability `--p-bc`, learner vs the frozen BC
policy (sampled, temperature 1; only the learner seat is trained). Reward +1/-1 at the end, 0 draw, `--trunc-reward`
for both seats when a game hits `--max-decisions`. Value head new (trained alone for `--value-warmup` iterations).
Loss: clipped PPO + value MSE - entropy + `--kl-coef` * KL(pi || pi_BC) over legal options.

Progress is measured only on the fixed gauntlet every `--eval-every` iterations (and at iteration 0 / the end):
greedy learner vs greedy BC (both seats, 4 train + 4 holdout decks) and vs ygo-agent 0546 (its 31 lists in
decks/gauntlet, each vs its next 4), via the same
play_match/BCPolicy path as scripts/playtest.py. Never read the training win rates as progress.

Output (`--out`): metrics.jsonl (first record "args" carries the BC architecture so BCPolicy/playtest load the
checkpoints), iter-NNNN.pt (fp16 policy weights at each eval, no value head), last.pt (full, with optimizer).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.bc_policy import BCPolicy, _run_args
from ygo_zero.eval.match import play_match, ring_pairs, wilson
from ygo_zero.model.policy import PolicyNet
from ygo_zero.rl.ppo import SlotStore, gae, gather, masked_log_softmax, ppo_loss, to_torch
from ygo_zero.rl.rollout import Collector

ROOT = Path(__file__).resolve().parents[1]


def _effective_cores() -> int:
    """Usable CPUs: sched affinity count, capped by the cgroup quota when set."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 4
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = int(parts[0]), int(parts[1])
            if period > 0:
                n = min(n, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, n)


def _autocast(dev: torch.device, prec: str):
    if prec == "bf16" and dev.type == "cuda":
        return torch.autocast(dev.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_net(ckpt: Path, emb: dict, a: dict, value: bool) -> PolicyNet:
    net = PolicyNet(emb, d=a.get("d", 256), layers=a.get("layers", 4), heads=a.get("heads", 4), dropout=0.0,
                    hist=int(a.get("hist", 0) or 0), known=bool(a.get("known", False)), value=value)  # fmt: skip
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()},
                                              strict=False)  # fmt: skip
    assert not unexpected and all(k.startswith("value_head.") for k in missing), (missing, unexpected)
    return net


@torch.no_grad()
def sample(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    lp = masked_log_softmax(logits)
    p = lp.exp() * torch.isfinite(logits)
    a = torch.multinomial(p, 1).squeeze(1)
    return a.cpu().numpy().astype(np.int32), lp.gather(1, a[:, None]).squeeze(1).cpu().numpy()


def policy_state(model: PolicyNet) -> dict:
    return {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()
            if not k.startswith("value_head.")}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, required=True, help="BC checkpoint (its metrics.jsonl gives the architecture)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--envs", type=int, default=512)  # open trajectories must fit --store (≈ envs × 550)
    ap.add_argument("--threads", type=int, default=_effective_cores())
    ap.add_argument("--recv-batch", type=int, default=None, help="async recv batch (default: envs // 4)")
    ap.add_argument("--precision", default=None, help="fp32|bf16 (default: bf16 on cuda, fp32 on cpu)")
    ap.add_argument("--max-upload-gb", type=float, default=6.0, help="device-upload-once budget per iteration")
    ap.add_argument("--max-decisions", type=int, default=1500, help="per game, both seats")
    ap.add_argument("--trunc-reward", type=float, default=-0.5)
    ap.add_argument("--p-bc", type=float, default=0.3, help="share of episodes vs the frozen BC policy")
    ap.add_argument("--rollout", type=int, default=32768, help="finished samples per PPO iteration")
    ap.add_argument("--store", type=int, default=400_000, help="sample slots (open + finished trajectories)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--value-warmup", type=int, default=2, help="first iterations train the value head only")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=2, help="vs BC: seeds per ordered deck pair")
    ap.add_argument("--ga-seeds", type=int, default=1, help="vs ygo-agent: seeds per deck pair")
    ap.add_argument("--gauntlet-decks", type=Path, default=ROOT / "decks/gauntlet",
                    help="ygo-agent's own lists (in its 864-card vocabulary); also in the training pool")
    ap.add_argument("--ga-opps", type=int, default=4, help="vs ygo-agent: each gauntlet deck vs its next k decks")
    ap.add_argument("--no-ygoagent", action="store_true")
    ap.add_argument("--hours", type=float, default=1.0, help="stop (then final eval) after this much training")
    ap.add_argument("--iters", type=int, default=0, help="stop after this many iterations (0 = by --hours)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    if args.recv_batch is None:
        args.recv_batch = max(1, args.envs // 4)
    if args.precision is None:
        args.precision = "bf16" if dev.type == "cuda" else "fp32"

    def fail(msg: str):
        line = f"ALERT train_rl: {msg}"
        print(line, flush=True)
        try:
            with open(args.out / "ALERT", "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        raise SystemExit(3)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    # decks: training pool = pool - holdout + gauntlet; eval = fixed 4 train + 4 holdout sample
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    pool = sorted((ROOT / "decks/pool").glob("*.ydk"))
    gfiles = sorted(args.gauntlet_decks.glob("*.ydk"))
    train_decks = [load_ydk(p) for p in pool if p.stem not in hold] + [load_ydk(p) for p in gfiles]
    r = random.Random(1234)
    ev_names = r.sample([p.stem for p in pool if p.stem not in hold], 4) + r.sample(sorted(hold & {p.stem for p in pool}), 4)
    ev_decks = [load_ydk(ROOT / "decks/pool" / f"{n}.ydk") for n in ev_names]
    g_decks = [load_ydk(p) for p in gfiles]

    a = _run_args(args.init)
    hist, known = int(a.get("hist", 0) or 0), bool(a.get("known", False))
    emb = dict(np.load(args.emb))
    model = load_net(args.init, emb, a, value=True).to(dev).eval()
    ref = load_net(args.init, emb, a, value=False).to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    vparams = list(model.value_head.parameters())
    trunk = [p for n, p in model.named_parameters() if not n.startswith("value_head.")]
    opt = torch.optim.AdamW([{"params": trunk}, {"params": vparams, "lr": 1e-4}], lr=args.lr, weight_decay=0.0)
    vopt = torch.optim.AdamW(model.value_head.parameters(), lr=3e-4)

    log = (args.out / "metrics.jsonl").open("a")
    arch = {k: a.get(k) for k in ("d", "layers", "heads", "hist", "known")}
    log.write(json.dumps({"args": {**arch, "rl": {k: str(v) for k, v in vars(args).items()}}, "eval_decks": ev_names,
                          "train_decks": len(train_decks)}) + "\n")  # fmt: skip
    log.flush()

    engine = Engine()
    ga = None
    if not args.no_ygoagent:
        from ygo_zero.eval.ygoagent import YgoAgentPolicy

        ga = YgoAgentPolicy(seed=3)
    base_pol = BCPolicy(args.init, args.emb, device=args.device)

    def evaluate(it: int) -> dict:
        t = time.time()
        ck = args.out / f"iter-{it:04d}.pt"
        torch.save(policy_state(model), ck)
        pol = BCPolicy(ck, args.emb, device=args.device)
        res: dict = {"iter": it, "eval": {}}
        pairs = [(i, j) for i in range(len(ev_decks)) for j in range(len(ev_decks)) if i != j]
        games = play_match(engine, ev_decks, pol, base_pol, pairs, range(args.eval_seeds), num_envs=128,
                           num_threads=args.threads)  # fmt: skip
        for name, gs in (("vs_bc", games), ("vs_bc_holdout", [g for g in games if ev_names[g.deck_a] in hold])):
            w = sum(g.winner_is_a for g in gs)
            lo, hi = wilson(w, len(gs))
            res["eval"][name] = {"n": len(gs), "wr": round(w / len(gs), 3), "ci": [round(lo, 3), round(hi, 3)],
                                 "cut": sum(g.truncated for g in gs)}  # fmt: skip
        if ga is not None:
            ga._states.clear()  # LSTM states are keyed by schedule index, which restarts at 0 every match
            gp = ring_pairs(len(g_decks), args.ga_opps)
            games = play_match(engine, g_decks, pol, ga, gp, range(args.ga_seeds), num_envs=64, num_threads=args.threads)
            w = sum(g.winner_is_a for g in games)
            lo, hi = wilson(w, len(games))
            res["eval"]["vs_ygoagent"] = {"n": len(games), "wr": round(w / len(games), 3),
                                          "ci": [round(lo, 3), round(hi, 3)], "cut": sum(g.truncated for g in games)}  # fmt: skip
        res["eval_seconds"] = round(time.time() - t)
        print(f"[eval it {it}] " + "  ".join(f"{k} {v['wr']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] n={v['n']} cut={v['cut']}"
                                             for k, v in res["eval"].items()) + f"  ({res['eval_seconds']} s)", flush=True)  # fmt: skip
        log.write(json.dumps(res) + "\n")
        log.flush()
        return res

    evaluate(0)

    # rollout state: async collector owns the envs + trajectory bookkeeping
    N = args.envs
    vec = VecEnv(engine, train_decks, N, args.threads, seed=args.seed)
    vec.set_max_decisions(args.max_decisions)
    store = SlotStore(args.store, gather(vec.buf, np.arange(1), hist, known))
    col = Collector(vec, model, ref, store, hist, known, dev, args.p_bc, args.trunc_reward,
                    args.recv_batch, rng, args.precision)
    it, t_train = 0, time.time()
    tot = {"games": 0, "trunc": 0, "decisions": 0}
    while True:
        it += 1
        t_it = time.time()
        finished, st = col.collect(args.rollout)
        t_roll = time.time() - t_it
        if not finished:
            fail(f"iteration {it}: collected 0 samples")

        # GAE per finished seat trajectory, then PPO epochs
        slots = np.concatenate([np.array(s, np.int64) for s, _ in finished])
        if len(slots) == 0:
            fail(f"iteration {it}: collected 0 samples")
        advs, rets = [], []
        for s, rew in finished:
            ad, rt = gae(store.value[np.array(s)], rew, args.lam)
            advs.append(ad)
            rets.append(rt)
        adv, ret = np.concatenate(advs), np.concatenate(rets)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        warm = it <= args.value_warmup
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        # upload the iteration's samples to the device ONCE, index minibatches there
        per_slot = (sum(int(np.prod(v.shape[1:], dtype=np.int64)) * np.dtype(v.dtype).itemsize
                        for v in store.obs.values()) + 8 + 4 + 4 + 64 * 2 + 4 + 4)
        upload_once = per_slot * len(slots) <= args.max_upload_gb * 1e9
        obs_dev = act_dev = olp_dev = ref_dev = adv_dev = ret_dev = perm_dev = None
        if upload_once:
            obs_dev = {k: torch.from_numpy(np.ascontiguousarray(store.obs[k][slots])).to(dev)
                       for k in store.obs}
            act_dev = torch.from_numpy(np.ascontiguousarray(store.action[slots])).to(dev)
            olp_dev = torch.from_numpy(np.ascontiguousarray(store.logp[slots])).to(dev)
            ref_dev = torch.from_numpy(np.ascontiguousarray(store.ref_lp[slots])).to(dev)
            adv_dev = torch.from_numpy(np.ascontiguousarray(adv_n)).to(dev)
            ret_dev = torch.from_numpy(np.ascontiguousarray(ret)).to(dev)
            perm_dev = torch.from_numpy(rng.permutation(len(slots))).to(dev)
        t_up = time.time()
        model.train()
        agg: dict[str, list] = {}
        bad_loss = None
        for _ in range(args.epochs):
            perm = rng.permutation(len(slots))
            mb_dev = torch.from_numpy(perm).to(dev) if upload_once else None
            for k in range(0, len(perm), args.minibatch):
                if upload_once:
                    idx = mb_dev[k : k + args.minibatch]
                    obs = {key: v[idx] for key, v in obs_dev.items()}
                    act, olp = act_dev[idx], olp_dev[idx]
                    rl = ref_dev[idx] if args.kl_coef > 0 else None
                    a_t = adv_dev[idx]
                    r_t = ret_dev[idx]
                else:
                    mb = perm[k : k + args.minibatch]
                    sl = slots[mb]
                    obs = to_torch({key: v[sl] for key, v in store.obs.items()}, dev)
                    act = torch.from_numpy(store.action[sl]).to(dev)
                    olp = torch.from_numpy(store.logp[sl]).to(dev)
                    rl = torch.from_numpy(store.ref_lp[sl]).to(dev) if args.kl_coef > 0 else None
                    a_t = torch.from_numpy(adv_n[mb]).to(dev)
                    r_t = torch.from_numpy(ret[mb]).to(dev)
                if warm:
                    with _autocast(dev, args.precision):
                        with torch.no_grad():  # value head only: no graph through the trunk
                            _, enc = model._run(obs)
                        v = model.value_head(enc[:, 0]).squeeze(-1)
                        loss = torch.nn.functional.mse_loss(v.float(), r_t)
                    if not bool(torch.isfinite(loss).item()):
                        bad_loss = loss.item()
                        break
                    vopt.zero_grad(set_to_none=True)
                    loss.backward()
                    vopt.step()
                    s = {"vf": loss.item()}
                else:
                    with _autocast(dev, args.precision):
                        loss, s = ppo_loss(model, rl, obs, act, olp, a_t, r_t, args.clip, args.vf_coef,
                                           args.kl_coef, args.ent_coef)  # fmt: skip
                    if not bool(torch.isfinite(loss).item()):
                        bad_loss = loss.item()
                        break
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                for key, val in s.items():
                    agg.setdefault(key, []).append(val)
            if bad_loss is not None:
                break
        if bad_loss is not None:
            fail(f"iteration {it}: non-finite loss {bad_loss}")
        t_update = time.time() - t_up
        model.eval()
        for s, _ in finished:
            store.release(s)

        if not warm and args.kl_coef > 0 and agg.get("kl_bc") and float(np.mean(agg["kl_bc"])) == 0.0:
            fail(f"iteration {it}: kl_bc exactly 0.0 with --kl-coef {args.kl_coef}")
        dt = time.time() - t_it
        tot["games"] += st["games"]
        tot["trunc"] += st["trunc"]
        tot["decisions"] += st["decisions"]
        rec = {
            "iter": it, "warmup": warm, "samples": len(slots), "games": st["games"],
            "trunc_rate": round(st["trunc"] / max(st["games"], 1), 3),
            "mean_game_decisions": round(float(np.mean(st["lengths"])) if st["lengths"] else 0.0, 1),
            "train_vs_bc_sampled": round(st["bc_wins"] / max(st["bc_games"], 1), 3), "bc_games": st["bc_games"],
            "adv_std": round(float(adv.std()), 3), "ret_mean": round(float(ret.mean()), 3),
            "dps": round(st["decisions"] / t_roll), "rollout_s": round(t_roll, 1), "iter_s": round(dt, 1),
            "t_env": round(st["t_env"], 1), "t_fwd": round(st["t_fwd"], 1),
            "update_s": round(t_update, 1), "sps": round(len(slots) / dt, 1),
            "store_free": store.free, "hours": round((time.time() - t_train) / 3600, 3),
            **{k: round(float(np.mean(v)), 4) for k, v in agg.items()},
        }  # fmt: skip
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        out_of_time = (time.time() - t_train) / 3600 >= args.hours or (args.iters and it >= args.iters)
        if it % args.eval_every == 0 or out_of_time:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it}, args.out / "last.pt")
            evaluate(it)
        if out_of_time:
            break
    print(f"done: {it} iterations, {tot['games']} games, {tot['decisions']} decisions, "
          f"{(time.time() - t_start) / 3600:.2f} h", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
