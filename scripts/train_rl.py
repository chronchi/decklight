"""RL v0: PPO self-play from a BC checkpoint, KL-anchored to the frozen BC policy (PLAN Phase 5, first feel).

    uv run python scripts/train_rl.py --init runs/<run>/bc/best.pt --out runs/<name>/rl --hours 1.5 --device cuda

Rollouts: `--envs` C++ envs over the pool decks (holdout excluded) + the gauntlet decks. Each episode is either
self-play (both seats sampled from the learner, both trained) or, with probability `--p-bc`, learner vs the frozen BC
policy (sampled, temperature 1; only the learner seat is trained). Reward +1/-1 at the end, 0 draw, `--trunc-reward`
for both seats when a game hits `--max-decisions`. Value head new (trained alone for `--value-warmup` iterations).
Loss: clipped PPO + value MSE - entropy + `--kl-coef` * KL(pi || pi_BC) over legal options.

Progress is measured only on the fixed gauntlet every `--eval-every` iterations (and at iteration 0 / the end):
greedy learner vs greedy BC (both seats, 4 train + 4 holdout decks) and vs ygo-agent 0546 (its 4 decks), via the same
play_match/BCPolicy path as scripts/playtest.py. Never read the training win rates as progress.

Output (`--out`): metrics.jsonl (first record "args" carries the BC architecture so BCPolicy/playtest load the
checkpoints), iter-NNNN.pt (fp16 policy weights at each eval, no value head), last.pt (full, with optimizer).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ygo_zero.decks import load_ydk
from ygo_zero.engine.cpp_env import VecEnv
from ygo_zero.engine.duel import Engine
from ygo_zero.eval.bc_policy import BCPolicy, _run_args
from ygo_zero.eval.match import play_match, wilson
from ygo_zero.model.policy import PolicyNet
from ygo_zero.rl.ppo import SlotStore, gae, gather, masked_log_softmax, ppo_loss, to_torch

ROOT = Path(__file__).resolve().parents[1]


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
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-decisions", type=int, default=1500, help="per game, both seats")
    ap.add_argument("--trunc-reward", type=float, default=-0.5)
    ap.add_argument("--p-bc", type=float, default=0.3, help="share of episodes vs the frozen BC policy")
    ap.add_argument("--rollout", type=int, default=32768, help="finished samples per PPO iteration")
    ap.add_argument("--store", type=int, default=400_000, help="sample slots (open + finished trajectories)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--value-warmup", type=int, default=2, help="first iterations train the value head only")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=2, help="vs BC: seeds per ordered deck pair")
    ap.add_argument("--ga-seeds", type=int, default=3, help="vs ygo-agent: seeds per ordered deck pair")
    ap.add_argument("--no-ygoagent", action="store_true")
    ap.add_argument("--hours", type=float, default=1.0, help="stop (then final eval) after this much training")
    ap.add_argument("--iters", type=int, default=0, help="stop after this many iterations (0 = by --hours)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    # decks: training pool = pool - holdout + gauntlet; eval = fixed 4 train + 4 holdout sample
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    pool = sorted((ROOT / "decks/pool").glob("*.ydk"))
    gfiles = sorted((ROOT / "tests/fixtures/decks").glob("*.ydk"))
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
            gp = [(i, j) for i in range(len(g_decks)) for j in range(len(g_decks)) if i != j]
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

    # rollout state
    N = args.envs
    vec = VecEnv(engine, train_decks, N, args.threads, seed=args.seed)
    vec.set_max_decisions(args.max_decisions)
    buf = vec.reset()
    store = SlotStore(args.store, gather(buf, np.arange(1), hist, known))
    traj = [[[], []] for _ in range(N)]  # traj[env][seat] = slots
    vs_bc = rng.random(N) < args.p_bc
    lseat = rng.integers(0, 2, N)
    finished: list[tuple[list[int], float]] = []
    n_fin = 0
    it, t_train = 0, time.time()
    tot = {"games": 0, "trunc": 0, "decisions": 0}
    while True:
        it += 1
        t_it = time.time()
        st = {"games": 0, "trunc": 0, "bc_games": 0, "bc_wins": 0.0, "decisions": 0, "lengths": []}
        while n_fin < args.rollout:
            actor = buf["player"].copy()
            learner = ~vs_bc | (actor == lseat)
            lrows, brows = np.flatnonzero(learner), np.flatnonzero(~learner)
            actions = np.zeros(N, np.int32)
            with torch.no_grad():
                if len(lrows):
                    ob = gather(buf, lrows, hist, known)
                    logits, v = model.forward_value(to_torch(ob, dev))
                    act, lp = sample(logits)
                    actions[lrows] = act
                    slots = store.put(ob, act, lp, v.float().cpu().numpy())
                    for i, s in zip(lrows, slots):
                        traj[i][actor[i]].append(int(s))
                if len(brows):
                    act, _ = sample(ref(to_torch(gather(buf, brows, hist, known), dev)))
                    actions[brows] = act
            vec.step(actions)
            st["decisions"] += N
            for i in np.flatnonzero(buf["done"]):
                w = int(buf["winner"][i])
                st["games"] += 1
                st["trunc"] += w == VecEnv.TRUNCATED
                st["lengths"].append(len(traj[i][0]) + len(traj[i][1]))
                for seat in (0, 1):
                    sl = traj[i][seat]
                    rew = args.trunc_reward if w == VecEnv.TRUNCATED else (0.0 if w == 2 else (1.0 if w == seat else -1.0))
                    if sl:
                        finished.append((sl, rew))
                        n_fin += len(sl)
                    traj[i][seat] = []
                if vs_bc[i]:
                    st["bc_games"] += 1
                    st["bc_wins"] += 0.5 if w not in (0, 1) else float(w == lseat[i])
                vs_bc[i] = rng.random() < args.p_bc
                lseat[i] = rng.integers(0, 2)
            vec.pop_finished()
        t_roll = time.time() - t_it

        # GAE per finished seat trajectory, then PPO epochs
        slots = np.concatenate([np.array(s, np.int64) for s, _ in finished])
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
        model.train()
        agg: dict[str, list] = {}
        for _ in range(args.epochs):
            perm = rng.permutation(len(slots))
            for k in range(0, len(perm), args.minibatch):
                mb = perm[k : k + args.minibatch]
                sl = slots[mb]
                obs = to_torch({key: v[sl] for key, v in store.obs.items()}, dev)
                act = torch.from_numpy(store.action[sl]).to(dev)
                olp = torch.from_numpy(store.logp[sl]).to(dev)
                a_t = torch.from_numpy(adv_n[mb]).to(dev)
                r_t = torch.from_numpy(ret[mb]).to(dev)
                if warm:
                    with torch.no_grad():  # value head only: no graph through the trunk
                        _, enc = model._run(obs)
                    v = model.value_head(enc[:, 0]).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(v.float(), r_t)
                    vopt.zero_grad(set_to_none=True)
                    loss.backward()
                    vopt.step()
                    s = {"vf": loss.item()}
                else:
                    loss, s = ppo_loss(model, ref, obs, act, olp, a_t, r_t, args.clip, args.vf_coef, args.kl_coef,
                                       args.ent_coef)  # fmt: skip
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                for key, val in s.items():
                    agg.setdefault(key, []).append(val)
        model.eval()
        for s, _ in finished:
            store.release(s)
        finished, n_fin = [], 0

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
