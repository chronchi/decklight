"""Can the policy learn the hint bank's winning lines, and do they transfer to unseen hard starts? (point 3, 2026-10-09)

    uv run python scripts/bank_ft.py train --bank data/cache/hint_gated/bank.jsonl --init data/cache/base_r3/best.pt \
        --out runs/<name>/bankft [--held-frac 0.25 --epochs 12 --lr 1e-4 --device cuda]
    uv run python scripts/bank_ft.py play --out runs/<name>/bankft --opp data/cache/base_r3/best.pt \
        --opp2 <rl5 it200> [--games 8 --nproc 16 --proc 0]
    uv run python scripts/bank_ft.py report --out runs/<name>/bankft

rl6 (NOTEBOOK 2026-10-09 06:40) gave the bank a small dose (one 256-decision step per PPO update, ~4.6 passes at lr
6e-5; hint_acc 0.66 -> 0.71) and moved nothing. This asks the question directly, offline:
- train: the bank's starts (start_id, actor seat) are split into train / held-out (seeded). base_r3 is fine-tuned by
  cross-entropy on the train starts' banked decisions only; per epoch: CE and argmax agreement on the train and
  held-out decisions (epoch "init" = base_r3 itself). Writes <out>/ft/{best.pt (lowest held-out CE), final.pt,
  metrics.jsonl} (fp16 weights; metrics' first line carries base_r3's run args so BCPolicy loads it) and
  <out>/split.json.
- play: from every banked start, the actor seat played by base_r3 and by ft, vs `--opp` (and `--opp2`): one argmax game
  (deterministic: the bank's raw games always gave the same result per start, NOTEBOOK 2026-10-09 06:50) plus
  `--games` sampled games (actor temperature 1, opponent argmax). Shardable (`--nproc/--proc`, by start).
- report: win rates by split x policy x opponent x mode, paired by (start, opponent, game); writes report.json.
Reading: ft wins train starts but not held-out ones -> the lines are memorised, not learned as play (the bank is too
small or too specific to transfer); ft wins neither -> the targets are not learnable by this policy from these
positions (lines need the search's lookahead at each step); ft wins held-out starts clearly more than base_r3 ->
scale the bank. Engine + model; the train step needs a GPU box (no ML on the laptop).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EMB = ROOT / "data/cards/emb/qwen3-0.6b-d512.npz"


def fail(msg: str, code: int = 1):
    print(f"ALERT bank_ft: {msg}", flush=True)
    raise SystemExit(code)


def start_key(r: dict) -> tuple[int, int]:
    return int(r["start_id"]), int(r["actor_seat"])


def split_bank(rows: list[dict], held_frac: float, seed: int) -> tuple[list, list]:
    """Seeded split of the bank's starts (start_id, actor seat) into train / held-out keys."""
    keys = sorted({start_key(r) for r in rows})
    if len(keys) < 4:
        fail(f"{len(keys)} starts in the bank: too few to split")
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_held = max(1, round(held_frac * len(keys)))
    return sorted(keys[n_held:]), sorted(keys[:n_held])


def cmd_train(args) -> None:
    import torch

    from ygo_zero.engine.duel import Engine
    from ygo_zero.eval.bc_policy import _run_args
    from ygo_zero.model.policy_v3 import make_policy
    from ygo_zero.rl.hints import replay_rows
    from ygo_zero.rl.ppo import masked_log_softmax, model_keys, to_torch

    rows = [json.loads(x) for x in args.bank.read_text().splitlines() if x.strip()]
    if not rows:
        fail(f"empty bank {args.bank}")
    tr_keys, ho_keys = split_bank(rows, args.held_frac, args.seed)
    a = _run_args(args.init)
    if not a:
        fail(f"no run args next to {args.init} (metrics.jsonl): cannot rebuild the net")
    hist, known = int(a.get("hist", 0) or 0), bool(a.get("known", False))
    keys = tuple(model_keys(hist, known))
    engine = Engine()
    t0 = time.time()
    tr_set, ho_set = set(tr_keys), set(ho_keys)
    tr_obs, tr_act = replay_rows(engine, [r for r in rows if start_key(r) in tr_set], hist, known, keys)
    ho_obs, ho_act = replay_rows(engine, [r for r in rows if start_key(r) in ho_set], hist, known, keys)
    print(f"bank_ft: {len(rows)} games, starts train {len(tr_keys)} / held {len(ho_keys)}, decisions train "
          f"{len(tr_act)} / held {len(ho_act)} (replay {time.time() - t0:.0f} s)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps({"train": tr_keys, "held": ho_keys, "seed": args.seed,
                                                     "bank": str(args.bank), "init": str(args.init)}))
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    net = make_policy(args.emb, a, dropout=0.0, hist=hist, known=known)
    state = torch.load(args.init, map_location="cpu")
    state = state.get("model", state)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    net.to(dev)
    init_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    @torch.no_grad()
    def score(obs: dict, act: np.ndarray) -> dict:
        net.eval()
        ce = acc = 0.0
        for i in range(0, len(act), args.batch):
            o = to_torch({k: v[i : i + args.batch] for k, v in obs.items()}, dev)
            y = torch.from_numpy(act[i : i + args.batch]).long().to(dev)
            lg = net(o)
            lp = masked_log_softmax(lg)
            ce += float(-lp.gather(1, y[:, None]).sum())
            acc += float((lg.argmax(1) == y).sum())
        if not np.isfinite(ce):
            fail("non-finite CE in eval")
        return {"ce": round(ce / len(act), 4), "acc": round(acc / len(act), 4)}

    ft = args.out / "ft"
    ft.mkdir(exist_ok=True)
    mpath = ft / "metrics.jsonl"
    with mpath.open("w") as mf:
        mf.write(json.dumps({"args": a, "bank_ft": vars(args) | {"bank": str(args.bank), "init": str(args.init),
                                                                  "out": str(args.out), "emb": str(args.emb)}},
                            default=str) + "\n")
        rec = {"epoch": "init", "train": score(tr_obs, tr_act), "held": score(ho_obs, ho_act)}
        mf.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        best = rec["held"]["ce"]
        init_held = best
        rng = np.random.default_rng(args.seed)
        for ep in range(1, args.epochs + 1):
            net.train()
            perm = rng.permutation(len(tr_act))
            tot = 0.0
            for i in range(0, len(perm), args.batch):
                idx = perm[i : i + args.batch]
                o = to_torch({k: np.ascontiguousarray(v[idx]) for k, v in tr_obs.items()}, dev)
                y = torch.from_numpy(tr_act[idx]).long().to(dev)
                loss = -masked_log_softmax(net(o)).gather(1, y[:, None]).mean()
                if not bool(torch.isfinite(loss)):
                    fail(f"non-finite training loss at epoch {ep}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()) * len(idx)
            rec = {"epoch": ep, "loss": round(tot / len(perm), 4), "train": score(tr_obs, tr_act),
                   "held": score(ho_obs, ho_act)}
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            print(json.dumps(rec), flush=True)
            half = {k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}
            if rec["held"]["ce"] < best:
                best = rec["held"]["ce"]
                torch.save(half, ft / "best.pt")
        torch.save(half, ft / "final.pt")
    if all(torch.equal(v, init_sd[k]) for k, v in net.state_dict().items()):
        fail("fine-tuned weights equal the init", 3)
    if not (ft / "best.pt").is_file():
        # held-out CE never went below base_r3's: the lines do not transfer at all by CE. Play final.pt then, and say so.
        print(f"ALERT bank_ft: held-out CE never below the init's ({init_held}); best.pt = final.pt", flush=True)
        torch.save(half, ft / "best.pt")


def cmd_play(args) -> None:
    from hint_bank import raw_game

    from ygo_zero.engine.duel import Engine
    from ygo_zero.eval.bc_policy import BCPolicy
    from ygo_zero.eval.turn1 import config_from_dict

    split = json.loads((args.out / "split.json").read_text())
    rows = [json.loads(x) for x in Path(split["bank"]).read_text().splitlines() if x.strip()]
    first: dict[tuple[int, int], dict] = {}
    for r in rows:
        first.setdefault(start_key(r), r)
    keys = [("train", tuple(k)) for k in split["train"]] + [("held", tuple(k)) for k in split["held"]]
    mine = keys[args.proc :: args.nproc]
    init = Path(split["init"])
    ft = args.out / "ft" / "best.pt"
    if not ft.is_file():
        fail(f"missing {ft}: run train first")
    engine = Engine()

    def pol(ck: Path, temp: float, seed: int = 0):
        return BCPolicy(ck, args.emb, temperature=temp, seed=seed, threads=1, device="cpu")

    opps = {"opp": pol(args.opp, 0.0)}
    if args.opp2 is not None:
        opps["opp2"] = pol(args.opp2, 0.0)
    # ft = lowest held-out CE (transfer), ftfinal = last epoch (most fitted: the memorisation check on train starts)
    actors = {"base": init, "ft": ft, "ftfinal": args.out / "ft" / "final.pt"}
    out = args.out / f"play-{args.proc}.jsonl"
    t0 = time.time()
    n = 0
    with out.open("w") as f:
        for sp, (sid, seat) in mine:
            r = first[(sid, seat)]
            cfg = config_from_dict(r["config"])
            prefix = [int(c) for c in r.get("prefix") or []]
            for an, ack in actors.items():
                greedy = pol(ack, 0.0)
                sampled = pol(ack, 1.0, seed=1000 * args.proc + sid)
                for on, op in opps.items():
                    for g in range(args.games + 1):
                        p = greedy if g == 0 else sampled
                        rec = raw_game(engine, cfg, prefix, seat, p, op, game_idx=g, max_dec=args.max_decisions)
                        f.write(json.dumps({"split": sp, "start_id": sid, "seat": seat, "actor": an, "opp": on,
                                            "game": g, "mode": "greedy" if g == 0 else "sampled",
                                            "win": rec["winner"] == seat, "truncated": rec["truncated"],
                                            "decisions": len(rec["choices"]) - len(prefix)}) + "\n")
                        n += 1
                f.flush()
    print(f"bank_ft play proc {args.proc}/{args.nproc}: {len(mine)} starts, {n} games, {time.time() - t0:.0f} s",
          flush=True)


def cmd_report(args) -> None:
    recs = [json.loads(x) for f in sorted(args.out.glob("play-*.jsonl")) for x in f.read_text().splitlines()
            if x.strip()]
    if not recs:
        fail(f"no play-*.jsonl in {args.out}")
    from collections import defaultdict
    from math import comb

    cell = defaultdict(list)
    for r in recs:
        cell[(r["split"], r["opp"], r["mode"], r["actor"])].append(r)
    rep = {}
    print(f"{'split':>5} {'opp':>4} {'mode':>7} {'actor':>8} | {'base wr':>9} {'wr':>9} | paired base-only / "
          f"actor-only, p")
    for sp, on, mode, an in ((sp, on, mode, an) for sp in ("train", "held") for on in sorted({r["opp"] for r in recs})
                             for mode in ("greedy", "sampled") for an in ("ft", "ftfinal")):
        b = {(r["start_id"], r["seat"], r["game"]): r["win"] for r in cell[(sp, on, mode, "base")]}
        t = {(r["start_id"], r["seat"], r["game"]): r["win"] for r in cell[(sp, on, mode, an)]}
        ks = sorted(set(b) & set(t))
        if not ks:
            continue
        bo = sum(1 for k in ks if b[k] and not t[k])
        fo = sum(1 for k in ks if t[k] and not b[k])
        nd = bo + fo
        p = min(1.0, 2 * sum(comb(nd, i) for i in range(min(bo, fo) + 1)) / 2**nd) if nd else 1.0
        bw, tw = np.mean([b[k] for k in ks]), np.mean([t[k] for k in ks])
        starts_won = len({k[:2] for k in ks if t[k]})
        rep[f"{sp}/{on}/{mode}/{an}"] = {"n": len(ks), "base_wr": round(float(bw), 3), "wr": round(float(tw), 3),
                                          "base_only": bo, "actor_only": fo, "p": round(p, 4),
                                          "starts_won": starts_won, "starts": len({k[:2] for k in ks})}
        print(f"{sp:>5} {on:>4} {mode:>7} {an:>8} | {bw:9.3f} {tw:9.3f} | {bo:>4} / {fo:<4} p={p:.3f}  "
              f"(wins >=1 game on {starts_won}/{len({k[:2] for k in ks})} starts; n {len(ks)})")
    trunc = sum(r["truncated"] for r in recs)
    if trunc:
        print(f"WARNING {trunc}/{len(recs)} games truncated")
    (args.out / "report.json").write_text(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--bank", type=Path, required=True)
    t.add_argument("--init", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--emb", type=Path, default=EMB)
    t.add_argument("--held-frac", type=float, default=0.25)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--device", default="cpu")
    p = sub.add_parser("play")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--opp", type=Path, required=True)
    p.add_argument("--opp2", type=Path, default=None)
    p.add_argument("--emb", type=Path, default=EMB)
    p.add_argument("--games", type=int, default=8, help="sampled games per (start, actor, opponent)")
    p.add_argument("--max-decisions", type=int, default=1500)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--proc", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"train": cmd_train, "play": cmd_play, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
