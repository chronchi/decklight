"""Free check before M2 (docs/plans/2026-10-20-m2-review.md): is the in-loop RL critic worse than value_p2?

Replays stored games (search_match records) and scores, at the first decision of each (actor, turn), the same states
with value_p2 (ValueNet) and with in-loop critics: the value head + trunk of train_rl checkpoints (P+value_p =
D0's critic at iteration 0; D0 last = after 160 iterations; iter-*.pt files hold no value head). Reports explained variance against the game
result (z = +-1, actor's view), correlation and sign accuracy per turn bucket, all on identical states.

    taskset -c 0-3 uv run python scripts/audit/critic_compare.py --games runs/value-p2-20261015-2044/hppo-p2 \
        --max-games 150 --rl data/cache/P/best.pt+data/cache/value_p/best.pt runs/ppo-d0-20261015-1438/rl/last.pt
CPU, a few hundred games only (laptop: tiny inference). Exits 2 on 0 states, NaN, or two scorers giving identical
values everywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from score_decisions import iter_records
from train_bc import _run_args

from decklight.decks import load_ydk
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine
from decklight.eval.value import ValueNet
from decklight.model.bc_data import OBS_KEYS
from decklight.model.policy_v3 import make_policy

BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5+", 5, 10**6))


def fail(msg: str) -> None:
    print(f"ALERT critic_compare: {msg}", flush=True)
    raise SystemExit(2)


def rl_scorer(spec: str, emb: Path):
    """`spec` is a train_rl checkpoint with a value head, or `TRUNK+HEAD`: the value head of HEAD on TRUNK's
    network, as train_rl --value-init builds the critic at iteration 0."""
    trunk, _, head = spec.partition("+")
    ckpt = Path(trunk)
    a = _run_args(ckpt)
    model = make_policy(emb, a, value=True)
    state = torch.load(ckpt, map_location="cpu")
    state = dict(state.get("model", state))
    if head:
        hs = torch.load(head, map_location="cpu")
        hs = hs.get("model", hs)
        state.update({k: v for k, v in hs.items() if k.startswith("value_head.")})
    if not any(k.startswith("value_head.") for k in state):
        fail(f"{spec}: no value_head weights")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.eval()

    @torch.no_grad()
    def score(buf: dict) -> np.ndarray:
        out = []
        for i in range(0, len(buf["global"]), 256):
            t = {k: torch.from_numpy(np.ascontiguousarray(buf[k][i : i + 256])) for k in OBS_KEYS}
            _, v = model.forward_value(t)
            out.append(v.float().numpy().reshape(-1))
        return np.clip(np.concatenate(out), -1, 1)

    return score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, required=True)
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--value", type=Path, default=ROOT / "data/cache/value_p2/best.pt")
    ap.add_argument("--rl", nargs="+", required=True, help="ckpt with a value head, or TRUNK+HEAD")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    a = ap.parse_args()
    torch.set_num_threads(4)

    engine = Engine()
    obs_l, z_l, turn_l = [], [], []
    for rec in iter_records([a.games], a.max_games):
        if rec.get("truncated") or rec.get("winner") not in (0, 1):
            continue
        d0, d1 = rec["decks"]
        cfg = DuelConfig(
            load_ydk(ROOT / "decks/gauntlet" / f"{d0}.ydk"),
            load_ydk(ROOT / "decks/gauntlet" / f"{d1}.ydk"),
            seed=int(rec["seed"]),
        )
        env = CppEnv(engine, cfg)
        obs = env.reset()
        seen = set()
        for ch in rec["choices"]:
            if obs is None:
                break
            actor, turn = int(env.player), int(obs["global"][2])
            if (actor, turn) not in seen:
                seen.add((actor, turn))
                obs_l.append({k: obs[k] for k in OBS_KEYS})
                z_l.append(1.0 if actor == int(rec["winner"]) else -1.0)
                turn_l.append(turn)
            obs = env.step(int(ch))
    if not obs_l:
        fail("0 states")
    buf = {k: np.stack([o[k] for o in obs_l]) for k in OBS_KEYS}
    z, turn = np.asarray(z_l), np.asarray(turn_l)
    print(f"{len(z)} turn-start states from {a.max_games} games", flush=True)

    scores = {"value_p2": ValueNet(a.value, a.emb, threads=4).values(buf)}
    for ck in a.rl:
        scores[ck if len(ck) < 40 else "..." + ck[-37:]] = rl_scorer(ck, a.emb)(buf)
    names = list(scores)
    for n, v in scores.items():
        if np.isnan(v).any():
            fail(f"NaN from {n}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.array_equal(scores[names[i]], scores[names[j]]):
                fail(f"{names[i]} and {names[j]} give identical values")
    for b, lo, hi in BUCKETS:
        m = (turn >= lo) & (turn <= hi)
        zz = z[m]
        line = f"turn {b:4s} n {m.sum():5d}"
        for n in names:
            v = scores[n][m]
            e = 1 - np.var(zz - v) / np.var(zz)
            line += (
                f" | {n}: EV {e:+.3f} r {np.corrcoef(zz, v)[0, 1]:.3f} sign {np.mean(np.sign(v) == zz):.3f}"
            )
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
