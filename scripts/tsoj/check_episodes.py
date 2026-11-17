#!/usr/bin/env python3
"""Check remote-agent episode files against the arbiter records and the learner's store (tsoj PPO step 2 smoke).

    uv run python scripts/tsoj/check_episodes.py --episodes DIR --records R.jsonl --weights W.pt [--init ckpt]

Checks (exit 3 on any failure, one ALERT line each):
- one episode per finished game, and each episode's reward = the arbiter's result for P's seat (seat/deck matched);
- the stored obs keys/shapes/dtypes = a VecEnv buffer's (the PPO store's layout);
- the stored log-prob of every action = the published weights' log-prob recomputed here (same weights version);
- ExtFeed.take puts every sample in a SlotStore and returns (slots, reward) per episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=ROOT / "data/cache/forge_r1/best.pt")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    a = ap.parse_args()

    import torch

    from decklight.decks.ydk import load_ydk
    from decklight.engine.cpp_env import VecEnv
    from decklight.engine.duel import Engine
    from decklight.eval.bc_policy import _run_args
    from decklight.model.policy_v3 import make_policy
    from decklight.rl.ppo import SlotStore, gather, masked_log_softmax, to_torch
    from decklight.rl.remote import ExtFeed, read_episode

    bad: list[str] = []
    eps = sorted(a.episodes.glob("ep-*.npz"))
    recs = [json.loads(x) for x in a.records.read_text().splitlines() if x.strip()]
    print(f"{len(eps)} episodes, {len(recs)} records", flush=True)
    if not eps or len(eps) != len(recs):
        bad.append(f"episodes {len(eps)} != games {len(recs)}")
    # rewards: match each episode to the record with P (dl) on its seat holding its deck, by outcome multiset
    want: dict[tuple, list[float]] = {}
    for r in recs:
        ps = r["agents"].index("dl")
        w = (r.get("result") or {}).get("winner")
        want.setdefault((r["decks"][ps]["name"], ps), []).append(0.0 if w not in (0, 1) else (1.0 if w == ps else -1.0))
    got: dict[tuple, list[float]] = {}
    for f in eps:
        e = read_episode(f)
        got.setdefault((e["meta"]["deck"], e["meta"]["seat"]), []).append(e["reward"])
    for k in sorted(set(want) | set(got), key=str):
        if sorted(want.get(k, [])) != sorted(got.get(k, [])):
            bad.append(f"reward mismatch {k}: arbiter {sorted(want.get(k, []))} episodes {sorted(got.get(k, []))}")
    print(f"rewards by (deck, seat): {got}", flush=True)

    # store layout from a real VecEnv buffer
    eng = Engine()
    deck = load_ydk(ROOT / "scripts/tsoj/decks/skydeck-tcg2026-10.ydk")
    vec = VecEnv(eng, [deck, deck], 2, 1, seed=0)
    hist, known = 0, False
    store = SlotStore(20_000, gather(vec.buf, np.arange(1), hist, known))
    e0 = read_episode(eps[0])
    for k, arr in store.obs.items():
        o = e0["obs"].get(k)
        if o is None or o.shape[1:] != arr.shape[1:] or o.dtype != arr.dtype:
            bad.append(f"obs {k}: episode {None if o is None else (o.dtype, o.shape[1:])} store {(arr.dtype, arr.shape[1:])}")
    extra = set(e0["obs"]) - set(store.obs)
    print(f"store keys {sorted(store.obs)}; extra episode keys {sorted(extra)}", flush=True)

    # log-probs recomputed with the published weights
    net = make_policy(dict(np.load(a.emb)), _run_args(a.init), dropout=0.0, value=True).eval()
    blob = torch.load(a.weights, map_location="cpu")
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in blob["model"].items()})
    worst = 0.0
    for f in eps:
        e = read_episode(f)
        if set(e["version"].tolist()) != {int(blob["version"])}:
            bad.append(f"{f.name}: versions {set(e['version'].tolist())} != weights {blob['version']}")
        with torch.no_grad():
            lg, v = net.forward_value(to_torch({k: e["obs"][k] for k in store.obs}, torch.device("cpu")))
            lp = masked_log_softmax(lg).gather(1, torch.from_numpy(e["action"])[:, None]).squeeze(1).numpy()
        worst = max(worst, float(np.abs(lp - e["logp"]).max()), float(np.abs(v.numpy() - e["value"]).max()))
    print(f"max |logp/value - recomputed| = {worst:.2e}", flush=True)
    if not worst < 1e-3:
        bad.append(f"logp/value differ from the published weights by {worst:.3g}")

    # ExtFeed into the store (on a copy of the episode dir)
    tmp = Path(tempfile.mkdtemp())
    for f in eps:
        shutil.copy(f, tmp / f.name)
    ref = make_policy(dict(np.load(a.emb)), _run_args(a.init), dropout=0.0).eval()
    ref.load_state_dict({k: v.float() for k, v in blob["model"].items() if not k.startswith("value_head.")})
    fin, st = ExtFeed(tmp).take(store, 10**9, 0, ref, torch.device("cpu"))
    n = sum(len(read_episode(f)["action"]) for f in eps)
    print(f"ExtFeed: {st}", flush=True)
    if len(fin) != len(eps) or st["ext_samples"] != n:
        bad.append(f"ExtFeed took {len(fin)} episodes / {st['ext_samples']} samples, want {len(eps)} / {n}")
    shutil.rmtree(tmp, ignore_errors=True)
    for b in bad:
        print(f"ALERT check_episodes: {b}", flush=True)
    print("OK" if not bad else f"{len(bad)} problems", flush=True)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
