"""Python side of the A3a parity dump (the oracle for `rust/decklight-search`).

    uv run python scripts/app/search_parts_dump.py --pairs A:B C:D --games 40 --seed 0 --choice-seed 0 \
        --out data/cache/app/search_parts_py.jsonl.gz

Plays the same seeded duels as `rust/decklight-search/src/bin/search_parts_dump.rs` (`auto_single=True`, the
`play_dump.py` splitmix64 choice stream), emits per decision `_masked`, the loop-guard-masked logits and state key,
argmax, softmax confidence (bit pattern), semantic keys, merged candidates and synthetic honest-rule decisions. Every
25th decision from turn 2 on builds a `Determinizer`, samples four worlds and records its proposals, counters and RNG
state. All synthetic inputs are SplitMix64 streams seeded by (choice-seed, game, decision); floats are emitted as bit
patterns. Compare with `scripts/app/search_parts_compare.py`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys

import numpy as np

from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine
from decklight.eval.loopguard import STATE_KEYS, LoopGuard
from decklight.search.determinize import Determinizer, SeenTracker, key_rank_of_choice
from decklight.search.honest import (
    HonestRule,
    decide,
    decide_pair,
    decisive_label,
    merge_candidates,
    paired_gap,
    semantic_key,
)
from decklight.search.play import argmax_select, softmax_conf
from decklight.search.turn1 import _masked

M64 = (1 << 64) - 1
INV53 = 1.0 / 9007199254740992.0


class SplitMix:
    def __init__(self, s: int) -> None:
        self.s = s & M64

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def unit53(self) -> float:
        return (self.next() >> 11) * INV53

    def unit11(self) -> float:
        return self.unit53() * 2.0 - 1.0


def fbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def synth_logits(cseed: int, g: int, k: int, mask: np.ndarray) -> np.ndarray:
    sm = SplitMix(cseed ^ g ^ ((k * 0xD6E8FEB86659FD93) & M64))
    out = np.empty(64, np.float32)
    for i in range(64):
        out[i] = np.float32(sm.unit53() * 16.0 - 8.0)
    if k % 7 == 0:
        out[mask.astype(bool)] = np.float32(0.0)
    return out


def synth_samples(cseed: int, g: int, k: int) -> list[list[float]]:
    sm = SplitMix(cseed ^ ((g << 1) & M64) ^ ((k << 17) & M64) ^ 0x51ED270B)
    nc = 2 + (sm.next() % 2)
    out = []
    for _ in range(nc):
        n = sm.next() % 6
        out.append([sm.unit11() for _ in range(n)])
    return out


def state_digest(obs: dict) -> str:
    h = hashlib.sha256()
    for k in STATE_KEYS:
        if k in obs:
            h.update(np.ascontiguousarray(obs[k]).tobytes())
    return h.hexdigest()


def hash_mains(m0, m1) -> str:
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(m0)))
    for x in m0:
        h.update(struct.pack("<I", int(x) & 0xFFFFFFFF))
    h.update(struct.pack("<Q", len(m1)))
    for x in m1:
        h.update(struct.pack("<I", int(x) & 0xFFFFFFFF))
    return h.hexdigest()


def hash_state(rng) -> str:
    st = rng.getstate()
    h = hashlib.sha256()
    for w in st[1]:
        h.update(struct.pack("<I", int(w) & 0xFFFFFFFF))
    return h.hexdigest()


def key_ints(k) -> list[int]:
    if len(k) == 3 and int(k[0]) == 19:
        z = k[2]
        zi = 2 if z == "mz" else 3 if z == "sz" else int(z)
        return [1, int(k[0]), int(k[1]), zi]
    return [0] + [int(x) for x in k]


def decision_json(d) -> dict:
    return {
        "chosen": int(d.chosen_j),
        "challenger": None if d.challenger_j is None else int(d.challenger_j),
        "gap_bits": fbits(d.gap),
        "se_bits": fbits(d.se),
        "n": int(d.n),
        "status": d.status,
    }


class LoggingDeterminizer(Determinizer):
    def __init__(self, *a, **kw) -> None:
        self.proposal_log: list[str | None] = []
        super().__init__(*a, **kw)

    def _propose_mains(self):
        out = super()._propose_mains()
        self.proposal_log.append(None if out is None else hash_mains(out[0], out[1]))
        return out


def resolve_deck(s: str):
    from pathlib import Path

    p = Path(s)
    if p.exists():
        return p
    pool = Path(__file__).resolve().parents[2] / "decks" / "pool" / f"{s}.ydk"
    if pool.exists():
        return pool
    pool2 = Path(__file__).resolve().parents[2] / "decks" / "pool" / s
    if pool2.exists():
        return pool2
    raise SystemExit(f"deck not found: {s}")


def parse_pairs(items):
    out = []
    for it in items:
        a, sep, b = it.partition(":")
        if not sep:
            raise SystemExit(f"--pairs entry {it!r} needs A:B")
        out.append((resolve_deck(a), resolve_deck(b)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck0", default=None)
    ap.add_argument("--deck1", default=None)
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choice-seed", type=int, default=0)
    ap.add_argument("--game-offset", type=int, default=0)
    ap.add_argument("--out", default="data/cache/app/search_parts_py.jsonl.gz")
    ap.add_argument("--det-every", type=int, default=25)
    ap.add_argument("--det-tries", type=int, default=32)
    ap.add_argument("--det-samples", type=int, default=4)
    ap.add_argument("--det-min-turn", type=int, default=2)
    ap.add_argument("--search-seed", type=int, default=7)
    ap.add_argument("--max-decisions", type=int, default=3000)
    a = ap.parse_args(argv)

    from decklight.decks import load_ydk

    if a.deck0 and a.deck1:
        pairs = [(resolve_deck(a.deck0), resolve_deck(a.deck1))]
    elif a.pairs:
        pairs = parse_pairs(a.pairs)
    else:
        raise SystemExit("--deck0/--deck1 or --pairs is required")
    decks = [(load_ydk(x), load_ydk(y)) for x, y in pairs]
    engine = Engine()

    rule = HonestRule()
    n_decisions = n_ties = n_det = n_det_proposals = n_det_accepted = 0
    with gzip.open(a.out, "wt", compresslevel=6) as f:
        for gi in range(a.games):
            g = a.game_offset + gi
            d0, d1 = decks[g % len(decks)]
            s = a.seed + g
            cfg = DuelConfig(d0, d1, seed=s)
            env = CppEnv(engine, cfg, True)
            obs = env.reset()
            choice_rng = SplitMix(a.choice_seed ^ g)
            trackers = {0: SeenTracker(0), 1: SeenTracker(1)}
            for t in trackers.values():
                t.update(env)
            guard = LoopGuard()
            prefix: list[int] = []
            prefix_keys: list = []
            k = 0
            while obs is not None and k < a.max_decisions:
                player = int(env.player)
                turn = int(obs["global"][2])
                mask = obs["action_mask"]
                logits = synth_logits(a.choice_seed, g, k, mask)
                n_decisions += 1
                if k % 7 == 0:
                    n_ties += 1
                z = _masked(logits, obs)
                z_g, _key = guard.mask(0, obs, z)
                am = int(argmax_select(z_g))
                conf = softmax_conf(z_g)
                legal = np.flatnonzero(mask.astype(bool)).tolist()
                keys = [key_ints(semantic_key(obs, j)) for j in legal]
                merged = [int(x) for x in merge_candidates(legal, obs)]
                samples = synth_samples(a.choice_seed, g, k)
                m, se, n = paired_gap(samples[0], samples[1])
                dec = decide(samples, 0, rule)
                dec_pair = decide_pair(samples, 0, 1, rule)
                label = decisive_label(samples, rule)
                f.write(
                    json.dumps(
                        {
                            "kind": "parts",
                            "g": g,
                            "k": k,
                            "player": player,
                            "turn": turn,
                            "msg": int(obs["global"][21]),
                            "z": [fbits(x) for x in z],
                            "zg": [fbits(x) for x in z_g],
                            "key": state_digest(obs),
                            "argmax": am,
                            "conf_bits": fbits(conf),
                            "sem_keys": keys,
                            "merged": merged,
                            "pair": {"m_bits": fbits(m), "se_bits": fbits(se), "n": int(n)},
                            "decide": decision_json(dec),
                            "decide_pair": decision_json(dec_pair),
                            "label": decision_json(label),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                if turn >= a.det_min_turn and a.det_every > 0 and k % a.det_every == 0:
                    rng = __import__("random").Random(f"{a.search_seed}:{s}:{k}:det")
                    det = LoggingDeterminizer(
                        engine,
                        cfg,
                        True,
                        player,
                        prefix,
                        trackers[player].seen,
                        obs,
                        rng,
                        tries=a.det_tries,
                        prefix_keys=prefix_keys,
                    )
                    hidden0 = [int(x) for x in det._hidden_slots(0)]
                    hidden1 = [int(x) for x in det._hidden_slots(1)]
                    results = det.sample_many(a.det_samples)
                    n_det += 1
                    n_det_proposals += det.proposals
                    n_det_accepted += det.accepted
                    res_json = []
                    for got in results:
                        if got is None:
                            res_json.append(None)
                        else:
                            world, resolved = got
                            res_json.append(
                                {
                                    "mains": hash_mains(world.deck0.main, world.deck1.main),
                                    "resolved": [int(x) for x in resolved],
                                }
                            )
                    f.write(
                        json.dumps(
                            {
                                "kind": "det",
                                "g": g,
                                "k": k,
                                "player": player,
                                "turn": turn,
                                "base": [int(det.base[0]), int(det.base[1])],
                                "hidden": [hidden0, hidden1],
                                "proposals": det.proposal_log,
                                "results": res_json,
                                "counts": {
                                    "proposals": int(det.proposals),
                                    "accepted": int(det.accepted),
                                    "fallbacks": int(det.fallbacks),
                                },
                                "reasons": {
                                    kk: int(det.reasons.get(kk, 0))
                                    for kk in ("no_key", "replay_error", "obs_mismatch")
                                },
                                "rng_after": hash_state(det.rng),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

                n_opt = int(mask.sum())
                if n_opt <= 0:
                    raise SystemExit(f"ALERT search_parts_dump: game {g} decision {k}: no legal option")
                choice = int(choice_rng.next() % n_opt)
                guard.note(0, _key, choice)
                prefix_keys.append(key_rank_of_choice(obs, choice))
                prefix.append(choice)
                obs = env.step(choice)
                for t in trackers.values():
                    t.update(env)
                k += 1
            f.write(
                json.dumps(
                    {
                        "kind": "end",
                        "g": g,
                        "decisions": k,
                        "winner": int(env.winner) if env.winner is not None else -1,
                        "choices": [int(x) for x in env.choices],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            if env.errors:
                raise SystemExit(f"ALERT search_parts_dump: {len(env.errors)} Lua/script errors in game {g}")
    print(
        f"search_parts_dump: games={a.games} decisions={n_decisions} ties={n_ties} det_runs={n_det} "
        f"proposals={n_det_proposals} accepted={n_det_accepted} -> {a.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
