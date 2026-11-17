"""Python oracle for the A3b-1 product games (the Rust bin is `product_games`).

    OMP_NUM_THREADS=1 YGO_SEARCH_CLONE=0 uv run python scripts/app/product_games_py.py \\
        --pairs blue-eyes:dark-magician branded:dinosaur --games 4 --seed 0 --choice-seed 0 \\
        --out data/cache/app/product_py.jsonl.gz [--io-hash-out data/cache/app/io_py.jsonl]

Plays the same games as the Rust bin with `search_game_gen` itself (never reimplements the loop): S = the product
(honest k4 s4 h1, det-tries 32, det-fallback skip, conf-skip 0.9, ho), R = a splitmix64 human fed as one-hot logits
and unguarded (`guard_mode="per_seat"`, `unguarded_seat=R`). The forward routes rollout obs and S's outer obs to the
ONNX policy (one ORT call per request) and R's outer obs to the one-hot, exactly as `ProductDriver._forward`.

Both sides must use the same onnxruntime library file (`ORT_DYLIB_PATH`) and intra_op threads 1.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

# The oracle is the default C++ core with prefix replays (A3a's path), not the Rust core.
os.environ.pop("YGO_CORE", None)
os.environ["YGO_SEARCH_CLONE"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from decklight.engine.duel import DuelConfig, Engine
from decklight.model.onnx_export import (
    codes_to_rows,
    with_history_source,
    pad_logits,
    raw_table,
    sorted_codes,
    trim_obs_np,
)
from decklight.search.honest import HonestRule
from decklight.search.play import argmax_select, search_game_gen
from decklight.search.product import build_search, product_search_kwargs, seat_of, tag_game
from decklight.search.turn1 import _batch

M64 = (1 << 64) - 1
DTYPE_TAG = {np.dtype(np.int32): 1, np.dtype(np.bool_): 2, np.dtype(np.float32): 3}


class SplitMix:
    def __init__(self, s: int) -> None:
        self.s = s & M64

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def resolve_deck(s: str) -> Path:
    p = Path(s)
    if p.exists():
        return p
    pool = ROOT / "decks" / "pool" / f"{s}.ydk"
    if pool.exists():
        return pool
    pool2 = ROOT / "decks" / "pool" / s
    if pool2.exists():
        return pool2
    raise SystemExit(f"deck not found: {s}")


def parse_pairs(items: list[str]) -> list[tuple[str, Path, Path]]:
    out = []
    for it in items:
        x, sep, y = it.partition(":")
        if not sep:
            raise SystemExit(f"--pairs entry {it!r} needs deck0:deck1")
        out.append((it, resolve_deck(x), resolve_deck(y)))
    return out


def _hash_inputs(inputs: list[str], feed: dict[str, np.ndarray], text_sha: bytes) -> str:
    h = hashlib.sha256()
    for name in inputs:
        a = np.ascontiguousarray(feed[name])
        tag = DTYPE_TAG[a.dtype]
        h.update(name.encode())
        h.update(b"\x00")
        h.update(bytes([tag]))
        h.update(struct.pack("<Q", a.ndim))
        for d in a.shape:
            h.update(struct.pack("<Q", int(d)))
        # The text table is large and constant; bind its shape and precomputed digest (as the Rust side).
        h.update(text_sha if name == "text" else a.tobytes())
    return h.hexdigest()


class OrtNets:
    """The product nets through onnxruntime with run_ort_full's feeding code (imported, not copied)."""

    def __init__(self, onnx_dir: Path, emb_path: Path, io_path: Path | None) -> None:
        import onnxruntime as ort

        self.ort = ort
        self.dir = onnx_dir
        man = json.loads((onnx_dir / "manifest.json").read_text())
        self.io_maps = man["graphs"]
        # Which obs key feeds each graph's history inputs (Y: the policy reads obs_history; absent = history).
        self.hist_src = {"policy": man.get("history_source", "history"), "value": man.get("value_history_source", "history")}
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = {
            k: ort.InferenceSession(
                str(onnx_dir / f"{k}.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
            )
            for k in ("card_proj", "card_proj_value", "policy", "value")
        }
        emb = dict(np.load(emb_path))
        self.codes = sorted_codes(emb)
        raw = raw_table(emb).astype(np.float32)
        unknown = np.load(onnx_dir / "unknown.npy").reshape(1, -1).astype(np.float32)
        unknown_v = np.load(onnx_dir / "unknown_value.npy").reshape(1, -1).astype(np.float32)
        self.text = self.sess["card_proj"].run(["text"], {"raw": np.concatenate([unknown, raw])})[0]
        self.text_v = self.sess["card_proj_value"].run(["text"], {"raw": np.concatenate([unknown_v, raw])})[0]
        self.text_sha = hashlib.sha256(np.ascontiguousarray(self.text).tobytes()).digest()
        self.text_v_sha = hashlib.sha256(np.ascontiguousarray(self.text_v).tobytes()).digest()
        self.io_path = io_path
        self.io_idx = 0
        self.game = 0
        self.io_f = None
        if io_path is not None:
            io_path.parent.mkdir(parents=True, exist_ok=True)
            self.io_f = open(io_path, "w")  # noqa: SIM115 (streamed per call)

    def _trim(self, batch: dict[str, np.ndarray], graph: str):
        b = with_history_source(codes_to_rows(batch, self.codes), self.hist_src[graph])
        b, n_opt = trim_obs_np(b)
        return b, n_opt

    def _run(self, graph: str, obs_list: list[dict], is_policy: bool) -> np.ndarray:
        batch = _batch(obs_list)
        sl, n_opt = self._trim(batch, graph)
        inputs = self.io_maps[graph]["inputs"]
        text = self.text if is_policy else self.text_v
        feed = {name: np.ascontiguousarray(sl[name]) for name in inputs if name != "text"}
        feed["text"] = text
        out = self.sess[graph].run([self.io_maps[graph]["outputs"][0]], feed)[0]
        if self.io_f is not None:
            out32 = np.asarray(out, dtype=np.float32)
            rec = {
                "kind": "io",
                "g": self.game,
                "idx": self.io_idx,
                "call": graph,
                "batch": len(obs_list),
                "input": _hash_inputs(inputs, feed, self.text_sha if is_policy else self.text_v_sha),
                "output": hashlib.sha256(out32.tobytes()).hexdigest(),
                "shape": list(out32.shape),
            }
            self.io_idx += 1
            self.io_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self.io_f.flush()
        if is_policy:
            return pad_logits(out, n_opt)
        return out

    def policy_logits(self, obs_list: list[dict]) -> np.ndarray:
        return self._run("policy", obs_list, True)

    def values_batch(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        n = next(iter(batch.values())).shape[0]
        out = np.zeros(n, np.float32)
        for i in range(0, n, 512):
            sl = {k: np.asarray(v[i : i + 512]) for k, v in batch.items()}
            out[i : i + 512] = np.clip(np.asarray(self._run("value", _unbatch(sl), False)).reshape(-1), -1, 1)
        return out


def _unbatch(batch: dict[str, np.ndarray]) -> list[dict]:
    n = next(iter(batch.values())).shape[0]
    return [{k: v[i] for k, v in batch.items()} for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", required=True)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choice-seed", type=int, default=0)
    ap.add_argument("--game-offset", type=int, default=0)
    ap.add_argument("--onnx", type=Path, default=ROOT / "data/cache/onnx/r1")
    ap.add_argument("--emb", type=Path, default=ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "data/cache/app/product_py.jsonl.gz")
    ap.add_argument("--io-hash-out", type=Path, default=None)
    a = ap.parse_args(argv)

    from decklight.decks import load_ydk

    pairs = parse_pairs(a.pairs)
    decks = [(load_ydk(x), load_ydk(y)) for _, x, y in pairs]
    engine = Engine()
    nets = OrtNets(a.onnx, a.emb, a.io_hash_out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    wall_search: list[float] = []
    n_searched = n_over = n_branches = n_capped = n_skip = 0

    with gzip.open(a.out, "wt") as w:
        for gi in range(a.games):
            g = a.game_offset + gi
            name, _, _ = pairs[g % len(pairs)]
            deck0, deck1 = decks[g % len(decks)]
            s = a.seed + g
            cfg = DuelConfig(deck0, deck1, seed=s)
            s_seat = int(g % 2)
            route: dict[int, tuple[int, int] | None] = {}
            kw = dict(product_search_kwargs(fast=False))
            search_kw = {k: kw.pop(k) for k in ("k", "horizon", "samples", "rollout_temp")}
            gen_kw = {
                "conf_skip": kw.pop("conf_skip"),
                "own_frac": kw.pop("own_frac"),
                "own_turn1": kw.pop("own_turn1"),
                "honest": kw.pop("honest"),
                "det_tries": kw.pop("det_tries"),
                "det_fallback": kw.pop("det_fallback"),
                "margin": kw.pop("margin"),
                "max_search_turn": kw.pop("max_search_turn"),
                "max_searches": kw.pop("max_searches"),
                "ho": HonestRule() if kw.pop("ho", False) else None,
            }
            assert not kw, kw
            search = build_search(
                engine, seed=s, value=nets.values_batch, route=route, pool=None, **search_kw
            )
            nets.game = g
            nets.io_idx = 0
            human = SplitMix(a.choice_seed ^ g)
            pending: list[dict] = []

            def forward(req):
                missing = object()
                base_at, base_obs, r_at = [], [], []
                outer = None
                for i, o in enumerate(req):
                    tag = route.pop(id(o), missing)  # noqa: B023 -- the closure runs within its own game iteration
                    if tag is missing:
                        raise RuntimeError("untagged obs in the batch")
                    if tag is None:
                        base_at.append(i)
                        base_obs.append(o)
                    else:
                        if len(req) != 1:
                            raise RuntimeError("mixed rollout/outer batch")
                        _, s_seat_tag = tag
                        if seat_of(o) == s_seat_tag:
                            base_at.append(i)
                            base_obs.append(o)
                            outer = ("s", o)
                        else:
                            r_at.append(i)
                            outer = ("r", o)
                a_width = int(req[0]["action_mask"].shape[0])
                lg = np.zeros((len(req), a_width), np.float32)
                if base_obs:
                    b = np.asarray(nets.policy_logits(base_obs))
                    if b.shape[1] != a_width:
                        raise RuntimeError(f"policy width {b.shape[1]} != {a_width}")
                    for k, i in enumerate(base_at):
                        lg[i] = b[k]
                if outer is not None:
                    kind, o = outer
                    pending.append(  # noqa: B023
                        {
                            "actor": int(seat_of(o)),
                            "n": int(o["action_mask"].sum()),
                            "turn": int(o["global"][2]),
                        }
                    )
                    if kind == "r":
                        n_legal = int(o["action_mask"].sum())
                        h = int(human.next() % n_legal)  # noqa: B023
                        lg[0, h] = 1.0
                return lg

            t0 = time.perf_counter()
            gen = tag_game(
                0,
                s_seat,
                search_game_gen(
                    engine,
                    cfg,
                    search,
                    s_seat,
                    argmax_select,
                    game_seed=s,
                    guard_mode="per_seat",
                    unguarded_seat=1 - s_seat,
                    latency=True,
                    **gen_kw,
                ),
                route,
            )
            req = next(gen)
            try:
                while True:
                    fb = forward(req)
                    req = gen.send(fb)
            except StopIteration as stop:
                res = stop.value
            game_wall = time.perf_counter() - t0
            wall_search.extend(res.lat_search)
            n_searched += res.n_searched
            n_over += res.n_override
            n_branches += res.n_branches
            n_capped += res.n_capped
            n_skip += res.det_skipped

            searched = set(res.searched_idx)
            assert len(pending) == len(res.choices), (len(pending), len(res.choices))
            si = ri = 0
            for k, (p, chosen) in enumerate(zip(pending, res.choices, strict=True)):
                is_s = k in searched
                if is_s:
                    wall = res.lat_search[si]
                    si += 1
                else:
                    wall = res.lat_raw[ri]
                    ri += 1
                w.write(
                    json.dumps(
                        {
                            "kind": "decision",
                            "g": g,
                            "k": k,
                            "actor": p["actor"],
                            "turn": p["turn"],
                            "n": p["n"],
                            "chosen": int(chosen),
                            "searched": bool(is_s),
                            "wall_bits": bits(wall),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            for e in res.log:
                scores_all = {kk: [bits(float(x)) for x in col] for kk, col in e["scores_all"].items()}
                samples = [[bits(round(float(x), 4)) for x in col] for col in e["samples"]]
                ho = None
                if "ho" in e:
                    h = e["ho"]
                    ho = {
                        "status": h["status"],
                        "gap_bits": bits(h["gap"]),
                        "se_bits": None if h["se"] is None else bits(h["se"]),
                        "n": int(h["n"]),
                        "challenger": h["challenger"],
                        "rounds": int(h["rounds"]),
                    }
                w.write(
                    json.dumps(
                        {
                            "kind": "log",
                            "g": g,
                            "k": e["k"],
                            "seat": e["seat"],
                            "kind2": e["kind"],
                            "turn": e["turn"],
                            "cands": [int(x) for x in e["cands"]],
                            "scores_bits": [bits(float(x)) for x in e["scores"]],
                            "scores_all_bits": scores_all,
                            "chosen": int(e["chosen"]),
                            "played": int(e["played"]),
                            "best": int(e["best"]),
                            "samples_bits": samples,
                            "ho": ho,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            w.write(
                json.dumps(
                    {
                        "kind": "game",
                        "g": g,
                        "pair": name,
                        "s_seat": s_seat,
                        "seed": s,
                        "choice_seed": a.choice_seed,
                        "ws_seed": s,
                        "game_seed": s,
                        "winner": res.winner,
                        "truncated": bool(res.truncated),
                        "turns": int(res.turns),
                        "choices": [int(x) for x in res.choices],
                        "actors": [int(x) for x in res.actors],
                        "decision_turns": [int(x) for x in res.decision_turns],
                        "n_searched": res.n_searched,
                        "n_override": res.n_override,
                        "n_branches": res.n_branches,
                        "n_capped": res.n_capped,
                        "det_proposals": res.det_proposals,
                        "det_accepted": res.det_accepted,
                        "det_fallbacks": res.det_fallbacks,
                        "det_skipped": res.det_skipped,
                        "search_cut": bool(res.search_cut),
                        "lat_raw_bits": [bits(x) for x in res.lat_raw],
                        "lat_search_bits": [bits(x) for x in res.lat_search],
                        "lat_turn_bits": [bits(x) for x in res.lat_turn],
                        "game_wall_bits": bits(game_wall),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            print(
                f"product_games_py: g={g} pair={name} S=seat{s_seat} winner={res.winner} "
                f"decisions={len(res.choices)} searched={res.n_searched} overrides={res.n_override} "
                f"det_skip={res.det_skipped} wall={game_wall:.1f}s",
                flush=True,
            )
    if nets.io_f is not None:
        nets.io_f.close()
        print(f"product_games_py: ORT I/O hashes -> {a.io_hash_out}", flush=True)
    arr = np.asarray(wall_search, dtype=np.float64)
    if len(arr):
        print(
            f"product_games_py: searched={n_searched} overrides={n_over} branches={n_branches} "
            f"capped={n_capped} det_skipped={n_skip} s/search mean={arr.mean():.3f} "
            f"p50={np.percentile(arr, 50):.3f} p90={np.percentile(arr, 90):.3f} max={arr.max():.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
