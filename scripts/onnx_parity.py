"""ONNX parity for the product nets (offline-app spike S2).

Collect: play engine games over the deck pool with the product policy itself
(argmax, 0.2 random legal) and save decision observations + torch outputs::

    uv run python scripts/onnx_parity.py --ckpt data/cache/forge_r1/best.pt \\
        --value data/cache/value_p2/best.pt --emb data/cards/emb/qwen3-0.6b-d512.npz \\
        --collect data/cache/onnx_parity/obs --n 10000

Compare onnxruntime (Python) against the saved torch outputs::

    uv run python scripts/onnx_parity.py --onnx data/cache/onnx/r1 --obs data/cache/onnx_parity/obs

Compare a Rust spike binary's outputs (it reads the same --onnx/--obs dirs)::

    uv run python scripts/onnx_parity.py --onnx data/cache/onnx/r1 --obs data/cache/onnx_parity/obs \\
        --rust rust/spikes/onnx-parity/target/release/onnx-parity --out /tmp/opencode/rust_out

Gate: max |dlogit| over legal actions <= 1e-4, max |dvalue| <= 1e-4, argmax
agreement 100% outside exact ties (top-two torch logits within 1e-5). Prints
a JSON summary; exits 2 on failure (ALERT line first).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GATE = 1e-4
TIE = 1e-5

PAIRS = [
    ("blue-eyes", "dark-magician"),
    ("branded", "dinosaur"),
    ("drytron", "eldlich"),
    ("burn", "exodia"),
]


def collect(out_dir: Path, ckpt: str, value_ckpt: str, emb_path: str, n: int, seed: int) -> Path:
    import torch

    from decklight.decks import load_ydk
    from decklight.engine.cpp_env import VecEnv
    from decklight.engine.duel import Engine
    from decklight.eval.bc_policy import BCPolicy
    from decklight.eval.value import ValueNet
    from decklight.model.bc_data import OBS_HISTORY_KEYS, OBS_KEYS

    out_dir.mkdir(parents=True, exist_ok=True)
    decks = {name: load_ydk(ROOT / "decks/pool" / f"{name}.ydk") for pair in PAIRS for name in pair}
    deck_list = sorted(decks.values(), key=lambda d: d.name)
    idx = {d.name: i for i, d in enumerate(deck_list)}
    sched, meta = [], []
    s = seed
    while len(sched) < 64:
        for a, b in PAIRS:
            for swap in (False, True):
                da, db = (b, a) if swap else (a, b)
                sched.append((idx[da], idx[db], s))
                meta.append((da, db, s))
                s += 1
    engine = Engine()
    vec = VecEnv(engine, deck_list, min(8, len(sched)), 3, seed=seed)
    vec.set_schedule(sched)
    vec.set_max_decisions(600)
    pol = BCPolicy(ckpt, emb_path, loop_guard=False, threads=1, batch=512)
    val = ValueNet(value_ckpt, emb_path, batch=512, threads=1)
    # An observable net (policy or value) needs obs_history/obs_history_mask; save them with the decision rows.
    observable = getattr(pol.model, "observable", False) or getattr(val.model, "observable", False)
    keys = tuple(OBS_KEYS) + (tuple(OBS_HISTORY_KEYS) if observable else ())
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    t0 = time.time()
    buf = vec.reset()
    while vec.remaining > 0 and len(rows) < n:
        player = buf["player"]
        live = np.flatnonzero(player >= 0)
        if len(live) == 0:
            break
        sub = {k: np.ascontiguousarray(buf[k][live]) for k in keys}
        logits = pol.logits(sub)
        acts = np.argmax(np.where(sub["action_mask"], logits, -np.inf), axis=1).astype(np.int32)
        flip = rng.random(len(live)) < 0.2
        for j in np.flatnonzero(flip):
            legal = np.flatnonzero(sub["action_mask"][j])
            if len(legal):
                acts[j] = legal[rng.integers(len(legal))]
        for j, e in zip(live.tolist(), range(len(live))):
            rows.append({k: np.array(buf[k][j]) for k in keys})
            if len(rows) >= n:
                break
        actions = np.zeros(vec.num_envs, np.int32)
        actions[live] = acts[: len(live)]
        vec.step(actions)
        if len(rows) % 2000 < len(live):
            print(f"  ... {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    print(f"collected {len(rows)} decisions from {len(sched) - vec.remaining} games "
          f"in {time.time() - t0:.0f}s", flush=True)
    if not rows:
        print("ALERT onnx_parity: collected 0 decisions")
        raise SystemExit(2)
    batch = {k: np.stack([r[k] for r in rows]) for k in keys}
    for k, v in batch.items():
        np.save(out_dir / f"{k}.npy", v)
    with torch.no_grad():
        ref_logits = pol.logits(batch)
    np.save(out_dir / "torch_logits.npy", ref_logits.astype(np.float32))
    np.save(out_dir / "torch_value.npy", val.values(batch).astype(np.float32))
    emb = dict(np.load(emb_path))
    from decklight.model.onnx_export import raw_table, sorted_codes

    np.save(out_dir / "raw.npy", raw_table(emb))
    np.save(out_dir / "codes.npy", sorted_codes(emb))
    (out_dir / "meta.json").write_text(json.dumps({
        "n": len(rows), "ckpt": ckpt, "value": value_ckpt, "emb": emb_path,
        "pairs": PAIRS, "seed": seed, "games": len(sched) - vec.remaining,
        "decks": sorted(decks),
    }, indent=2))
    return out_dir


def _load_obs(obs_dir: Path):
    from decklight.model.bc_data import OBS_HISTORY_KEYS, OBS_KEYS

    keys = tuple(OBS_KEYS) + tuple(k for k in OBS_HISTORY_KEYS if (obs_dir / f"{k}.npy").is_file())
    batch = {k: np.load(obs_dir / f"{k}.npy") for k in keys}
    ref_logits = np.load(obs_dir / "torch_logits.npy")
    ref_value = np.load(obs_dir / "torch_value.npy")
    return batch, ref_logits, ref_value


def compare(torch_logits, torch_value, got_logits, got_value, mask) -> dict:
    legal = mask.astype(bool)
    dl = np.abs(got_logits - torch_logits)
    dl = float(dl[legal].max()) if legal.any() else 0.0
    dv = float(np.abs(got_value - torch_value).max())
    agree, ties, total = 0, 0, len(torch_logits)
    for i in range(total):
        lv = torch_logits[i][legal[i]]
        go = got_logits[i][legal[i]]
        if not len(lv):
            total -= 1
            continue
        a, b = int(np.argmax(lv)), int(np.argmax(go))
        if len(lv) >= 2:
            s = np.sort(lv)
            if s[-1] - s[-2] < TIE:
                ties += 1
                continue
        agree += a == b
    denom = total - ties
    return {"n": total, "max_abs_logit": dl, "max_abs_value": dv,
            "argmax_agree": agree, "argmax_denom": denom,
            "argmax_rate": (agree / denom) if denom else 1.0, "ties": ties,
            "passed": bool(dl <= GATE and dv <= GATE and (denom == 0 or agree == denom))}


def run_ort_full(onnx_dir: Path, obs_dir: Path, chunk: int = 64,
                 force_policy_source: str | None = None):
    import onnxruntime as ort

    from decklight.model.bc_data import OBS_HISTORY_KEYS, OBS_KEYS
    from decklight.model.onnx_export import (
        codes_to_rows,
        pad_logits,
        read_metadata,
        trim_obs_np,
        with_history_source,
    )

    codes = np.load(obs_dir / "codes.npy")
    man = json.loads((onnx_dir / "manifest.json").read_text())
    io_maps = man["graphs"]
    # The metadata on each graph names its history input source (absent = "history", the old exports).
    src_p = force_policy_source or read_metadata(onnx_dir / "policy.onnx").get("history_source", "history")
    src_v = read_metadata(onnx_dir / "value.onnx").get("history_source", "history")
    keys = tuple(OBS_KEYS) + tuple(k for k in OBS_HISTORY_KEYS if (obs_dir / f"{k}.npy").is_file())
    unknown = np.load(onnx_dir / "unknown.npy").reshape(1, -1).astype(np.float32)
    unknown_v = np.load(onnx_dir / "unknown_value.npy").reshape(1, -1).astype(np.float32)
    raw = np.load(obs_dir / "raw.npy").astype(np.float32)
    sess = {k: ort.InferenceSession(str(onnx_dir / f"{k}.onnx"), providers=["CPUExecutionProvider"])
            for k in ("card_proj", "card_proj_value", "policy", "value")}
    n = next(iter(np.load(obs_dir / "cards.npy").shape[:1]))
    text = sess["card_proj"].run(["text"], {"raw": np.concatenate([unknown, raw])})[0]
    text_v = sess["card_proj_value"].run(["text"], {"raw": np.concatenate([unknown_v, raw])})[0]
    outs_l, outs_v = [], []
    for i in range(0, n, chunk):
        sl = {k: np.load(obs_dir / f"{k}.npy", mmap_mode="r")[i:i + chunk] for k in keys}
        sl = codes_to_rows({k: np.asarray(v) for k, v in sl.items()}, codes)
        # dynamic token axes: the app trims the padding tail like PolicyNet.trim, on the graph's history source
        sl_p, n_opt = trim_obs_np(with_history_source(sl, src_p))
        feed_p = {name: np.ascontiguousarray(sl_p[name]) for name in io_maps["policy"]["inputs"]
                  if name != "text"}
        feed_p["text"] = text
        outs_l.append(pad_logits(sess["policy"].run(["logits"], feed_p)[0], n_opt))
        sl_v, _ = trim_obs_np(with_history_source(sl, src_v))
        feed_v = {name: np.ascontiguousarray(sl_v[name]) for name in io_maps["value"]["inputs"]
                  if name != "text"}
        feed_v["text"] = text_v
        outs_v.append(sess["value"].run(["value"], feed_v)[0])
    return np.concatenate(outs_l), np.concatenate(outs_v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/cache/forge_r1/best.pt")
    ap.add_argument("--value", default="data/cache/value_p2/best.pt")
    ap.add_argument("--emb", default="data/cards/emb/qwen3-0.6b-d512.npz")
    ap.add_argument("--collect", default=None, help="collect N decisions into DIR and exit")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--obs", default="data/cache/onnx_parity/obs")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rust", default=None, help="rust spike binary to compare")
    ap.add_argument("--out", default=None, help="rust output dir")
    ap.add_argument("--history-source", choices=("history", "obs_history"), default=None,
                    help="override the policy graph's history source (wrong-source check)")
    args = ap.parse_args()

    if args.collect:
        collect(Path(args.collect), args.ckpt, args.value, args.emb, args.n, args.seed)
        return 0
    if not args.onnx:
        print("need --onnx DIR (or --collect DIR)")
        return 2
    onnx_dir, obs_dir = Path(args.onnx), Path(args.obs)
    batch, ref_logits, ref_value = _load_obs(obs_dir)
    summary: dict = {"gate": GATE, "tie_eps": TIE, "n": len(ref_logits)}
    if args.rust:
        out = Path(args.out or "data/cache/onnx_parity/rust_out")
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cmd = [args.rust, "--onnx", str(onnx_dir), "--obs", str(obs_dir), "--out", str(out)]
        if args.history_source:
            cmd += ["--history-source", args.history_source]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
        summary["rust_wall_s"] = round(time.time() - t0, 1)
        summary["rust_stdout_tail"] = r.stdout[-2000:]
        summary["rust_stderr_tail"] = r.stderr[-2000:]
        print(r.stdout[-2000:], flush=True)
        if r.returncode != 0:
            print(f"ALERT onnx_parity: rust binary failed rc={r.returncode}")
            print(r.stderr[-3000:])
            return 2
        try:
            rust_bench = json.loads(r.stdout[r.stdout.rfind("{"):])
            summary["rust_bench"] = rust_bench
        except (ValueError, IndexError):
            pass
        got_logits = np.load(out / "policy_logits.npy")
        got_value = np.load(out / "value.npy")
        summary["rust"] = compare(ref_logits, ref_value, got_logits, got_value, batch["action_mask"])
    else:
        t0 = time.time()
        got_logits, got_value = run_ort_full(
            onnx_dir, obs_dir, force_policy_source=args.history_source
        )
        summary["ort_wall_s"] = round(time.time() - t0, 1)
        summary["python_ort"] = compare(ref_logits, ref_value, got_logits, got_value,
                                        batch["action_mask"])
    print(json.dumps(summary, indent=2))
    ok = all(v.get("passed", True) for v in summary.values() if isinstance(v, dict) and "passed" in v)
    if not ok:
        print("ALERT onnx_parity: gate failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
