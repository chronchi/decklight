"""Export the product policy/value nets to ONNX (offline-app spike S2).

    uv run python scripts/export_onnx.py --ckpt data/cache/forge_r1/best.pt \\
        --value data/cache/value_p2/best.pt --emb data/cards/emb/qwen3-0.6b-d512.npz \\
        --out data/cache/onnx/r1

Writes ``policy.onnx``, ``value.onnx``, ``card_proj.onnx``, ``codes.json``,
``unknown.npy`` (the raw learned unknown vector: the app builds the lookup
table as ``concat([proj(unknown), proj(raw_rows)])``) and ``manifest.json``
(sha256 of inputs/outputs). Runs a torch-vs-ORT parity check on a random
batch first; on failure it prints ``ALERT`` and exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.eval.bc_policy import _run_args
from decklight.model.ckpt import load_state
from decklight.model.onnx_export import (
    CODE_COLUMNS,
    CODES_VERSION,
    codes_to_rows,
    example_obs,
    export_nets,
    history_source,
    project_table,
    raw_table,
    sha256,
    sorted_codes,
    with_history_source,
)
from decklight.model.policy_v3 import make_policy

# The in-script smoke check runs on random (deliberately hostile) obs, where
# kernel-ordering noise amplifies through attention (~2e-4 observed); it exists
# to catch structural export bugs (wrong columns, dropped terms: O(0.1-10)
# effects), not to certify numerics. The 1e-4 gate on real engine states lives
# in scripts/onnx_parity.py.
SMOKE_GATE = 1e-3


def build_net(ckpt: Path, emb: dict, value: bool):
    a = _run_args(Path(ckpt))
    net = make_policy(emb, a, value=value)
    state = load_state(ckpt)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return net.eval()


def random_obs(codes: np.ndarray, batch: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = {k: v.numpy() for k, v in example_obs(batch).items()}
    obs["cards"][:, :, 0] = rng.choice(np.concatenate([[0], codes]), size=obs["cards"][:, :, 0].shape)
    obs["card_mask"] = rng.random(obs["card_mask"].shape) < 0.5
    obs["chain"][:, :, 1] = rng.choice(np.concatenate([[0], codes]), size=obs["chain"][:, :, 1].shape)
    obs["chain_mask"] = rng.random(obs["chain_mask"].shape) < 0.2
    obs["actions"][:, :, 2:4] = rng.choice(
        np.concatenate([[0], codes]), size=obs["actions"][:, :, 2:4].shape
    )
    obs["actions"][:, :, 1] = rng.integers(-1, 160, size=obs["actions"][:, :, 1].shape)
    obs["action_mask"] = rng.random(obs["action_mask"].shape) < 0.3
    obs["action_mask"][:, 0] = True  # at least one legal option per row
    obs["history"][:, :, 2:4] = rng.choice(
        np.concatenate([[0], codes]), size=obs["history"][:, :, 2:4].shape
    )
    obs["history_mask"] = rng.random(obs["history_mask"].shape) < 0.5
    # Observable-history window: independent values, so a wrong source (history vs obs_history) shows up.
    obs["obs_history"] = obs["history"].copy()
    obs["obs_history"][:, :, 2:4] = rng.choice(
        np.concatenate([[0], codes]), size=obs["obs_history"][:, :, 2:4].shape
    )
    obs["obs_history_mask"] = rng.random(obs["history_mask"].shape) < 0.5
    obs["global"] = rng.integers(0, 8000, size=obs["global"].shape, dtype=np.int32)
    other = rng.integers(0, 40, size=obs["cards"].shape, dtype=np.int32)
    other[:, :, 0] = obs["cards"][:, :, 0]  # keep the sampled codes
    obs["cards"] = other
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    emb = dict(np.load(args.emb))
    codes = sorted_codes(emb)
    raw = raw_table(emb)
    policy_net = build_net(Path(args.ckpt), emb, value=False)
    value_net = build_net(Path(args.value), emb, value=True)
    d = policy_net.text.proj[0].out_features
    r = raw.shape[1]
    assert policy_net.text.proj[0].in_features == r, (policy_net.text.proj[0].in_features, r)

    paths = export_nets(policy_net, value_net, (len(codes) + 1, d, r), out, opset=args.opset)

    (out / "codes.json").write_text(json.dumps({
        "version": CODES_VERSION,
        "columns": CODE_COLUMNS,
        "row_of": "sorted codes from the npz; row = 1 + position; 0 = unknown or code <= 0",
    }, indent=2))
    unknown_raw = policy_net.text.unknown.detach().float().cpu().numpy().astype(np.float32)
    np.save(out / "unknown.npy", unknown_raw)
    unknown_value_raw = value_net.text.unknown.detach().float().cpu().numpy().astype(np.float32)
    np.save(out / "unknown_value.npy", unknown_value_raw)

    # torch-vs-ORT smoke check on a random batch, per graph with its own table
    # (fail loud: ALERT + exit 2).
    import onnxruntime as ort

    obs = random_obs(codes, 8, seed=0)
    rows = codes_to_rows(obs, codes)
    text = project_table(policy_net.text, raw)
    text_v = project_table(value_net.text, raw)
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obs.items()}
    torch.backends.mha.set_fastpath_enabled(False)  # slow path: what the graphs trace
    try:
        with torch.no_grad():
            ref_logits = policy_net(t).numpy()
            ref_value = value_net.forward_value(t)[1].numpy()
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    # The graph always takes `history` / `history_mask`; an observable policy is fed obs_history there.
    src_p, src_v = history_source(policy_net), history_source(value_net)
    feed = {k: np.ascontiguousarray(v) for k, v in with_history_source(rows, src_p).items()}
    pol_sess = ort.InferenceSession(str(paths["policy"]), providers=["CPUExecutionProvider"])
    val_sess = ort.InferenceSession(str(paths["value"]), providers=["CPUExecutionProvider"])
    proj_sess = ort.InferenceSession(str(paths["card_proj"]), providers=["CPUExecutionProvider"])
    proj_v_sess = ort.InferenceSession(str(paths["card_proj_value"]), providers=["CPUExecutionProvider"])
    got_text = proj_sess.run(["text"], {"raw": np.ascontiguousarray(
        np.concatenate([unknown_raw.reshape(1, -1), raw]))})[0]
    got_text_v = proj_v_sess.run(["text"], {"raw": np.ascontiguousarray(
        np.concatenate([unknown_value_raw.reshape(1, -1), raw]))})[0]
    dt = float(np.abs(got_text - text).max())
    dtv = float(np.abs(got_text_v - text_v).max())
    feed["text"] = np.ascontiguousarray(text)
    got_logits = pol_sess.run(
        ["logits"], {i.name: feed[i.name] for i in pol_sess.get_inputs()})[0]
    feed_v = {k: np.ascontiguousarray(v) for k, v in with_history_source(rows, src_v).items()}
    feed_v["text"] = np.ascontiguousarray(text_v)
    got_value = val_sess.run(
        ["value"], {i.name: feed_v[i.name] for i in val_sess.get_inputs()})[0]
    legal = obs["action_mask"].astype(bool)
    dl = float(np.abs(got_logits[legal] - ref_logits[legal]).max()) if legal.any() else 0.0
    dv = float(np.abs(got_value - ref_value).max())
    ok = dl <= SMOKE_GATE and dv <= SMOKE_GATE and dt <= SMOKE_GATE and dtv <= SMOKE_GATE
    import onnx

    def io_names(p: Path):
        m = onnx.load(str(p))
        return {"inputs": [i.name for i in m.graph.input],
                "outputs": [o.name for o in m.graph.output]}

    files = {p.name: sha256(p) for p in
             [paths["policy"], paths["value"], paths["card_proj"], paths["card_proj_value"],
              out / "codes.json", out / "unknown.npy", out / "unknown_value.npy"]}
    (out / "manifest.json").write_text(json.dumps({
        "inputs": {str(args.ckpt): sha256(Path(args.ckpt)), str(args.emb): sha256(Path(args.emb)),
                   str(args.value): sha256(Path(args.value))},
        "outputs": files,
        "graphs": {name: io_names(paths[name]) for name in
                   ("policy", "value", "card_proj", "card_proj_value")},
        "dims": {"table_rows": len(codes) + 1, "d": d, "row_width": r, "n_codes": len(codes)},
        "opset": args.opset,
        "history_source": src_p,
        "value_history_source": src_v,
        "parity": {"max_abs_logit": dl, "max_abs_value": dv, "max_abs_text": dt,
                   "max_abs_text_value": dtv, "gate": SMOKE_GATE, "passed": bool(ok)},
    }, indent=2))
    print(json.dumps({"max_abs_logit": dl, "max_abs_value": dv, "max_abs_text": dt,
                      "max_abs_text_value": dtv, "gate": SMOKE_GATE,
                      "sizes": {k: Path(paths[k]).stat().st_size for k in paths}}))
    if not ok:
        print(f"ALERT export parity failed: dlogit={dl:.3g} dvalue={dv:.3g} "
              f"dtext={dt:.3g} dtext_v={dtv:.3g} gate={SMOKE_GATE}")
        return 2
    for name in ("policy", "value", "card_proj", "card_proj_value"):
        print(f"{name}.onnx: {paths[name].stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
