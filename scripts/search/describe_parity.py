"""Parity check: the Rust fp16 query encoder vs the Python reference (`semantic_eval.Encoder`).

Runs `describe_encode` (built from `rust/decklight-server`) and the reference on 6 fixed queries (with the
query instruction), prints the cosine per query and exits 1 if any is below 0.999.

    ORT_DYLIB_PATH=<libonnxruntime.so> uv run --no-project --with onnxruntime==1.31.1 \
      --with tokenizers==0.22.1 --with numpy python scripts/search/describe_parity.py --model-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from semantic_eval import INSTRUCT, Encoder  # noqa: E402

QUERIES = [
    "a light tuner that special summons itself and searches",
    "a card that negates a monster effect and destroys it",
    "a card that special summons a monster from the deck",
    "a trap that negates the activation of a spell",
    "a card that banishes cards from the graveyard and draws",
    "a level 4 dragon that can attack twice and adds a card from deck to hand",
]


def cargo() -> str:
    return os.environ.get("CARGO") or shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")


def run_bin(model_dir: Path, texts: list[str]) -> list[np.ndarray]:
    subprocess.run(
        [cargo(), "build", "-q", "-p", "decklight-server", "--bin", "describe_encode"],
        cwd=ROOT / "rust",
        check=True,
    )
    exe = ROOT / "rust/target/debug/describe_encode"
    if not exe.is_file():
        raise SystemExit(f"ALERT describe_parity: built binary not found at {exe}")
    out = subprocess.run(
        [str(exe), str(model_dir), *texts], check=True, capture_output=True, text=True
    )
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    if len(rows) != len(texts):
        raise SystemExit(f"ALERT describe_parity: {len(rows)} vectors for {len(texts)} texts")
    return [np.asarray(r, np.float32) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    a = ap.parse_args()
    if not os.environ.get("ORT_DYLIB_PATH"):
        raise SystemExit("ALERT describe_parity: ORT_DYLIB_PATH is not set")

    texts = [INSTRUCT + q for q in QUERIES]
    ref = Encoder(a.model_dir, "fp16")
    want = [ref(t) for t in texts]
    got = run_bin(a.model_dir, texts)

    worst = 1.0
    for q, a_vec, b_vec in zip(QUERIES, want, got):
        if a_vec.shape != b_vec.shape:
            raise SystemExit(f"ALERT describe_parity: shape {b_vec.shape} != {a_vec.shape}")
        cos = float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))
        worst = min(worst, cos)
        print(f"cosine {cos:.9f}  {q}", flush=True)
    print(f"min cosine {worst:.9f}", flush=True)
    if worst < 0.999:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
