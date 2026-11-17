"""Generate the Rust sampling-walk fixtures (`rust/decklight-search/tests/fixtures/window_sampling.jsonl`).

    uv run python scripts/app/gen_window_fixtures.py

Each line is one `WindowSearch.rollouts_gen` sample walk: a 64-entry float64 logit vector with some `-inf` entries,
a rollout temperature, a CPython `random.Random` string seed, and the chosen option. Floats are bit patterns so the
comparison is exact. Additive/refreshable; the Rust unit test reads the committed file.
"""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rust/decklight-search/tests/fixtures/window_sampling.jsonl"


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def walk(z: np.ndarray, temp: float, seed: str) -> int:
    finite = np.isfinite(z)
    vals = z[finite] / temp
    vals = vals - np.max(vals)
    exp = np.exp(vals)
    probs = exp / exp.sum()
    r = random.Random(seed).random()
    cum = 0.0
    legal = np.flatnonzero(finite)
    a = int(legal[-1])
    for li, p in zip(legal, probs, strict=True):
        cum += float(p)
        if r < cum:
            a = int(li)
            break
    return a


def main() -> int:
    rng = np.random.default_rng(20261102)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temps = [1.0, 0.5, 2.0, 0.1, 1.25]
    n = 300
    with open(OUT, "w") as f:
        for i in range(n):
            z = np.empty(64, np.float64)
            mode = i % 5
            for j in range(64):
                if rng.random() < (0.15 if mode else 0.45):
                    z[j] = -np.inf
                elif mode == 1:
                    z[j] = float(rng.integers(-4, 5)) * 0.5  # ties
                elif mode == 2:
                    z[j] = float(rng.normal()) * 20.0  # wide scale
                else:
                    z[j] = float(rng.uniform(-8.0, 8.0))
            if not np.isfinite(z).any():
                z[0] = 0.0
            temp = temps[i % len(temps)]
            seed = f"window-fixture:{i}:{i % 7}"
            a = walk(z, temp, seed)
            f.write(
                json.dumps(
                    {
                        "seed": seed,
                        "z_bits": [bits(float(x)) for x in z],
                        "temp_bits": bits(temp),
                        "a": a,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    print(f"wrote {n} sampling fixtures -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
