"""Generate CPython `random.Random` fixtures for the Rust A3a port (the Python side is the oracle).

    uv run python scripts/app/gen_search_fixtures.py --out data/cache/app/search_rng_fixtures.jsonl.gz

One JSON line per seed (int or str). Each line records the first 50 `random()` values (f64 bit patterns),
`getrandbits(k)` for k in 1..=70, `_randbelow(n)` for small and huge n, `shuffle` of lists of length 0..60 (a fresh
`Random(seed)` per length), and a `getstate`/`setstate` round trip mid-stream (the 625 state words and the next five
values after restoring). `scripts/app/search_rng_check.rs` (Rust) replays every line and compares.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import struct
import sys

M32 = (1 << 32) - 1


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def int_seed() -> str:
    return (
        "0,1,-1,-7,4294967295,4294967297,18446744073709551615,18446744073709551620,"
        "1267650600228229401496703205375,-1267650600228229401496703205375"
    )


def build_seeds() -> tuple[list[int], list[str]]:
    ints: list[int] = [int(x) for x in int_seed().split(",")]
    # The search's real int seed form: seed * 1_000_003 + game_seed.
    for base in (0, 1, 7, 12345, 2**20 + 3, 2**32 - 1, 2**40 + 17):
        for gs in (0, 1, 2, 10**6):
            ints.append(base * 1_000_003 + gs)
    # A deterministic spread of arbitrary sizes (including > u64).
    v = 0x9E3779B97F4A7C15
    for _ in range(220):
        v = (v * 6364136223846793005 + 1442695040888963407) & ((1 << 101) - 1)
        ints.append(v - (1 << 100))
    seen = set()
    uniq = []
    for s in ints:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    uniq = uniq[:260]

    strings: list[str] = ["", " ", "Δ", "日本語", "a:b:c:det", "0:0:0:det", "7:3:5:det"]
    for s in (0, 1, 7, 12345, 2**32 - 1):
        for g in (0, 1, 10**6):
            for k in (0, 1, 17):
                strings.append(f"{s}:{g}:{k}:det")
                strings.append(f"{s}:{g}:{k}:4:2")
    v = 0x2545F4914F6CDD1D
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz:Δ日-_"
    for _ in range(220):
        v = (v * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        n = 1 + (v % 24)
        chars = []
        for j in range(n):
            v = (v * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            chars.append(alphabet[v % len(alphabet)])
        strings.append("".join(chars))
    seen = set()
    uniq_s = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            uniq_s.append(s)
    return uniq, uniq_s[:260]


def record(kind: str, seed: int | str) -> dict:
    rng = random.Random(seed)
    random50 = [bits(rng.random()) for _ in range(50)]
    grb = [[k, str(rng.getrandbits(k))] for k in range(1, 71)]
    ns = [1, 2, 3, 7, 10, 1000, 2**53 - 1, 2**64 + 1, 2**100 + 7]
    randbelow = [[str(n), str(rng._randbelow(n))] for n in ns]
    shuffles = []
    for n in range(61):
        r = random.Random(seed)
        lst = list(range(n))
        r.shuffle(lst)
        shuffles.append(lst)
    r = random.Random(seed)
    for _ in range(10):
        r.random()
    st = r.getstate()
    words = list(st[1])
    r2 = random.Random(0)
    r2.setstate(st)
    after = [bits(r2.random()) for _ in range(5)]
    also = [bits(r.random()) for _ in range(5)]
    assert after == also, "getstate/setstate round trip changed the stream"
    return {
        "kind": kind,
        "seed": str(seed),
        "random50": random50,
        "getrandbits": grb,
        "randbelow": randbelow,
        "shuffle": shuffles,
        "state_words": words,
        "after_restore": after,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cache/app/search_rng_fixtures.jsonl.gz")
    a = ap.parse_args(argv)
    ints, strings = build_seeds()
    n = 0
    with gzip.open(a.out, "wt", compresslevel=6) as f:
        for s in ints:
            f.write(json.dumps(record("int_seed", s), separators=(",", ":")) + "\n")
            n += 1
        for s in strings:
            f.write(json.dumps(record("str_seed", s), separators=(",", ":")) + "\n")
            n += 1
    print(f"wrote {n} fixtures ({len(ints)} int, {len(strings)} str) to {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
