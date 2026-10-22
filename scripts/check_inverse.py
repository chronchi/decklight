"""Round-trip check for engine/inverse.py on stored game records (EDOPro play D1).

For N records, replays with PyEnv and splits each game into per-response groups
(`inverse.decision_groups`: one raw core response per pending decision; decisions taken
entirely by auto-single steps have empty index slices). For every group with a non-empty
slice, `choices_for_response` on an incremental replay at that decision must return the
record's own index sequence -- or a different sequence that yields identical bytes
(verified by re-sending it and reading the first new response). Reports counts per
message type, Nones, mismatches, max nodes searched and call latency. Exit 2 on any None
or mismatch.

Two record formats: gen_selfplay (`config` + `choices`, e.g. runs/aux-*/games/p*/) and
search_match (`choices` + deck-name `decks` + `seed`, e.g. runs/*/g*/games-0.jsonl).

    uv run python scripts/check_inverse.py --records runs/aux-20261016-1405/games --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk
from decklight.demos.dataset import record_config
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.engine.inverse import MSG_NAMES, choices_for_response, decision_groups


def deck_path(name: str) -> Path:
    for d in ("pool", "gauntlet", "newcards"):
        p = ROOT / "decks" / d / f"{name}.ydk"
        if p.is_file():
            return p
    sys.exit(f"deck {name} not found in decks/pool|gauntlet|newcards")


def config_of(rec: dict) -> DuelConfig:
    if "config" in rec:
        return record_config(rec)
    d0, d1 = rec["decks"]
    return DuelConfig(load_ydk(deck_path(d0)), load_ydk(deck_path(d1)), seed=rec["seed"])


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True, help="games jsonl file or directory (recursive)")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    files = sorted(args.records.rglob("*.jsonl")) if args.records.is_dir() else [args.records]
    recs = []
    for f in files:
        if not f.stat().st_size:
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
                if len(recs) >= args.n:
                    break
        if len(recs) >= args.n:
            break
    print(f"{len(recs)} records from {len(files)} files", flush=True)

    engine = Engine()
    counts: Counter = Counter()
    auto_counts: Counter = Counter()
    nones: Counter = Counter()
    mismatches: list[str] = []  # real mismatches (bytes differ, replay/duel failures)
    none_lines: list[str] = []
    alternates = skipped = replay_fails = 0
    max_nodes = 0
    ms: list[float] = []
    t0 = time.time()
    for gi, rec in enumerate(recs):
        gid = rec.get("game_id", rec.get("line_id", f"game {gi}"))
        if rec.get("truncated"):
            skipped += 1
            continue
        cfg = config_of(rec)
        auto = rec.get("auto_single", True)
        choices = [int(c) for c in rec["choices"]]
        try:
            groups = decision_groups(engine, cfg, choices, auto)
        except Exception as e:  # noqa: BLE001 - report and keep going
            replay_fails += 1
            mismatches.append(f"{gid}: replay failed: {f'{e!r}'[:200]}")
            continue
        if sum(len(sl) for _, sl, _ in groups) != len(choices):
            mismatches.append(f"{gid}: groups consume {sum(len(sl) for _, sl, _ in groups)} of {len(choices)} choices")
            continue
        env = PyEnv(engine, cfg, auto)
        env.reset()
        k = 0
        for di, (t, sl, resp) in enumerate(groups):
            tname = MSG_NAMES.get(t, hex(t))
            if not sl:
                auto_counts[tname] += 1
                continue  # decided entirely by auto-single steps: no index sequence to invert
            counts[tname] += 1
            if env.dec is None or env.dec.msg.type != t:
                got = None if env.dec is None else MSG_NAMES.get(env.dec.msg.type)
                mismatches.append(f"{gid} decision {di}: replay at {got}, record at {tname}")
                break
            st: dict = {}
            seq = choices_for_response(env, resp, st)
            ms.append(st["ms"])
            max_nodes = max(max_nodes, st.get("nodes", 0))
            if seq is None:
                nones[tname] += 1
                none_lines.append(f"{gid} decision {di} ({tname}, k={k}): None for {resp.hex()}")
            elif list(seq) != sl:
                probe = PyEnv.replay(engine, cfg, choices[:k], auto)
                n_before = len(probe.duel.responses)
                for i in seq:
                    probe.step(i)
                got_resp = probe.duel.responses[n_before] if len(probe.duel.responses) > n_before else None
                if got_resp == resp:
                    alternates += 1
                else:
                    mismatches.append(
                        f"{gid} decision {di} ({tname}, k={k}): record {sl} vs inverse {list(seq)} "
                        f"(bytes differ: want {resp.hex()}, "
                        f"got {None if got_resp is None else got_resp.hex()})"
                    )
            for i in sl:
                env.step(i)
            k += len(sl)
        if [r for _, _, r in groups] != list(env.duel.responses):
            mismatches.append(f"{gid}: incremental replay responses differ from grouped replay")
        if gi % 20 == 0:
            print(f"  {gi}/{len(recs)} games, {sum(counts.values())} decisions", flush=True)
    dt = time.time() - t0

    print(f"{'message':28s} {'n':>7s} {'auto':>7s} {'none':>6s}", flush=True)
    for t in sorted(set(counts) | set(auto_counts)):
        print(f"{t:28s} {counts.get(t, 0):7d} {auto_counts.get(t, 0):7d} {nones.get(t, 0):6d}", flush=True)
    print(
        f"games {len(recs)} decisions {sum(counts.values())} auto {sum(auto_counts.values())} "
        f"none {sum(nones.values())} alternate {alternates} mismatch {len(mismatches)} "
        f"skipped {skipped} replay_fails {replay_fails}",
        flush=True,
    )
    print(f"max_nodes {max_nodes} call_ms_p50 {pct(ms, 0.5):.2f} p99 {pct(ms, 0.99):.2f} ({dt:.0f}s total)",
          flush=True)  # fmt: skip
    for m in (none_lines + mismatches)[:20]:
        print(f"MISMATCH {m}", flush=True)
    if nones or mismatches:
        sys.exit(2)


if __name__ == "__main__":
    main()
