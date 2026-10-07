"""Build DAGS mid-game starts from recorded games (RL v2 step 3).

For each replayable record under --src, emit cut points: every decision index k >= 1 at a
turn boundary (turns[k] != turns[k-1]) plus 2 random cuts per game. Each output line holds
final (shuffled) deck orders + duel seed/hand/draw/lp/setup plus the prefix choices[:k].

Each game is replayed once with CppEnv; that single replay validates all of its cuts
(a prefix k is valid iff the replay is still alive after step k-1). Games whose either
deck equals a holdout deck are excluded (eval leakage). Exit 2 if a holdout deck slips
through the final self-check, or if fewer than --min-starts valid starts were emitted.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.engine.cpp_env import CppEnv  # noqa: E402
from ygo_zero.engine.duel import Engine  # noqa: E402
from ygo_zero.rl import dags  # noqa: E402

_ENGINE = None


def _init() -> None:
    global _ENGINE
    _ENGINE = Engine()


def _replay(task: tuple[int, dict, str]) -> tuple[int, int, str]:
    """Replay one game; return (idx, first_dead_step, error). Prefix k is valid iff k <= first_dead."""
    idx, rec, _src = task
    try:
        cfg = dags.build_config(rec)
        ch = [int(a) for a in rec["choices"]]
        assert _ENGINE is not None
        env = CppEnv(_ENGINE, cfg, auto_single=bool(rec.get("auto_single", True)))
        if env.reset() is None:
            return idx, 0, "ended-before-first-decision"
        for j, a in enumerate(ch):
            try:
                ob = env.step(a)
            except Exception as e:  # noqa: BLE001 - any core rejection invalidates later cuts
                return idx, j, f"step-{j}-{type(e).__name__}"
            if ob is None:
                return idx, j, ""  # duel ended at step j (normal for the last step)
        return idx, len(ch), ""
    except Exception as e:  # noqa: BLE001 - unbuildable configs invalidate the whole game
        return idx, 0, f"config-{type(e).__name__}-{e}"


def _holdout_names() -> set[str]:
    return set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="dirs or globs of demo jsonl files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=4000, help="games replayed (shuffled sample)")
    ap.add_argument("--min-starts", type=int, default=500, help="exit 2 below this many valid starts")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    holdout = dags.holdout_keys(ROOT / "decks/pool", _holdout_names())
    print(f"[dags] {len(holdout)} holdout decks", flush=True)

    files: list[str] = []
    for s in args.src:
        files += glob.glob(s, recursive=True) if glob.has_magic(s) else (
            [str(p) for p in Path(s).rglob("*.jsonl")] if Path(s).is_dir() else [s])
    files = sorted(set(files))
    print(f"[dags] {len(files)} files", flush=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(files)).tolist() if files else []
    tasks: list[tuple[int, dict, str]] = []
    stats: dict[str, Counter] = {}
    skipped_holdout = skipped_auto = skipped_bad = 0
    for fi in order:
        f = files[fi]
        src = Path(f).parent.name
        st = stats.setdefault(src, Counter())
        try:
            lines = Path(f).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if len(tasks) >= args.max_games:
                break
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                ch = rec.get("choices")
                cfg = rec.get("config")
                if not ch or cfg is None or "deck0_main" not in cfg:
                    skipped_bad += 1
                    st["skipped_bad"] += 1
                    continue
                if not bool(rec.get("auto_single", True)):
                    skipped_auto += 1
                    st["skipped_auto"] += 1
                    continue
                if dags.is_holdout_game(rec, holdout):
                    skipped_holdout += 1
                    st["excluded_holdout"] += 1
                    continue
            except (ValueError, KeyError, TypeError):
                skipped_bad += 1
                st["skipped_bad"] += 1
                continue
            st["games"] += 1
            tasks.append((len(tasks), rec, src))
        if len(tasks) >= args.max_games:
            break
    print(f"[dags] {len(tasks)} games sampled "
          f"(excluded-holdout {skipped_holdout}, skipped chaining {skipped_auto}, bad {skipped_bad})",
          flush=True)
    if not tasks:
        print("[dags] no games sampled", flush=True)
        raise SystemExit(2)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init) as pool:
        results = pool.map(_replay, tasks, chunksize=8)
    by_idx = {i: (fd, err) for i, fd, err in results}

    crng = np.random.default_rng(args.seed + 0xC07)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with open(args.out, "w") as fh:
        for idx, rec, src in tasks:
            first_dead, err = by_idx[idx]
            st = stats[src]
            n = len(rec["choices"])
            cuts = dags.cut_points(n, rec.get("turns"), crng)
            st["cuts"] += len(cuts)
            for k in cuts:
                if k <= first_dead:
                    fh.write(json.dumps(dags.start_record(rec, k, src)) + "\n")
                    emitted += 1
                    st["emitted"] += 1
                else:
                    st["dropped"] += 1
    for src in sorted(stats):
        st = stats[src]
        print(f"[dags] src={src} games={st['games']} cuts={st['cuts']} "
              f"emitted={st['emitted']} dropped={st['dropped']}", flush=True)
    print(f"[dags] {emitted} valid starts -> {args.out}", flush=True)

    # Final self-check: no holdout deck may slip through (eval leakage), fail loud.
    bad = 0
    for line in args.out.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k0 = dags.deck_key(r["main0"], r["extra0"])
        k1 = dags.deck_key(r["main1"], r["extra1"])
        if k0 in holdout or k1 in holdout:
            bad += 1
    if bad:
        print(f"[dags] HOLDOUT LEAKAGE: {bad} starts use a holdout deck", flush=True)
        raise SystemExit(2)
    if emitted < args.min_starts:
        print(f"[dags] only {emitted} valid starts (< {args.min_starts})", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
