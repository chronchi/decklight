"""Combo-line starts for step 2 (docs/plans/2026-11-03-opp-turn-interaction.md; `src/decklight/rl/lines.py`).

- Turn-1 lines: the best verified line per opening hand (`data/cache/combo_lines/ranked.jsonl`, `best_for_hand`) of
  each `admit_turn1_lines` deck in the admit list; each line gets `--variants` starts with our deck in seat 1 (drawn
  from the custom lists + train pool decks) and a sampled opening hand. Kept only if the line still reaches its turn-1
  field with our hand present (`lines.first_start`).
- Turn-2 lines: `plan_turns` going-second records (`--turns` globs, e.g. `data/demos/turns/second/*.jsonl` and the
  step-1b output): the opponent's verified turn 2 into a real board of ours.
Eval-only decks (holdout, newcards) and gauntlet decks never appear on either side (overfitting rule).

    uv run python scripts/build_line_starts.py --admit config/league/combo_pool_v2.json --decks branded --variants 1 \
        --out /tmp/line-starts.jsonl
    uv run python scripts/build_line_starts.py --admit config/league/combo_pool_v2.json \
        --turns 'data/demos/turns/second/*.jsonl' --out data/cache/line_starts/starts.jsonl

Exits 3 when no start was built, `--turns` gave no turn-2 start, or more than 20% of the attempted turn-1 or turn-2
starts fail to replay. The training file is rebalanced to `--second-share` turn-2 starts (the eval file is not).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import playtest as PT

from decklight.decks import load_ydk
from decklight.engine.duel import Engine
from decklight.model.bc_data import eval_only_decks
from decklight.rl import lines as L
from decklight.rl.dags import deck_key

CUSTOM = ("decks/custom/skydeck-tcg2026-10.ydk", "decks/custom/light-magistus-f10-tcg2026-10.ydk")


def is_eval(line_id: str, frac: float) -> bool:
    return zlib.crc32(line_id.encode()) % 1000 < frac * 1000


def eval_path(out: Path) -> Path:
    return out.with_suffix(".eval.jsonl")


def split_eval(out: Path, frac: float) -> int:
    """Move the eval lines' starts from `out` to `<out>.eval.jsonl`; returns how many moved."""
    rows = [x for x in out.read_text().splitlines() if x.strip()]
    ev = [x for x in rows if is_eval(json.loads(x)["line_id"], frac)]
    out.write_text("".join(x + "\n" for x in rows if not is_eval(json.loads(x)["line_id"], frac)))
    eval_path(out).write_text("".join(x + "\n" for x in ev))
    return len(ev)


def rebalance(out: Path, share: float, max_repeat: int) -> tuple[int, int, int]:
    """Training file only: make turn-2 starts (our set board, the opponent's turn) about `share` of the starts
    (plan step 2: "Episodes start either at turn 1 or (50%) at the start of an opponent turn with our set cards").
    Repeats each turn-2 start up to `max_repeat` times, then drops turn-1 starts (lowest crc32 of line_id/hand kept,
    so the choice is stable). Returns (turn-1 kept, turn-2 unique, repeat)."""
    rows = [x for x in out.read_text().splitlines() if x.strip()]
    first = [x for x in rows if json.loads(x)["kind"] == "first"]
    second = [x for x in rows if json.loads(x)["kind"] == "second"]
    if not second or share <= 0:
        return len(first), len(second), 1
    rep = max(1, min(max_repeat, math.ceil(share * len(first) / ((1 - share) * len(second)))))
    keep = round(len(second) * rep * (1 - share) / share)
    if keep < len(first):
        first = sorted(first, key=lambda x: zlib.crc32(x.encode()))[:keep]
    out.write_text("".join(x + "\n" for x in first + second * rep))
    return len(first), len(second), rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admit", type=Path, required=True)
    ap.add_argument("--lines", type=Path, default=ROOT / "data/cache/combo_lines/ranked.jsonl")
    ap.add_argument("--turns", nargs="*", default=[], help="plan_turns going-second record globs")
    ap.add_argument("--decks", default="", help="only these opponent decks (turn-1 lines)")
    ap.add_argument("--variants", type=int, default=2, help="starts per turn-1 line (our deck + hand)")
    ap.add_argument("--variant-from", type=int, default=0,
                    help="first variant index (fresh starts for lines already built with variants 0..N-1)")
    ap.add_argument("--eval-only", action="store_true", help="build only the eval split's turn-1 lines")
    ap.add_argument("--max-lines", type=int, default=12, help="turn-1 lines per deck (best-for-hand, most acts first)")
    ap.add_argument("--p-custom", type=float, default=0.5, help="share of turn-1 starts with a custom list as our deck")
    ap.add_argument("--gauntlet-decks", type=Path, default=ROOT / "decks/gauntlet")
    ap.add_argument("--eval-frac", type=float, default=0.1,
                    help="share of lines (by line_id hash, all variants together) written to <out>.eval.jsonl instead: "
                         "the step-2 gates play on lines training never saw")
    ap.add_argument("--second-share", type=float, default=0.5,
                    help="target share of turn-2 starts in the training file (rebalance(); 0 = keep as built)")
    ap.add_argument("--max-repeat", type=int, default=3, help="max copies of one turn-2 start when rebalancing")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    hold, new = eval_only_decks()
    banned = hold | new
    g_keys = {deck_key(d.main, d.extra) for d in (load_ydk(p) for p in a.gauntlet_decks.glob("*.ydk"))}
    pool = {p.stem: load_ydk(p) for p in sorted((ROOT / "decks/pool").glob("*.ydk"))}
    ours_pool = [(n, d) for n, d in pool.items() if n not in banned and deck_key(d.main, d.extra) not in g_keys]
    custom = [(Path(c).stem, load_ydk(ROOT / c)) for c in CUSTOM]
    admit = json.loads(a.admit.read_text())["decks"]
    want = {d["deck"] for d in admit if d.get("admit_turn1_lines")}
    if a.decks:
        want &= set(a.decks.split(","))
    skipped = sorted(d for d in want if d in banned or d not in pool or deck_key(pool[d].main, pool[d].extra) in g_keys)
    want -= set(skipped)
    best = defaultdict(list)
    for raw in a.lines.read_text().splitlines():
        r = json.loads(raw)
        if r["best_for_hand"] and r["deck"] in want:
            best[r["deck"]].append(r)
    engine = Engine()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n1 = n1_fail = n2 = n2_fail = 0
    kinds = Counter()
    with a.out.open("w") as fo:
        for deck in sorted(best):
            recs = {r["line_id"]: r for r in PT.verified_records(deck, limit=1_000_000)}
            rows = sorted(best[deck], key=lambda r: (-r["acts"], r["line_id"]))[: a.max_lines]
            got = 0
            for row in rows:
                rec = recs.get(row["line_id"])
                if rec is None or (a.eval_only and not is_eval(row["line_id"], a.eval_frac)):
                    continue
                for v in range(a.variant_from, a.variants):
                    seed = zlib.crc32(f"{row['line_id']}/{v}".encode())
                    use_custom = (seed % 1000) / 1000 < a.p_custom
                    on, od = (custom if use_custom else ours_pool)[seed % (len(custom) if use_custom else len(ours_pool))]
                    try:
                        st = L.first_start(engine, rec, od, on, seed)
                    except Exception as e:  # noqa: BLE001 - counted; the rate check below fails the job if many
                        print(f"  {row['line_id']} / {on}: {type(e).__name__}: {e}"[:200], flush=True)
                        st = None
                    if st is None or not st["script"]:
                        n1_fail += 1
                        continue
                    fo.write(json.dumps(st) + "\n")
                    n1 += 1
                    got += 1
                    kinds["first"] += 1
            print(f"{deck:44s} {got} turn-1 starts from {len(rows)} lines", flush=True)
        files = sorted({f for g in a.turns for f in glob.glob(g)})
        for f in files:
            for raw in Path(f).read_text().splitlines():
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                if any(d in banned for d in rec.get("decks", [])) or rec.get("bucket") != "second":
                    continue
                c = rec["config"]
                if {deck_key(c["deck0_main"], c["deck0_extra"]), deck_key(c["deck1_main"], c["deck1_extra"])} & g_keys:
                    continue
                try:
                    st = L.second_start(engine, rec)
                except Exception as e:  # noqa: BLE001
                    print(f"  {rec.get('line_id')}: {type(e).__name__}: {e}"[:200], flush=True)
                    n2_fail += 1
                    continue
                if not st["script"]:
                    n2_fail += 1
                    continue
                fo.write(json.dumps(st) + "\n")
                n2 += 1
                kinds["second"] += 1
    n_eval = split_eval(a.out, a.eval_frac)
    print(f"held out {n_eval} starts (eval split) -> {eval_path(a.out)}")
    k1, u2, rep = rebalance(a.out, a.second_share, a.max_repeat)
    print(f"training file: {k1} turn-1 starts + {u2} turn-2 starts x{rep} "
          f"(turn-2 share {u2 * rep / max(k1 + u2 * rep, 1):.2f}, target {a.second_share})")
    print(f"skipped decks (eval-only / gauntlet / no pool file): {skipped}")
    print(f"{n1} turn-1 starts ({n1_fail} failed), {n2} turn-2 starts ({n2_fail} failed) from {len(files)} files "
          f"-> {a.out}", flush=True)
    if n1 + n2 == 0:
        print("ALERT build_line_starts: no start built", flush=True)
        return 3
    if a.turns and n2 == 0:
        print(f"ALERT build_line_starts: --turns given but 0 turn-2 starts ({len(files)} files)", flush=True)
        return 3
    if n2 + n2_fail and n2_fail / (n2 + n2_fail) > 0.2:
        print(f"ALERT build_line_starts: {n2_fail}/{n2 + n2_fail} turn-2 starts failed to replay", flush=True)
        return 3
    if n1 + n1_fail and n1_fail / (n1 + n1_fail) > 0.2:
        print(f"ALERT build_line_starts: {n1_fail}/{n1 + n1_fail} turn-1 starts failed to replay", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
