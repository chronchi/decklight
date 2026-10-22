"""The per-deck paired card: our bot always pilots the target deck, ygo-agent pilots each other gauntlet deck.

    uv run python scripts/deck_card.py --deck decks/gauntlet/BlueEyes.ydk --arm base=data/cache/P/best.pt:data/cache/value_p2/best.pt \\
        --arm ft=runs/ft-x/policy/best.pt:runs/ft-x/value/best.pt --search 0|1 --seeds 8 --out runs/card-x [--nproc N --proc i] [--summarize]

Deals: our side plays the target deck in both seats against each other gauntlet deck (30 opponents),
--seeds seeds each starting at --deal-seed0. Every arm plays the same deals via scripts/search_match.py
(--pairs, --seed-per-pair, --raw, --opp ygoagent): --search 0 is the raw policy, --search 1 the product
search settings from scripts/product.sh. --summarize writes <out>/card.json (per-arm win rate with Wilson
95% CI, paired difference vs the first arm with SE and 95% CI, going-first/second splits).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from paired_sm import games as paired_games

from decklight.decks import Deck, load_ydk
from decklight.eval.match import wilson


def _product_search() -> list[str]:
    """The product's search flags, read from scripts/product.sh (the single definition)."""
    for line in (ROOT / "scripts" / "product.sh").read_text().splitlines():
        if line.startswith("PRODUCT_SEARCH="):
            return line.split("=", 1)[1].strip().strip('"').split()
    raise SystemExit("ALERT deck_card: PRODUCT_SEARCH not found in scripts/product.sh")


PRODUCT_SEARCH = _product_search()


def fail(msg: str) -> NoReturn:
    print(f"ALERT deck_card: {msg}", flush=True)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_content(a: Deck, b: Deck) -> bool:
    return tuple(sorted(a.main)) == tuple(sorted(b.main)) and tuple(sorted(a.extra)) == tuple(sorted(b.extra))


def parse_arm(spec: str) -> tuple[str, Path, Path | None]:
    """name=ckpt:value (value may be empty = no value net)."""
    if "=" not in spec:
        fail(f"--arm {spec!r}: want name=ckpt:value")
    name, rest = spec.split("=", 1)
    if ":" not in rest:
        fail(f"--arm {spec!r}: want name=ckpt:value")
    ckpt_s, value_s = rest.rsplit(":", 1)
    name = name.strip()
    if not name or not ckpt_s.strip():
        fail(f"--arm {spec!r}: want name=ckpt:value")
    return name, Path(ckpt_s), (Path(value_s) if value_s.strip() else None)


def find_target(target: Deck, gauntlet: list[tuple[str, Deck]]) -> tuple[str, list[str]]:
    """The gauntlet stem matching the target deck's content + the opponent stems (content-identical excluded)."""
    hits = [stem for stem, d in gauntlet if same_content(d, target)]
    if not hits:
        fail("target deck matches no deck in the gauntlet dir (the card needs the target to be one of them)")
    opps = [stem for stem, d in gauntlet if not same_content(d, target)]
    return hits[0], opps


def build_deals(target_stem: str, opponents: list[str], seeds: int, seed0: int) -> list[dict]:
    """The deal list in search_match schedule order: per pair, per seed, both seats (S always plays target)."""
    deals = []
    for k, opp in enumerate(opponents):
        for s in range(seed0 + k * seeds, seed0 + (k + 1) * seeds):
            for seat in (0, 1):
                deals.append({"target": target_stem, "opp": opp, "seed": s, "seat": seat})
    return deals


def pairs_spec(target_stem: str, opponents: list[str]) -> str:
    return ",".join(f"{target_stem}:{o}" for o in opponents)


def shard(deals: list[dict], nproc: int, proc: int) -> list[dict]:
    return deals[proc::nproc]


def build_arm_cmd(search_match: Path, ckpt: Path, value: Path | None, search: int, pairs: str,
                 gauntlet_dir: Path, seeds: int, seed0: int, nproc: int, proc: int,
                 out_arm: Path, device: str) -> list[str]:
    """Pure command builder for one arm's search_match shard (tested: must contain --device <device>)."""
    cmd = [sys.executable, str(search_match), "--ckpt", str(ckpt), "--opp", "ygoagent",
           "--gauntlet-decks", str(gauntlet_dir), "--pairs", pairs,
           "--seed-per-pair", "1", "--seeds", str(seeds), "--deal-seed0", str(seed0),
           "--nproc", str(nproc), "--proc", str(proc), "--device", device,
           "--out", str(out_arm)]  # fmt: skip
    if search == 1:
        if value is not None:
            cmd += ["--value", str(value)]
        cmd += PRODUCT_SEARCH
    else:
        cmd += ["--raw", "1"]
    return cmd


def run_arm(search_match: Path, ckpt: Path, value: Path | None, search: int, pairs: str,
            gauntlet_dir: Path, seeds: int, seed0: int, nproc: int, proc: int,
            out_arm: Path, overwrite: bool, expected: int, device: str = "cpu") -> None:
    """Play one arm's shard with search_match (reuse the games file when it already holds the shard)."""
    out_arm.mkdir(parents=True, exist_ok=True)
    path = out_arm / f"games-{proc}.jsonl"
    if path.is_file() and not overwrite:
        have = sum(1 for _ in path.read_text().splitlines() if _.strip())
        if have == expected:
            print(f"[{out_arm.name}] reuse {path} ({have} games)", flush=True)
            return
        print(f"[{out_arm.name}] {path} has {have} games, want {expected}: replaying", flush=True)
    cmd = build_arm_cmd(search_match, ckpt, value, search, pairs, gauntlet_dir,
                        seeds, seed0, nproc, proc, out_arm, device)
    print(f"[{' '.join(cmd)}]", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        fail(f"search_match for arm {out_arm.name} exited {r.returncode}")


def summarize(out: Path, arms: list[tuple[str, Path, Path | None]]) -> dict:
    """Pair arms by (decks, seed, s_seat) and write <out>/card.json. Fails loud on 0 games,
    differing deal sets, identical checkpoints or NaN."""
    per: dict[str, dict] = {}
    for name, _, _ in arms:
        d = paired_games(out / name)
        if not d:
            fail(f"arm {name}: 0 games in {out / name}/games-*.jsonl")
        per[name] = d
    first = arms[0][0]
    keys = set(per[first])
    for name, d in per.items():
        if set(d) != keys:
            fail(f"arm {name} deals differ from arm {first} "
                 f"({len(set(d) & keys)}/{len(keys)} pair up)")
    # identical checkpoints (ckpt sha256, and value sha256 with None == None)
    shas = {}
    for name, ckpt, value in arms:
        for p, label in ((ckpt, "ckpt"),) + (((value, "value"),) if value is not None else ()):
            if not p.is_file():
                fail(f"arm {name}: {label} {p} missing")
        shas[name] = (sha256_file(ckpt), sha256_file(value) if value is not None else None)
    seen = {}
    for name, s in shas.items():
        if s in seen:
            fail(f"arms {seen[s]} and {name} have identical checkpoints (sha256)")
        seen[s] = name
    for name, d in per.items():
        bad = [k for k, v in d.items() if not math.isfinite(v)]
        if bad:
            fail(f"arm {name}: {len(bad)} NaN scores")

    def stats(scores: list[float]) -> dict:
        n = len(scores)
        wr = sum(scores) / n
        lo, hi = wilson(sum(scores), n)
        return {"win_rate": round(wr, 3), "n": n, "ci": [round(lo, 3), round(hi, 3)]}

    def paired(a: list[float], b: list[float]) -> dict:
        n = len(a)
        diffs = [y - x for x, y in zip(a, b, strict=True)]
        m = sum(diffs) / n
        se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1) / n) if n > 1 else 0.0
        for v in (m, se):
            if not math.isfinite(v):
                fail("NaN in a paired difference")
        return {"diff_pp": round(m * 100, 2), "se_pp": round(se * 100, 2),
                "ci_pp": [round((m - 1.96 * se) * 100, 2), round((m + 1.96 * se) * 100, 2)], "n": n}

    by_name = {name: (ckpt, value) for name, ckpt, value in arms}
    card: dict = {"arms": {}, "diff_vs_first": {"first": first}, "seats": {}}
    for name, d in per.items():
        card["arms"][name] = stats(list(d.values()))
        ckpt, value = by_name[name]
        card["arms"][name].update({"ckpt": str(ckpt), "value": str(value) if value is not None else None})
        if name != first:
            ks = sorted(keys)
            card["diff_vs_first"][name] = paired([per[first][k] for k in ks], [d[k] for k in ks])
    for label, seat in (("first", 0), ("second", 1)):
        card["seats"][label] = {name: stats([v for k, v in d.items() if k[2] == seat])
                                for name, d in per.items()}
    (out / "card.json").write_text(json.dumps(card, indent=1))
    for name in per:
        s = card["arms"][name]
        print(f"{name}: win rate {s['win_rate']} [{s['ci'][0]}, {s['ci'][1]}] n={s['n']}", flush=True)
    for name in per:
        if name == first:
            continue
        p = card["diff_vs_first"][name]
        print(f"{name} - {first}: {p['diff_pp']:+.2f} pp [{p['ci_pp'][0]:+.2f}, {p['ci_pp'][1]:+.2f}] "
              f"(SE {p['se_pp']:.2f}) n={p['n']}", flush=True)
    for label in ("first", "second"):
        line = " | ".join(f"{name} {card['seats'][label][name]['win_rate']} "
                          f"(n={card['seats'][label][name]['n']})" for name in per)
        print(f"{label}: {line}", flush=True)
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, required=True, help="target .ydk file, anywhere")
    ap.add_argument("--arm", action="append", default=[], metavar="name=ckpt:value",
                    help="repeatable; value may be empty (no value net)")
    ap.add_argument("--search", type=int, choices=(0, 1), default=0,
                    help="0 = raw policy (search_match --raw), 1 = product search settings")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--deal-seed0", type=int, default=9000)
    ap.add_argument("--gauntlet-decks", type=Path, default=ROOT / "decks" / "gauntlet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--summarize", action="store_true", help="write <out>/card.json after playing")
    ap.add_argument("--overwrite", action="store_true", help="replay even when the games file exists")
    ap.add_argument("--device", default=None, help="cuda (default when available) | cpu")
    ap.add_argument("--dry-run", action="store_true", help="print the deal list size + first 5 deals, play nothing")
    args = ap.parse_args()

    try:
        target = load_ydk(args.deck)
    except OSError as e:
        fail(f"--deck {args.deck}: cannot read: {e}")
    if not target.main:
        fail(f"--deck {args.deck}: no main-deck cards (unparseable deck)")
    gfiles = sorted(args.gauntlet_decks.glob("*.ydk"))
    if len(gfiles) < 2:
        fail(f"only {len(gfiles)} gauntlet decks in {args.gauntlet_decks}")
    gauntlet = [(p.stem, load_ydk(p)) for p in gfiles]
    target_stem, opps = find_target(target, gauntlet)
    if not args.arm:
        fail("no --arm given (at least one name=ckpt:value)")
    arms = [parse_arm(s) for s in args.arm]
    if len({a[0] for a in arms}) != len(arms):
        fail("duplicate --arm names")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    deals = build_deals(target_stem, opps, args.seeds, args.deal_seed0)
    print(f"deals: {len(deals)} ({len(opps)} opponents x 2 seats x {args.seeds} seeds), "
          f"S always pilots {target_stem}", flush=True)
    for d in deals[:5]:
        print(f"  opp={d['opp']} seed={d['seed']} seat={d['seat']}", flush=True)
    if args.dry_run:
        return
    mine = shard(deals, args.nproc, args.proc)
    print(f"proc {args.proc}/{args.nproc} runs {len(mine)} deals", flush=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pairs = pairs_spec(target_stem, opps)
    for name, ckpt, value in arms:
        run_arm(ROOT / "scripts" / "search_match.py", ckpt, value, args.search, pairs,
                args.gauntlet_decks, args.seeds, args.deal_seed0,
                args.nproc, args.proc, out / name, args.overwrite, len(mine), device)
    if args.summarize:
        summarize(out, arms)


if __name__ == "__main__":
    main()
