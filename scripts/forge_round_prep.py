"""Forge round prep (docs/plans/2026-10-29-deck-forge.md, Pipeline steps 2-5).

Laptop and box; pure Python, no torch. Subcommands:

    pairs --pack <pack dir> --opponents <opponents.txt> --out <file>
    anchor --packs <dir>... --out <dir>
    klmap --packs <dir>... --default <release best.pt> --out kl_map.json
    seeds --pairs <file> --games <G>
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_lists

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "decks" / "pool"
HOLDOUT = ROOT / "decks" / "pool" / "HOLDOUT"
X4B = ROOT / "decks" / "x4b"
FORGE = ROOT / "decks" / "forge"


def pack_decks(pack: Path) -> list[dict]:
    return json.loads((pack / "pack.json").read_text())["decks"]


def cmd_pairs(args: argparse.Namespace) -> int:
    info = json.loads((Path(args.pack) / "pack.json").read_text())
    arch = info["archetype"]
    stems = [d["stem"] for d in info["decks"]]
    opps = [l.strip() for l in Path(args.opponents).read_text().splitlines() if l.strip()]
    usable = forge_lists.opponents_for(arch, opps)
    lines = [f"{d}:{o}\n" for d in stems for o in usable]
    if not lines:
        print(f"forge_round_prep pairs: 0 pairs for {args.pack} (no decks or no opponents)")
        return 2
    Path(args.out).write_text("".join(lines))
    print(f"pairs {len(lines)}")
    return 0


def cmd_anchor(args: argparse.Namespace) -> int:
    hold = set(HOLDOUT.read_text().split()) | {"kewl-tune"}
    x4b_scored = {p.stem for p in X4B.rglob("*.ydk") if X4B / "train" not in p.parents}
    forge_stems = {p.stem for p in FORGE.rglob("*.ydk")}
    pack_stems = {d["stem"] for pack in args.packs for d in pack_decks(Path(pack))}
    excluded = hold | x4b_scored | forge_stems | pack_stems
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    for p in sorted(POOL.glob("*.ydk")):
        if p.stem in excluded:
            continue
        shutil.copy(p, out / p.name)
        written.add(p.stem)
    for p in sorted((X4B / "train").rglob("*.ydk")):
        shutil.copy(p, out / p.name)
        written.add(p.stem)
    n = len(written)
    print(f"anchor {n}")
    if n < 100:
        print(f"forge_round_prep anchor: only {n} anchor decks")
        return 2
    return 0


def cmd_klmap(args: argparse.Namespace) -> int:
    decks: dict[str, str] = {}
    for pack in args.packs:
        info = json.loads((Path(pack) / "pack.json").read_text())
        for d in info["decks"]:
            if d["role"] != "train":
                continue
            if d["stem"] in decks and decks[d["stem"]] != f"{pack}/policy/best.pt":
                print(f"forge_round_prep klmap: stem {d['stem']} maps to two packs")
                return 2
            decks[d["stem"]] = f"{pack}/policy/best.pt"
    Path(args.out).write_text(json.dumps({"default": args.default, "decks": decks}, indent=1) + "\n")
    print(f"klmap {len(decks)}")
    return 0


def cmd_seeds(args: argparse.Namespace) -> int:
    n = sum(1 for l in Path(args.pairs).read_text().splitlines() if l.strip())
    if n == 0:
        print("forge_round_prep seeds: no pairs")
        return 2
    print(math.ceil(args.games / (2 * n)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--pack", required=True)
    p.add_argument("--opponents", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_pairs)
    q = sub.add_parser("anchor")
    q.add_argument("--packs", nargs="+", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(func=cmd_anchor)
    r = sub.add_parser("klmap")
    r.add_argument("--packs", nargs="+", required=True)
    r.add_argument("--default", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_klmap)
    s = sub.add_parser("seeds")
    s.add_argument("--pairs", required=True)
    s.add_argument("--games", type=int, required=True)
    s.set_defaults(func=cmd_seeds)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
