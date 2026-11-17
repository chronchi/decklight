#!/usr/bin/env python3
"""Deck-bound opponents for `train_rl.py --deck-opps` (tsoj PPO step 3, arm B; NOTEBOOK 2026-11-07).

    uv run python scripts/tsoj/deck_opps.py [--out data/cache/deck_opps]

Collects the 16 archetype specialists (X4b: data/cache/x4b_spec/<arch>/best.pt on decks/x4b/train/<arch>/*.ydk;
Forge F0 and the r0 -> r1 round: runs/<job>/ft-<arch>/policy/best.pt on the lists of its FINETUNE.json), copies each
checkpoint to <out>/<arch>/best.pt (so a vast job ships one directory) and writes <out>/manifest.json:
[{name, ckpt, decks, source}]. Exit 3 when a checkpoint or list is missing or a count differs from 16 / 3.

    uv run python scripts/tsoj/deck_opps.py --packs data/forge/c1/packs --drop sky-striker --want 25 \
        --out data/cache/c1_league/opps

Forge-pack mode (cycle-1 league, prereg amendment 6): one specialist per pack in --packs, its weights read in place
(policy/weights.safetensors, metrics.jsonl beside: full history) on the pack's decks/train lists; --drop names the
specialists G1 rejected. Writes only the manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORGE_JOBS = ("runs/forge-f0-20261029-1408", "runs/forge-round-20261029-2133")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/cache/deck_opps")
    ap.add_argument("--packs", type=Path, default=None, help="forge-pack mode: a directory of packs")
    ap.add_argument("--drop", default="", help="--packs: comma-separated archetypes to leave out")
    ap.add_argument("--want", type=int, default=16, help="expected specialist count")
    a = ap.parse_args()
    if a.packs is not None:
        return from_packs(a)
    specs: list[dict] = []
    for d in sorted((ROOT / "data/cache/x4b_spec").iterdir()):
        if d.is_dir():
            decks = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "decks/x4b/train" / d.name).glob("*.ydk"))
            specs.append({"name": d.name, "src": d / "best.pt", "decks": decks, "source": "x4b"})
    for job in FORGE_JOBS:
        for d in sorted((ROOT / job).glob("ft-*")):
            if d.is_dir():
                ft = json.loads((d / "FINETUNE.json").read_text())
                specs.append({"name": d.name[3:], "src": d / "policy/best.pt", "decks": list(ft["decks"]),
                              "source": job})
    bad = [f"{s['name']}: {len(s['decks'])} lists" for s in specs if len(s["decks"]) != 3]
    bad += [f"{s['name']}: no {s['src']}" for s in specs if not s["src"].is_file()]
    bad += [f"{s['name']}: no {p}" for s in specs for p in s["decks"] if not (ROOT / p).is_file()]
    names = [s["name"] for s in specs]
    if len(specs) != a.want or len(set(names)) != len(names):
        bad.append(f"{len(specs)} specialists ({len(set(names))} names), want {a.want}")
    for b in bad:
        print(f"ALERT deck_opps: {b}", flush=True)
    if bad:
        return 3
    a.out.mkdir(parents=True, exist_ok=True)
    out = []
    for s in specs:
        dst = a.out / s["name"] / "best.pt"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy(s["src"], dst)
        out.append({"name": s["name"], "ckpt": str(dst.relative_to(ROOT)), "decks": s["decks"],
                    "source": s["source"]})
    (a.out / "manifest.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.out / 'manifest.json'}: {len(out)} specialists, {sum(len(s['decks']) for s in out)} lists", flush=True)
    return 0


def from_packs(a: argparse.Namespace) -> int:
    drop = {x for x in a.drop.split(",") if x}
    packs = sorted(d for d in (ROOT / a.packs).iterdir() if (d / "pack.json").is_file())
    unknown = drop - {d.name for d in packs}
    out, bad = [], [f"--drop {x}: no such pack" for x in sorted(unknown)]
    for d in packs:
        if d.name in drop:
            continue
        w, decks = d / "policy/weights.safetensors", sorted((d / "decks/train").glob("*.ydk"))
        if not w.is_file():  # older (f3) packs ship best.pt; train_rl's load_opp reads either
            w = d / "policy/best.pt"
        if not (w.is_file() and (d / "policy/metrics.jsonl").is_file()):
            bad.append(f"{d.name}: no weights or metrics.jsonl")
        if len(decks) != 3:
            bad.append(f"{d.name}: {len(decks)} train lists")
        out.append({"name": d.name, "ckpt": str(w.relative_to(ROOT)), "decks": [str(p.relative_to(ROOT)) for p in decks],
                    "source": json.loads((d / "pack.json").read_text())["pack_id"]})
    if len(out) != a.want:
        bad.append(f"{len(out)} specialists, want {a.want}")
    for b in bad:
        print(f"ALERT deck_opps: {b}", flush=True)
    if bad:
        return 3
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "manifest.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.out / 'manifest.json'}: {len(out)} specialists, {sum(len(s['decks']) for s in out)} lists", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
