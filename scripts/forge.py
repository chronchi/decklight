"""Forge packs (docs/plans/2026-10-29-deck-forge.md, "Protocol: the pack").

    uv run python scripts/forge.py pack --ft runs/<job>/ft-<arch> --release data/cache/x4b_D/best.pt \\
        --release-name r0 --archetype <a> --train-decks decks/forge/f0/train/<a>/*.ydk \\
        --heldout-decks decks/forge/f0/heldout/<a>-*.ydk --out data/forge/packs/<a>
    uv run python scripts/forge.py validate <pack dir or .tar.gz> [--release data/cache/x4b_D/best.pt]
    uv run python scripts/forge.py validate <pack> --fingerprint src/decklight/forge/fingerprints/r1.json
    uv run python scripts/forge.py fingerprint <release .pt|.safetensors> --name r1 --out <json>
    uv run python scripts/forge.py credits <pack dir>...
    uv run python scripts/forge.py archive <pack dir> [--out-dir .]   # -> <pack_id>.tar.gz to submit
    uv run python scripts/forge.py migrate <v1 pack dir> --out <dir> [--license CC-BY-4.0]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.forge.pack import archive, build_pack, extract, fingerprint_of, migrate_pack, validate_pack

DEFAULT_EMB = ROOT / "data" / "cards" / "emb" / "qwen3-0.6b-d512.npz"


def _default_cards_db() -> list[Path]:
    """The card DBs the engine loads (official + release sets, no custom extras): the pack's cards fingerprint."""
    from decklight.engine.resources import default_cdbs

    return default_cdbs(extra_dirs=[])


def _core() -> str:
    try:
        r = subprocess.run(["git", "-C", str(ROOT / "third_party" / "ygopro-core"), "rev-parse",
                            "--short", "HEAD"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _cmd_pack(args: argparse.Namespace) -> int:
    ft = Path(args.ft)
    policy = ft / "policy" / "best.pt"
    train_metrics = ft / "rl" / "metrics.jsonl"
    finetune_json = ft / "FINETUNE.json"
    for p in (policy, train_metrics, finetune_json):
        if not p.is_file():
            print(f"forge pack: {p} missing", file=sys.stderr)
            return 1
    for p in [Path(args.release), *args.train_decks]:
        if not Path(p).is_file():
            print(f"forge pack: {p} missing", file=sys.stderr)
            return 1
    info = json.loads(finetune_json.read_text(encoding="utf-8"))
    try:
        seed = int(info.get("seed", 0))
    except (TypeError, ValueError):
        seed = 0
    try:
        iterations = int(info.get("iterations", 0))
    except (TypeError, ValueError):
        print("forge pack: FINETUNE.json has no integer 'iterations'", file=sys.stderr)
        return 1
    train = {"config": info, "seed": seed, "iterations": iterations, "stop": "fixed"}
    _ = args.sources  # accepted; sources stay empty in v1 (optional per brief)
    out = build_pack(
        ft_policy=policy, train_metrics=train_metrics, release=Path(args.release),
        release_name=args.release_name, archetype=args.archetype,
        train_decks=[Path(p) for p in args.train_decks],
        heldout_decks=[Path(p) for p in (args.heldout_decks or [])], out=Path(args.out),
        contributor={"name": args.contributor}, license=args.license, train=train, sources=None,
        compute=None, card=None, emb=DEFAULT_EMB, cards_db=_default_cards_db(), core=_core(),
    )
    pack_id = json.loads((out / "pack.json").read_text(encoding="utf-8"))["pack_id"]
    print(f"forge pack: wrote {out} ({pack_id})")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    target = Path(args.pack)
    if target.suffixes[-2:] == [".tar", ".gz"] or target.suffix == ".tgz":
        tmp = Path(tempfile.mkdtemp(prefix="forge-validate-"))
        try:
            target = extract(target, tmp)
        except ValueError as e:
            print(f"ERROR {e}")
            return 2
    cards_db = [Path(p) for p in args.cards_db] if args.cards_db else _default_cards_db()
    errors, warnings = validate_pack(target, Path(args.release) if args.release else None,
                                     emb=DEFAULT_EMB, cards_db=cards_db,
                                     fingerprint=args.fingerprint)
    for w in warnings:
        print(f"WARN {w}")
    for e in errors:
        print(f"ERROR {e}")
    if not errors:
        print("OK")
        return 0
    return 2


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    try:
        fp = fingerprint_of(args.release, args.name)
    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"forge fingerprint: {e}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=1) + "\n", encoding="utf-8")
    print(f"forge fingerprint: wrote {out} ({fp['release']}, {len(fp['tensors'])} tensors)")
    return 0


def _cmd_credits(args: argparse.Namespace) -> int:
    """One release-notes line per pack, sorted by archetype; exit 2 if any pack is invalid."""
    rows: list[tuple[str, str, str]] = []
    failed = False
    for raw in args.packs:
        pack = Path(raw)
        errors, _warnings = validate_pack(pack)
        if errors:
            print(f"forge credits: {pack} is not a valid pack:", file=sys.stderr)
            for e in errors:
                print(f"  ERROR {e}", file=sys.stderr)
            failed = True
            continue
        try:
            manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
            rows.append((manifest["archetype"], manifest["contributor"]["name"], manifest["pack_id"]))
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"forge credits: {pack} manifest unreadable: {e}", file=sys.stderr)
            failed = True
    if failed:
        return 2
    for archetype, name, pack_id in sorted(rows):
        print(f"- {archetype}: pack by {name}, CC BY 4.0 ({pack_id})")
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    errors, _warnings = validate_pack(Path(args.pack))
    if errors:
        for e in errors:
            print(f"ERROR {e}")
        return 2
    print(archive(Path(args.pack), Path(args.out_dir)))
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    try:
        out = migrate_pack(args.pack, args.out, license=args.license)
    except (ValueError, FileExistsError, FileNotFoundError, OSError) as e:
        print(f"forge migrate: {e}", file=sys.stderr)
        return 1
    cards_db = [Path(p) for p in args.cards_db] if args.cards_db else _default_cards_db()
    errors, warnings = validate_pack(out, Path(args.release) if args.release else None,
                                     emb=DEFAULT_EMB, cards_db=cards_db)
    for w in warnings:
        print(f"WARN {w}")
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        return 2
    print(f"forge migrate: wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="build a pack from a finetune_deck output")
    p.add_argument("--ft", required=True, help="finetune_deck output dir")
    p.add_argument("--release", required=True, help="release best.pt the fine-tune started from")
    p.add_argument("--release-name", required=True)
    p.add_argument("--archetype", required=True)
    p.add_argument("--train-decks", nargs="+", required=True)
    p.add_argument("--heldout-decks", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--contributor", default="ygo-zero")
    p.add_argument("--license", default="CC-BY-4.0")
    p.add_argument("--sources", default=None)
    p.set_defaults(func=_cmd_pack)
    v = sub.add_parser("validate", help="validate a pack dir or .tar.gz")
    v.add_argument("pack")
    v.add_argument("--release", default=None)
    v.add_argument("--cards-db", nargs="*", default=None)
    v.add_argument("--fingerprint", default=None)
    v.set_defaults(func=_cmd_validate)
    f = sub.add_parser("fingerprint", help="write a release fingerprint JSON for offline pack checks")
    f.add_argument("release", help="release best.pt or .safetensors")
    f.add_argument("--name", required=True, help="release name recorded in the fingerprint")
    f.add_argument("--out", required=True, help="output JSON path")
    f.set_defaults(func=_cmd_fingerprint)
    c = sub.add_parser("credits", help="print one release-notes line per pack dir")
    c.add_argument("packs", nargs="+", help="pack directories")
    c.set_defaults(func=_cmd_credits)
    a = sub.add_parser("archive", help="write <pack_id>.tar.gz (the file to submit) from a valid pack dir")
    a.add_argument("pack")
    a.add_argument("--out-dir", default=".")
    a.set_defaults(func=_cmd_archive)
    m = sub.add_parser("migrate", help="migrate a retired v1 pack dir to v2")
    m.add_argument("pack")
    m.add_argument("--out", required=True)
    m.add_argument("--license", default="CC-BY-4.0")
    m.add_argument("--release", default=None)
    m.add_argument("--cards-db", nargs="*", default=None)
    m.set_defaults(func=_cmd_migrate)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
