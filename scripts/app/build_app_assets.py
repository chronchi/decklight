"""Build the Android app's data archives (A6): everything `decklight-server` reads, laid out as in the repo.

The app embeds one archive per unit (each tree below, plus one for FILES) into the app library and unpacks only the
units whose content hash changed (rust/app/src-tauri/src/lib.rs). CardScripts is shipped as one `DLSPACK1` file
(`third_party/CardScripts.pack`) rather than ~23k files, so the first launch writes one file and the engine reads the
few hundred scripts a duel needs from it. `assets.json` lists the units in a fixed order (`web` first, so the client
can be served while the rest unpacks), each with its archive and content id.
`ASSETS.txt` in every archive lists the paths the app checks after unpacking. Card images are never included
(ADR 0010 / the plan's "Card images").

    uv run python scripts/app/build_app_assets.py [--out data/cache/app/apk]
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import struct
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (unit name, tree). The first unit is always the client itself: the gate serves it before the rest is ready.
TREES = [
    ("web", "web"),
    ("babelcdb", "third_party/BabelCDB"),
    ("cardscripts", "third_party/CardScripts"),
    ("onnx", "data/cache/onnx/y"),
    ("decks-pool", ""),  # curated: manifest + starters + every deck the phone references (see curated_decks)
    ("decks-gauntlet", "decks/gauntlet"),
    ("primers", "data/primers/studied"),
]
FILES_UNIT = "card-data"
FILES = [
    "config/app/prebuilt_decks.txt",
    "third_party/edopro-strings.conf",
    "data/cards/emb/qwen3-0.6b-d512.npz",
    "src/decklight/web/winchance_calib.json",
    "src/decklight/web/deck_index.json",
    "src/decklight/web/deck_coverage.json",
    "src/decklight/web/content_sources.json",
    "src/decklight/web/puzzles.json",
    "src/decklight/web/primers.json.gz",
    # Sample games (Ask replay/bot_view and the games list; never the user's history): replay.rs writes_sample_games.
    "config/samples/games/sample-1.match.json.gz",
    "config/samples/games/sample-2.match.json.gz",
    "config/samples/games/sample-3.match.json.gz",
]
# Checked by the app after unpacking (a missing one is a startup error, not a half-working app).
REQUIRED = [
    "web/index.html",
    "third_party/BabelCDB/cards.cdb",
    "third_party/CardScripts.pack",
    "data/cache/onnx/y/manifest.json",
    "data/cache/onnx/y/policy.onnx",
    "data/cache/onnx/y/value.onnx",
    "decks/pool",
] + FILES
SKIP_PARTS = {".git", "__pycache__", "ci", ".github"}


def die(msg: str) -> None:
    print(f"ALERT build_app_assets: {msg}", file=sys.stderr)
    sys.exit(2)


def files_of(rel: str) -> list[Path]:
    p = ROOT / rel
    if not p.is_dir():
        die(f"missing {rel}")
    return sorted(f for f in p.rglob("*") if f.is_file() and not SKIP_PARTS & set(f.relative_to(ROOT).parts))


PACK_MAGIC = b"DLSPACK1"


def build_scripts_pack(tree: str, out: Path) -> tuple[bytes, int]:
    """CardScripts -> one `DLSPACK1` file: magic, u32 count, {u16 name_len, name, u64 offset, u32 len}, bytes.
    Entries are sorted by their path relative to the tree; offsets are absolute in the pack."""
    base = ROOT / tree
    if not base.is_dir():
        die(f"missing {tree}")
    entries = []
    for p in files_of(tree):
        rel = p.relative_to(base).as_posix()
        entries.append((rel.encode(), p.read_bytes()))
    entries.sort(key=lambda e: e[0])
    header = 8 + 4 + sum(2 + len(name) + 8 + 4 for name, _ in entries)
    parts = [PACK_MAGIC, struct.pack("<I", len(entries))]
    offset = header
    for name, data in entries:
        parts.append(struct.pack("<H", len(name)))
        parts.append(name)
        parts.append(struct.pack("<QI", offset, len(data)))
        offset += len(data)
    head = b"".join(parts)
    assert len(head) == header
    pack = head + b"".join(data for _, data in entries)
    out.write_bytes(pack)
    return pack, len(entries)


def archive_entries(entries: list[tuple[str, bytes]]) -> tuple[bytes, int]:
    """One deterministic .tar.gz for these (repo-relative name, bytes) pairs, plus ASSETS.txt."""
    buf = io.BytesIO()
    # Fixed mtime/owner and gzip mtime 0: the same inputs give the same archive and id.
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for rel, data in entries:
                ti = tarfile.TarInfo(rel)
                ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
                tar.addfile(ti, io.BytesIO(data))
            need = ("\n".join(REQUIRED) + "\n").encode()
            ti = tarfile.TarInfo("ASSETS.txt")
            ti.size, ti.mtime, ti.mode = len(need), 0, 0o644
            tar.addfile(ti, io.BytesIO(need))
    return buf.getvalue(), len(entries) + 1


def archive_of(paths: list[Path]) -> tuple[bytes, int]:
    """One deterministic .tar.gz for these repo-relative files, plus ASSETS.txt. Returns (blob, entries)."""
    entries = [(p.relative_to(ROOT).as_posix(), p.read_bytes()) for p in paths]
    return archive_entries(entries)


STARTER_DECKS = ("blue-eyes", "branded")
PREBUILT = ROOT / "config/app/prebuilt_decks.txt"


def read_prebuilt() -> list[str]:
    if not PREBUILT.is_file():
        die(f"missing {PREBUILT.relative_to(ROOT)}")
    stems = [
        line.strip()
        for line in PREBUILT.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not stems:
        die(f"{PREBUILT.relative_to(ROOT)} lists no decks")
    return stems


def puzzle_stems() -> set[str]:
    path = ROOT / "src/decklight/web/puzzles.json"
    try:
        puzzles = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        die(f"puzzles.json unreadable ({e})")
    return {d for p in puzzles for d in p.get("decks", []) if isinstance(d, str)}


def curated_decks() -> tuple[list[Path], dict[str, str]]:
    """The pool/newcards files the phone uses: the prebuilt manifest, the starters, and every puzzle deck.
    Gauntlet and primer decks stay in their own units, so only their absent pool/newcards copies are skipped."""
    reasons: dict[str, str] = {}
    for stem in read_prebuilt():
        reasons.setdefault(stem, "manifest")
    for stem in STARTER_DECKS:
        reasons.setdefault(stem, "starter")
    for stem in sorted(puzzle_stems()):
        reasons.setdefault(stem, "puzzle")
    paths: list[Path] = []
    kept: dict[str, str] = {}
    missing: list[str] = []
    for stem, why in reasons.items():
        for sub in ("pool", "newcards"):
            p = ROOT / "decks" / sub / f"{stem}.ydk"
            if p.is_file():
                paths.append(p)
                kept[stem] = why
                break
        else:
            in_other = (ROOT / "decks/gauntlet" / f"{stem}.ydk").is_file() or (
                ROOT / "data/primers/studied/decks" / f"{stem}.ydk"
            ).is_file()
            if not in_other:
                missing.append(stem)
    if missing:
        die(f"deck stems not in decks/pool|newcards|gauntlet|primers: {sorted(missing)}")
    paths.sort(key=lambda p: p.relative_to(ROOT).as_posix())
    return paths, kept


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "unit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/cache/app/apk"))
    ap.add_argument("--list", action="store_true", help="print the curated deck files and exit")
    a = ap.parse_args()
    out = Path(a.out)

    curated, kept = curated_decks()
    why = {w: sum(1 for v in kept.values() if v == w) for w in ("manifest", "starter", "puzzle")}
    pool_n = sum(1 for p in curated if p.parent.name == "pool")
    print(
        f"decks-pool: {len(curated)} curated files ({pool_n} pool, {len(curated) - pool_n} newcards; "
        f"manifest {why['manifest']}, starters {why['starter']}, puzzles {why['puzzle']})"
    )
    for stem in sorted(kept):
        if kept[stem] != "manifest":
            print(f"  keep {stem}: {kept[stem]}")
    if a.list:
        return

    out.mkdir(parents=True, exist_ok=True)
    # CardScripts ships as one pack: the engine reads the few hundred scripts a duel needs straight from it,
    # instead of the app writing ~23k files on first launch (NOTEBOOK 2026-11-12).
    pack, pack_count = build_scripts_pack("third_party/CardScripts", out / "CardScripts.pack")
    print(f"cardscripts.pack: {pack_count} entries, {len(pack) / 1e6:.1f} MB")

    units = []
    for name, t in TREES:
        if name == "cardscripts":
            units.append((name, [], ["third_party/CardScripts", "third_party/CardScripts.pack"]))
        elif name == "decks-pool":
            units.append((name, [], ["decks/pool"]))
        else:
            units.append((name, [t], [t]))
    files: list[Path] = []
    for f in FILES:
        if not (ROOT / f).is_file():
            die(f"missing {f}")
        files.append(ROOT / f)
    units.append((FILES_UNIT, [], FILES))

    manifest = []
    total_gz = 0
    for name, trees, removes in units:
        if name == "cardscripts":
            blob, entries = archive_entries([("third_party/CardScripts.pack", pack)])
            # The unit id is the pack's own hash (the archive also holds ASSETS.txt).
            sha = hashlib.sha256(pack).hexdigest()
        elif name == "decks-pool":
            blob, entries = archive_of(curated)
            sha = hashlib.sha256(blob).hexdigest()
        else:
            paths: list[Path] = []
            for t in trees:
                paths += files_of(t)
            if not trees:
                paths = list(files)
            if not paths:
                die(f"unit {name} has no files")
            if name == "web" and not any(p.name == "index.html" for p in paths):
                die("web unit has no index.html")
            blob, entries = archive_of(paths)
            sha = hashlib.sha256(blob).hexdigest()
        stem = safe_name(name)
        (out / f"{stem}.tar.gz").write_bytes(blob)
        (out / f"{stem}.id").write_text(sha[:16] + "\n")
        total_gz += len(blob)
        manifest.append(
            {"name": name, "archive": f"{stem}.tar.gz", "id": sha[:16], "entries": entries, "remove": removes}
        )
        print(f"{name}: {entries} entries, {len(blob) / 1e6:.1f} MB gz, id {sha[:16]}")

    # The global id stays for the server's art-sync token (.assets_id, read by decklight-server); the app's own
    # staleness decisions use the per-unit ids above.
    overall = hashlib.sha256("".join(f"{u['name']}:{u['id']}\n" for u in manifest).encode()).hexdigest()[:16]
    (out / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "assets.id").write_text(overall + "\n")
    for stale in ("assets.tar.gz",):
        (out / stale).unlink(missing_ok=True)
    print(
        f"assets: {len(manifest)} units, {sum(u['entries'] for u in manifest)} entries, "
        f"{total_gz / 1e6:.1f} MB gz total, id {overall}"
    )


if __name__ == "__main__":
    main()
