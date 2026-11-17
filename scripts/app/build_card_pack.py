"""Build a versioned card pack (A5): cdb rows, Lua scripts, strings, embeddings and lua_desc rows.

A pack is a zip with `manifest.json` (format `decklight.cardpack/1`, file sizes and sha256) and the
files the Rust loader mounts as an extra card dir. The weekly job pulls CardScripts/BabelCDB, diffs
against the previous baseline, and writes only the new/changed cards.

Examples:
  # first snapshot of the current official cards
  uv run python scripts/app/build_card_pack.py --write-baseline data/cache/app/a5/baseline.json

  # weekly pack (after --fetch): only codes/scripts changed since the baseline
  uv run python scripts/app/build_card_pack.py --fetch --baseline data/cache/app/a5/baseline.json \
      --id cards-2026-11-08 --write-baseline data/cache/app/a5/baseline.json

  # a pack from a custom card repo (the A5 end-to-end test)
  uv run python scripts/app/build_card_pack.py --source-dir data/cache/custom/zedja \
      --codes 270000111,270000315 --id zedja-smoke --fake-emb --out /tmp/zedja.zip

Nothing is uploaded or published; the caller copies the zip to the host that serves it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPTS = ROOT / "third_party" / "CardScripts"
DEFAULT_CDB = ROOT / "third_party" / "BabelCDB"
DEFAULT_EMB = ROOT / "data" / "cards" / "emb" / "qwen3-0.6b-d512.npz"
DEFAULT_OUT = ROOT / "data" / "cache" / "app" / "a5"
FORMAT = "decklight.cardpack/1"
DATAS_COLS = ("id", "ot", "alias", "setcode", "type", "atk", "def", "level", "race", "attribute", "category")
TEXTS_COLS = ("id", "name", "desc", *(f"str{i}" for i in range(1, 17)))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_hash(row: tuple) -> str:
    return hashlib.sha256(repr(row).encode("utf-8")).hexdigest()


def cdbs_for(cdb_dir: Path, source_dir: Path | None) -> list[Path]:
    if source_dir is not None:
        return sorted(source_dir.glob("*.cdb"))
    out = [cdb_dir / "cards.cdb"]
    out += sorted(cdb_dir.glob("release-*.cdb"))
    out += [p for p in sorted(cdb_dir.glob("prerelease-*.cdb")) if "rush" not in p.name]
    return [p for p in out if p.is_file()]


def read_rows(cdbs: list[Path]) -> dict[int, tuple[tuple, tuple]]:
    rows: dict[int, tuple[tuple, tuple]] = {}
    for path in cdbs:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for d in con.execute(f"SELECT {', '.join(DATAS_COLS)} FROM datas"):
                rows[int(d[0])] = (tuple(d), ())
            for t in con.execute(f"SELECT {', '.join(TEXTS_COLS)} FROM texts"):
                code = int(t[0])
                if code in rows:
                    rows[code] = (rows[code][0], tuple(t))
        finally:
            con.close()
    return rows


def script_index(scripts_dir: Path, source_dir: Path | None) -> dict[str, Path]:
    dirs = [scripts_dir, scripts_dir / "official", scripts_dir / "pre-release"]
    if source_dir is not None:
        dirs.insert(0, source_dir / "script")
    out: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("c*.lua")):
            out.setdefault(p.name, p)
    return out


def snapshot(cdbs: list[Path], scripts: dict[str, Path], strings: Path | None) -> dict:
    rows = read_rows(cdbs)
    return {
        "cards": {str(c): row_hash((d, t)) for c, (d, t) in sorted(rows.items())},
        "scripts": {n: sha256(p) for n, p in sorted(scripts.items())},
        "strings": sha256(strings) if strings and strings.is_file() else None,
    }


def choose_codes(rows: dict[int, tuple[tuple, tuple]], baseline: dict | None, codes: list[int] | None) -> list[int]:
    if codes is not None:
        missing = [c for c in codes if c not in rows]
        if missing:
            raise SystemExit(f"ALERT build_card_pack: codes not in the source cdb: {missing}")
        return sorted(set(codes))
    if baseline is not None:
        old = baseline.get("cards", {})
        return sorted(c for c, (d, t) in rows.items() if old.get(str(c)) != row_hash((d, t)))
    return sorted(rows)


def choose_scripts(scripts: dict[str, Path], baseline: dict | None, codes: list[int], explicit: list[int] | None) -> list[str]:
    wanted = {f"c{c}.lua" for c in codes}
    if explicit is not None:
        return sorted(n for n in scripts if n in wanted)
    if baseline is not None:
        old = baseline.get("scripts", {})
        return sorted(n for n, p in scripts.items() if old.get(n) != sha256(p))
    return sorted(scripts)


def write_cdb(path: Path, rows: dict[int, tuple[tuple, tuple]], codes: list[int]) -> int:
    con = sqlite3.connect(path)
    try:
        con.execute(f"CREATE TABLE datas ({', '.join(c + ' INT' for c in DATAS_COLS)})")
        text_types = ["INT"] + ["TEXT"] * (len(TEXTS_COLS) - 1)
        con.execute(f"CREATE TABLE texts ({', '.join(c + ' ' + t for c, t in zip(TEXTS_COLS, text_types))})")
        n = 0
        for code in codes:
            d, t = rows[code]
            con.execute(f"INSERT INTO datas VALUES ({', '.join('?' * len(DATAS_COLS))})", d)
            if t:
                con.execute(f"INSERT INTO texts VALUES ({', '.join('?' * len(TEXTS_COLS))})", t)
                n += 1
        con.commit()
    finally:
        con.close()
    return n


def fake_encoder(dim: int):
    def encode(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            v = rng.normal(size=dim)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float16)
    return encode


def build_lua_desc(script_dir: Path, out: Path) -> int:
    from decklight.cards import effect_strings as ES

    if not any(script_dir.glob("c*.lua")):
        return 0
    keys, feats = ES.lua_desc_table(scripts_dir=script_dir, extra_dirs=[])
    if not len(keys):
        return 0
    np.savez_compressed(out, keys=keys, feats=feats, cols=np.array(ES.LUA_COLS))
    return len(keys)


def build_pack(
    *,
    out_zip: Path,
    pack_id: str,
    cdb_dir: Path = DEFAULT_CDB,
    scripts_dir: Path = DEFAULT_SCRIPTS,
    source_dir: Path | None = None,
    baseline: dict | None = None,
    codes: list[int] | None = None,
    base_emb: Path | None = None,
    dim: int = 512,
    fake_emb: bool = False,
    no_emb: bool = False,
    strings: Path | None = None,
    write_baseline: Path | None = None,
) -> dict:
    cdbs = cdbs_for(cdb_dir, source_dir)
    if not cdbs:
        raise SystemExit(f"ALERT build_card_pack: no cdb files under {source_dir or cdb_dir}")
    rows = read_rows(cdbs)
    scripts = script_index(scripts_dir, source_dir)
    if strings is None and source_dir is not None and (source_dir / "strings.conf").is_file():
        strings = source_dir / "strings.conf"

    selected = choose_codes(rows, baseline, codes)
    selected_scripts = choose_scripts(scripts, baseline, selected, codes)
    script_codes = {
        int(n[1:-4]) for n in selected_scripts
        if n.startswith("c") and n.endswith(".lua") and n[1:-4].isdigit()
    }
    selected = sorted(set(selected) | {c for c in script_codes if c in rows})
    strings_changed = (
        baseline is not None and strings is not None and baseline.get("strings") != sha256(strings)
    )
    if not selected and not selected_scripts and not strings_changed:
        print("no new or changed cards; nothing to pack", flush=True)
        if write_baseline is not None:
            write_baseline.parent.mkdir(parents=True, exist_ok=True)
            write_baseline.write_text(json.dumps(snapshot(cdbs, scripts, strings), indent=1, sort_keys=True))
        return {"format": FORMAT, "id": pack_id, "codes": [], "files": [], "empty": True}

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cardpack-", dir=str(out_zip.parent)) as td:
        work = Path(td)
        n_texts = write_cdb(work / "cards.cdb", rows, selected)
        if selected and not n_texts:
            raise SystemExit("ALERT build_card_pack: selected cards have no texts rows")

        script_dir = work / "script"
        script_dir.mkdir()
        for name in selected_scripts:
            shutil.copy2(scripts[name], script_dir / name)

        if strings is not None and Path(strings).is_file():
            shutil.copy2(strings, work / "strings.conf")

        if not no_emb and selected:
            from decklight.cards import embed as E

            old_env = os.environ.get("YGO_EXTRA_CARDS")
            if source_dir is not None:
                os.environ["YGO_EXTRA_CARDS"] = str(source_dir)
            try:
                E.build(
                    dim=dim,
                    codes=selected,
                    tag="emb",
                    out_dir=work,
                    encode=fake_encoder(dim) if fake_emb else None,
                )
            finally:
                if source_dir is not None:
                    if old_env is None:
                        os.environ.pop("YGO_EXTRA_CARDS", None)
                    else:
                        os.environ["YGO_EXTRA_CARDS"] = old_env
            emb = work / "emb.npz"
            z = np.load(emb)
            if int(z["dim"]) != dim or not np.isfinite(z["card"]).all() or (np.abs(z["card"]).sum(1) == 0).any():
                raise SystemExit("ALERT build_card_pack: non-finite, zero or misshaped card vectors")
            if base_emb is not None:
                base = np.load(base_emb)
                if int(base["dim"]) != dim:
                    raise SystemExit(f"ALERT build_card_pack: base embedding dim {int(base['dim'])} != {dim}")

        n_lua = build_lua_desc(script_dir, work / "lua_desc.npz")

        files = []
        for p in sorted(work.rglob("*")):
            if p.is_file():
                files.append({
                    "path": p.relative_to(work).as_posix(),
                    "size": p.stat().st_size,
                    "sha256": sha256(p),
                })
        counts = {
            "cards": len(selected),
            "scripts": len(selected_scripts),
            "strings": 1 if (work / "strings.conf").is_file() else 0,
            "embeddings": len(selected) if (work / "emb.npz").is_file() else 0,
            "lua_desc": n_lua,
        }
        manifest = {
            "format": FORMAT,
            "id": pack_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base": {
                "cdb": str(cdb_dir),
                "scripts": str(scripts_dir),
                "source": str(source_dir) if source_dir else None,
                "versions": {
                    "cdb": git_rev(cdb_dir),
                    "scripts": git_rev(scripts_dir),
                    "format": FORMAT,
                },
            },
            "counts": counts,
            "codes": selected,
            "files": files,
        }
        (work / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(work.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(work).as_posix())

    if write_baseline is not None:
        write_baseline.parent.mkdir(parents=True, exist_ok=True)
        write_baseline.write_text(json.dumps(snapshot(cdbs, scripts, strings), indent=1, sort_keys=True))
    return manifest


def git_rev(path: Path) -> str | None:
    import subprocess

    r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def verify_pack(zip_path: Path, games: int) -> None:
    """Load the pack in the Python engine and play `games` random duels for every new card."""
    import random
    import tempfile

    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig, Engine
    from decklight.engine.env import PyEnv
    from decklight.engine.resources import default_cdbs, index_scripts, load_cards

    with tempfile.TemporaryDirectory(prefix="cardpack-verify-") as td:
        pack_dir = Path(td)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(pack_dir)
        cards = load_cards(default_cdbs(extra_dirs=[pack_dir]))
        scripts = index_scripts()
        for p in sorted((pack_dir / "script").glob("*.lua")):
            scripts[p.name] = p
        engine = Engine(cards=cards, scripts=scripts)
        con = sqlite3.connect(f"file:{pack_dir / 'cards.cdb'}?mode=ro", uri=True)
        try:
            rows = {int(r[0]): int(r[1]) for r in con.execute("SELECT id, type FROM datas")}
        finally:
            con.close()
        codes = sorted(rows)
        if not codes:
            return
        missing = [c for c in codes if c not in engine.cards]
        if missing:
            raise SystemExit(f"ALERT build_card_pack: pack codes missing from the engine: {missing}")
        opp = load_ydk(ROOT / "decks/pool/blue-eyes.ydk")
        extra_bits = 0x40 | 0x2000 | 0x800000 | 0x4000000
        filler = 89631139
        errors: list[tuple[int, str]] = []
        for code in codes:
            for g in range(games):
                class Deck:
                    if rows[code] & extra_bits:
                        main = [filler] * 40
                        extra = [code]
                    else:
                        main = [code] * 40
                        extra: list[int] = []
                    side: list[int] = []
                env = PyEnv(engine, DuelConfig(Deck(), opp, seed=91000 + g))
                obs = env.reset()
                rng = random.Random(91000 + g)
                steps = 0
                while obs is not None and steps < 3000:
                    obs = env.step(rng.randrange(int(obs["action_mask"].sum())))
                    steps += 1
                for t, msg in env.duel.log:
                    if t == 0:
                        errors.append((code, msg))
                env.duel.close()
        if errors:
            raise SystemExit(f"ALERT build_card_pack: {len(errors)} Lua script errors: {errors[:3]}")


def fetch_repo(path: Path) -> None:
    if not (path / ".git").exists():
        raise SystemExit(f"ALERT build_card_pack: {path} is not a git checkout")
    import subprocess

    r = subprocess.run(["git", "-C", str(path), "pull", "--ff-only"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ALERT build_card_pack: git pull in {path} failed: {r.stderr.strip()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a versioned decklight card pack")
    ap.add_argument("--out", type=Path, default=None, help=f"zip path or directory (default {DEFAULT_OUT})")
    ap.add_argument("--id", default=None, help="pack id (default cards-YYYYMMDD-HHMMSS)")
    ap.add_argument("--cdb-dir", type=Path, default=DEFAULT_CDB)
    ap.add_argument("--scripts-dir", type=Path, default=DEFAULT_SCRIPTS)
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="extra card dir (*.cdb, script/, strings.conf); used instead of cdb-dir")
    ap.add_argument("--codes", default=None, help="explicit comma-separated card codes")
    ap.add_argument("--baseline", type=Path, default=None, help="baseline JSON from a previous run")
    ap.add_argument("--write-baseline", type=Path, default=None, help="write the source snapshot here")
    ap.add_argument("--base-emb", type=Path, default=DEFAULT_EMB)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--fake-emb", action="store_true", help="random unit vectors (tests only)")
    ap.add_argument("--no-emb", action="store_true", help="omit embeddings")
    ap.add_argument("--strings", type=Path, default=None, help="strings.conf to include")
    ap.add_argument("--fetch", action="store_true", help="git pull the card script and cdb checkouts first")
    ap.add_argument("--verify", action="store_true", help="load the pack in the Python engine and play random games")
    ap.add_argument("--verify-games", type=int, default=1)
    a = ap.parse_args(argv)

    if a.fetch:
        fetch_repo(a.scripts_dir)
        if a.source_dir is None:
            fetch_repo(a.cdb_dir)
    baseline = json.loads(a.baseline.read_text()) if a.baseline else None
    codes = None
    if a.codes:
        codes = [int(x) for x in a.codes.replace(" ", ",").split(",") if x]
    pack_id = a.id or time.strftime("cards-%Y%m%d-%H%M%S")
    out = a.out or DEFAULT_OUT
    if out.suffix.lower() != ".zip":
        out = out / f"{pack_id}.zip"
    manifest = build_pack(
        out_zip=out,
        pack_id=pack_id,
        cdb_dir=a.cdb_dir,
        scripts_dir=a.scripts_dir,
        source_dir=a.source_dir,
        baseline=baseline,
        codes=codes,
        base_emb=a.base_emb,
        dim=a.dim,
        fake_emb=a.fake_emb,
        no_emb=a.no_emb,
        strings=a.strings,
        write_baseline=a.write_baseline,
    )
    if a.verify and not manifest.get("empty"):
        verify_pack(out, a.verify_games)
    print(json.dumps({"zip": str(out), "id": manifest.get("id"), "counts": manifest.get("counts", {}),
                      "empty": manifest.get("empty", False), "sha256": sha256(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
