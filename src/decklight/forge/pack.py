"""Forge pack format v2: build, validate and archive a specialist fine-tune submission.

A pack is a directory (archived as ``<pack_id>.tar.gz``) holding the specialist's
policy weights plus the metadata the coordinator needs to verify and distill it.
See docs/plans/2026-10-29-deck-forge.md, section "Protocol: the pack".

v2 (``forge-pack/2``): weights ship as safetensors (pure data, never unpickled)
at ``policy/weights.safetensors``; packs are licensed CC-BY-4.0. v1 packs
(``policy/best.pt`` pickles) are retired: ``forge.py migrate`` converts them.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from decklight.decks.ydk import load_ydk
from decklight.engine.obs import OBS_VERSION
from decklight.model.ckpt import load_state

PROTOCOL = "forge-pack/2"
RETIRED = "forge-pack/1"

ARCH_DEFAULTS = {"d": 256, "layers": 4, "heads": 4, "v3": False, "hist": 0, "known": False}
ARCH_KEYS = ("d", "layers", "heads", "v3", "hist", "known")

MAX_MEMBERS = 200
MAX_UNCOMPRESSED = 64 * 1024 * 1024

# Bound on ||W - B|| / ||B|| (global L2 over floating tensors) for a pack's
# weights W against its base release B. The 11 F2 packs (1 h fine-tunes) measured 0.014-0.028 (NOTEBOOK
# 2026-10-30 "forge pack v2"); 0.10 leaves ~3.5x headroom for longer fine-tunes.
MAX_REL_DIST = 0.10

# Shell scripts interpolate these names: strict names are the command-injection guard.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
PACK_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}-[0-9a-f]{12}$")

CODE_RE = re.compile(r"^[0-9]{1,10}$")
# Reach release notes / issue comments: reject, never sanitise.
FORBIDDEN_TEXT = set("<>[]()`@")

SIZE_PACK_JSON = 256 * 1024
SIZE_CARD_JSON = 256 * 1024
SIZE_TRAIN_JSONL = 4 * 1024 * 1024
SIZE_POLICY_METRICS = 1024 * 1024
SIZE_YDK = 16 * 1024
SIZE_WEIGHTS = 32 * 1024 * 1024

WEIGHTS_REL = "policy/weights.safetensors"
METRICS_REL = "policy/metrics.jsonl"
ALLOWED_EXACT = {"pack.json", "card.json", "train.jsonl", WEIGHTS_REL, METRICS_REL}


def _is_allowed(rel: str) -> bool:
    if rel in ALLOWED_EXACT:
        return True
    parts = rel.split("/")
    return (
        len(parts) == 3
        and parts[0] == "decks"
        and parts[1] in ("train", "heldout")
        and parts[2].endswith(".ydk")
        and bool(NAME_RE.fullmatch(parts[2][:-4]))
    )


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deck_hash(path: Path) -> str:
    """Canonical deck hash: main/extra sorted codes, side ignored."""
    d = load_ydk(path)
    main = ",".join(str(c) for c in sorted(d.main))
    extra = ",".join(str(c) for c in sorted(d.extra))
    return hashlib.sha256(f"main:{main}|extra:{extra}".encode()).hexdigest()


def _read_args(metrics_path: Path) -> dict:
    try:
        for raw in metrics_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                rec = None
            if isinstance(rec, dict) and "args" in rec and isinstance(rec["args"], dict):
                return rec["args"]
    except OSError:
        pass
    return {}


def _arch_of(args: dict) -> dict:
    arch = dict(ARCH_DEFAULTS)
    for k in ARCH_KEYS:
        if k in args:
            arch[k] = args[k]
    arch["d"] = int(arch["d"])
    arch["layers"] = int(arch["layers"])
    arch["heads"] = int(arch["heads"])
    arch["hist"] = int(arch["hist"] or 0)
    arch["known"] = bool(arch["known"])
    arch["v3"] = bool(arch["v3"])
    return arch


def cards_db_hash(cards_db: Path | list[Path]) -> str:
    """sha256 of one card DB, or of the list the engine loads (default_cdbs(): base + release sets) as the sha256 of
    "<name>:<sha256>" lines in order. "" when no file exists."""
    paths = [Path(cards_db)] if isinstance(cards_db, (str, Path)) else [Path(p) for p in cards_db]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return ""
    if len(paths) == 1 and isinstance(cards_db, (str, Path)):
        return sha256_file(paths[0])
    return hashlib.sha256("".join(f"{p.name}:{sha256_file(p)}\n" for p in paths).encode()).hexdigest()


def _cards_db_ids(cards_db: Path | list[Path]) -> set[int] | None:
    """All card codes in the given DB(s), or None when no DB is usable (check skipped). Read-only."""
    paths = [Path(cards_db)] if isinstance(cards_db, (str, Path)) else [Path(p) for p in cards_db]
    ids: set[int] = set()
    for p in paths:
        if not p.is_file():
            continue
        try:
            empty = p.stat().st_size == 0
        except OSError:
            empty = True
        if empty:
            continue
        try:
            db = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                rows = db.execute("SELECT id FROM datas").fetchall()
            finally:
                db.close()
        except Exception:  # noqa: BLE001, S112 - an unreadable DB skips the code-exists check
            continue
        ids.update(int(r[0]) for r in rows)
    return ids or None


def compat_fingerprint(release_metrics: Path, emb: Path, cards_db: Path | list[Path], core: str = "") -> dict:
    args = _read_args(Path(release_metrics))
    emb_p = Path(emb)
    return {
        "obs_version": OBS_VERSION,
        "arch": _arch_of(args),
        "emb": emb_p.name,
        "emb_sha256": sha256_file(emb_p) if emb_p.is_file() else "",
        "cards_db_sha256": cards_db_hash(cards_db),
        "core": core,
    }


def fingerprint_of(release_path: str | Path, name: str) -> dict:
    """Small JSON-able description of a release checkpoint for offline pack checks.

    Lets the coordinator validate pack key sets, shapes and arch without moving
    the (large) release weights to the checking machine.
    """
    release_path = Path(release_path)
    state = load_state(release_path)
    return {
        "release": name,
        "sha256": sha256_file(release_path),
        "obs_version": OBS_VERSION,
        "arch": _arch_of(_read_args(release_path.parent / "metrics.jsonl")),
        "tensors": {k: list(v.shape) for k, v in state.items() if torch.is_tensor(v)},
    }


def _load_state_flat(path: Path) -> dict:
    # Trusted local files only (our own fine-tune / release .pt files), never stranger packs.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


def _pack_id(archetype: str, weights_path: Path) -> str:
    return f"{archetype}-{sha256_file(weights_path)[:12]}"


def _files_map(out: Path) -> dict:
    files = {}
    for p in sorted(f for f in out.rglob("*") if f.is_file()):
        rel = p.relative_to(out).as_posix()
        if rel == "pack.json":
            continue
        files[rel] = sha256_file(p)
    return files


def build_pack(*, ft_policy: Path, train_metrics: Path, release: Path, release_name: str, archetype: str,
               train_decks: list[Path], heldout_decks: list[Path], out: Path, contributor: dict,
               license: str, train: dict, sources: dict[str, dict] | None = None,
               compute: dict | None = None, card: Path | None = None, emb: Path,
               cards_db: Path | list[Path], core: str = "") -> Path:
    ft_policy, train_metrics, release = Path(ft_policy), Path(train_metrics), Path(release)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    if not train_decks:
        raise ValueError("need at least one train deck")
    if not (isinstance(contributor, dict) and isinstance(contributor.get("name"), str)
            and contributor["name"]):
        raise ValueError("contributor needs a 'name' string")
    release_metrics = release.parent / "metrics.jsonl"
    if not release_metrics.is_file():
        raise FileNotFoundError(f"{release_metrics} missing: cannot read the release architecture")

    out.mkdir(parents=True)
    (out / "policy").mkdir(parents=True)
    (out / "decks" / "train").mkdir(parents=True)
    (out / "decks" / "heldout").mkdir(parents=True)

    state = _load_state_flat(ft_policy)
    fp16 = {k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in state.items()}
    weights = out / "policy" / "weights.safetensors"
    save_file(fp16, weights)
    shutil.copyfile(release_metrics, out / "policy" / "metrics.jsonl")
    shutil.copyfile(train_metrics, out / "train.jsonl")

    decks: list[dict] = []
    for role, paths in (("train", train_decks), ("heldout", heldout_decks)):
        for p in paths:
            p = Path(p)
            dest = out / "decks" / role / f"{p.stem}.ydk"
            shutil.copyfile(p, dest)
            entry: dict = {"stem": p.stem, "role": role, "sha256": deck_hash(dest)}
            if sources and p.stem in sources:
                entry["source"] = sources[p.stem]
            decks.append(entry)
    if card is not None:
        shutil.copyfile(card, out / "card.json")

    pack_id = _pack_id(archetype, weights)
    manifest = {
        "protocol": PROTOCOL,
        "pack_id": pack_id,
        "created": datetime.now(UTC).isoformat(),
        "archetype": archetype,
        "contributor": contributor,
        "license": license,
        "base": {"release": release_name, "sha256": sha256_file(release)},
        "compat": compat_fingerprint(release_metrics, emb, cards_db, core),
        "decks": decks,
        "train": train,
        "files": _files_map(out),
    }
    if compute is not None:
        manifest["compute"] = compute
    (out / "pack.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return out


def migrate_pack(v1dir: str | Path, out: str | Path, license: str = "CC-BY-4.0") -> Path:
    """Convert our own v1 pack (retired pickle format) to v2. The v1 input is trusted (F0/F2 packs only)."""
    v1dir, out = Path(v1dir), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    manifest = json.loads((v1dir / "pack.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != RETIRED:
        raise ValueError(f"migrate needs a {RETIRED} pack, has {manifest.get('protocol')!r}")
    old_id = manifest.get("pack_id", "")
    state = torch.load(v1dir / "policy" / "best.pt", map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    fp16 = {k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in state.items()}

    out.mkdir(parents=True)
    (out / "policy").mkdir(parents=True)
    weights = out / WEIGHTS_REL
    save_file(fp16, weights)
    for rel in ("policy/metrics.jsonl", "train.jsonl", "card.json"):
        src = v1dir / rel
        if src.is_file():
            shutil.copyfile(src, out / rel)
    for e in manifest.get("decks", []):
        if not isinstance(e, dict) or e.get("role") not in ("train", "heldout"):
            continue
        src = v1dir / "decks" / e["role"] / f"{e['stem']}.ydk"
        dest = out / "decks" / e["role"] / f"{e['stem']}.ydk"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    new = {
        "protocol": PROTOCOL,
        "pack_id": _pack_id(manifest["archetype"], weights),
        "created": datetime.now(UTC).isoformat(),
        "archetype": manifest["archetype"],
        "contributor": manifest["contributor"],
        "license": license,
        "migrated_from": old_id,
        "base": manifest["base"],
        "compat": manifest["compat"],
        "decks": manifest["decks"],
        "train": manifest["train"],
        "files": _files_map(out),
    }
    if isinstance(manifest.get("compute"), dict):
        new["compute"] = manifest["compute"]
    (out / "pack.json").write_text(json.dumps(new, indent=1) + "\n", encoding="utf-8")
    return out


def _err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _parse_ydk_raw(text: str) -> tuple[dict[str, list[int]], list[str]]:
    """Data-only .ydk parse: section headers, `#` comment lines, and decimal card codes. Anything else is an error."""
    sections: dict[str, list[int]] = {"main": [], "extra": [], "side": []}
    cur = "main"
    bad: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s in ("#main", "#extra", "!side"):
            cur = s[1:]
            continue
        if s.startswith("#"):
            continue  # comment lines (#created by ..., #source ..., ...): ignored, as load_ydk does
        if CODE_RE.fullmatch(s):
            sections[cur].append(int(s))
        else:
            bad.append(f"line {lineno}: {s[:40]!r} is not a decimal card code")
    return sections, bad


def _check_text(errors: list[str], what: str, value: object, *, min_len: int, max_len: int) -> None:
    if not isinstance(value, str) or not value:
        return  # the required-key skeleton above already reported it
    if len(value) < min_len:
        _err(errors, f"contributor {what} is empty")
    elif len(value) > max_len:
        _err(errors, f"contributor {what} is longer than {max_len} characters")
    if isinstance(value, str) and value and not all(c.isprintable() and c not in "\x7f" for c in value):
        _err(errors, f"contributor {what} has non-printable or control characters")
    if isinstance(value, str) and any(c in FORBIDDEN_TEXT for c in value):
        _err(errors, f"contributor {what} has forbidden characters <>[]()`@ (reaches release notes)")


def validate_pack(pack: str | Path, release: str | Path | None = None, *, emb: str | Path | None = None,
                  cards_db: Path | list[Path] | None = None,
                  fingerprint: str | Path | None = None) -> tuple[list[str], list[str]]:
    """See :func:`_validate_pack`; any unexpected exception becomes one error (a crafted pack must not crash the
    checker)."""
    try:
        return _validate_pack(pack, release, emb=emb, cards_db=cards_db, fingerprint=fingerprint)
    except Exception as e:  # noqa: BLE001
        return ([f"validator failed on this pack: {type(e).__name__}"], [])


def _validate_pack(pack: str | Path, release: str | Path | None = None, *, emb: str | Path | None = None,
                   cards_db: Path | list[Path] | None = None,
                   fingerprint: str | Path | None = None) -> tuple[list[str], list[str]]:
    """Validate a v2 pack directory. Every failure is one short error string; nothing escapes, nothing executes.

    Cheap structural checks (names, allowed paths, sizes) run first; pack weights
    are never loaded before they pass. The reference for key sets / shapes is the
    ``release`` checkpoint (a ``.pt`` or ``.safetensors``, loaded with
    :func:`~decklight.model.ckpt.load_state`) or, when no release is at hand, a
    ``fingerprint`` JSON file (see :func:`fingerprint_of`).
    """
    errors: list[str] = []
    warnings: list[str] = []
    pack = Path(pack)
    manifest_path = pack / "pack.json"
    if not pack.is_dir():
        return ([f"pack {pack} is not a directory"], warnings)
    if not manifest_path.is_file():
        return ([f"{manifest_path} missing"], warnings)
    if manifest_path.stat().st_size > SIZE_PACK_JSON:
        return ([f"pack.json is {manifest_path.stat().st_size} bytes, over the {SIZE_PACK_JSON} limit"], warnings)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as e:
        return ([f"pack.json unparseable: {type(e).__name__}"], warnings)
    if not isinstance(manifest, dict):
        return (["pack.json must hold a JSON object"], warnings)

    if manifest.get("protocol") == RETIRED:
        return ([f"protocol {RETIRED} is retired: run forge.py migrate"], warnings)
    if manifest.get("protocol") != PROTOCOL:
        _err(errors, f"wrong protocol {manifest.get('protocol')!r}, want {PROTOCOL!r}")

    required: dict[str, type] = {
        "protocol": str, "pack_id": str, "created": str, "archetype": str, "contributor": dict,
        "license": str, "base": dict, "compat": dict, "decks": list, "train": dict, "files": dict,
    }
    for key, typ in required.items():
        if key not in manifest:
            _err(errors, f"pack.json missing required key {key!r}")
        elif not isinstance(manifest[key], typ):
            _err(errors, f"pack.json key {key!r} must be {typ.__name__}")
    comp = manifest.get("contributor")
    if isinstance(comp, dict) and (not isinstance(comp.get("name"), str) or not comp.get("name")):
        _err(errors, "pack.json contributor needs a 'name' string")
    elif isinstance(comp, dict) and "contact" in comp and not isinstance(comp["contact"], str):
        _err(errors, "pack.json contributor 'contact' must be a string")
    base = manifest.get("base")
    if isinstance(base, dict):
        if not isinstance(base.get("release"), str):
            _err(errors, "pack.json base needs a 'release' string")
        if not isinstance(base.get("sha256"), str):
            _err(errors, "pack.json base needs a 'sha256' string")
    compat = manifest.get("compat")
    if isinstance(compat, dict):
        for key, typ in (("obs_version", int), ("arch", dict), ("emb", str), ("emb_sha256", str),
                         ("cards_db_sha256", str), ("core", str)):
            if key not in compat:
                _err(errors, f"pack.json compat missing required key {key!r}")
            elif not isinstance(compat[key], typ):
                _err(errors, f"pack.json compat key {key!r} must be {typ.__name__}")
        if isinstance(compat.get("arch"), dict):
            for k in ARCH_KEYS:
                if k not in compat["arch"]:
                    _err(errors, f"pack.json compat.arch missing required key {k!r}")
    tr = manifest.get("train")
    if isinstance(tr, dict):
        for key, typ in (("config", dict), ("seed", int), ("iterations", int), ("stop", str)):
            if key not in tr:
                _err(errors, f"pack.json train missing required key {key!r}")
            elif not isinstance(tr[key], typ):
                _err(errors, f"pack.json train key {key!r} must be {typ.__name__}")
    decks = manifest.get("decks")
    if isinstance(decks, list):
        for i, e in enumerate(decks):
            if not isinstance(e, dict):
                _err(errors, f"pack.json decks[{i}] must be an object")
                continue
            for key, typ in (("stem", str), ("role", str), ("sha256", str)):
                if key not in e:
                    _err(errors, f"pack.json decks[{i}] missing required key {key!r}")
                elif not isinstance(e[key], typ):
                    _err(errors, f"pack.json decks[{i}] key {key!r} must be {typ.__name__}")
            if e.get("role") not in ("train", "heldout"):
                _err(errors, f"pack.json decks[{i}] role must be 'train' or 'heldout'")
            if "source" in e and not isinstance(e["source"], dict):
                _err(errors, f"pack.json decks[{i}] source must be an object")
    files = manifest.get("files")
    if isinstance(files, dict):
        for k, v in files.items():
            if not isinstance(k, str) or not isinstance(v, str):
                _err(errors, "pack.json files must map path strings to sha256 strings")
                break
    comp_block = manifest.get("compute")
    if "compute" in manifest and not isinstance(comp_block, dict):
        _err(errors, "pack.json compute must be an object")

    # 1. names (the command-injection guard: our shell scripts interpolate these)
    arch = manifest.get("archetype")
    if isinstance(arch, str) and not NAME_RE.fullmatch(arch):
        _err(errors, f"archetype {arch!r} is not a valid name (want {NAME_RE.pattern})")
    pid = manifest.get("pack_id")
    if isinstance(pid, str) and not PACK_RE.fullmatch(pid):
        _err(errors, f"pack_id {pid!r} is not a valid pack name (want {PACK_RE.pattern})")
    if isinstance(decks, list):
        for i, e in enumerate(decks):
            if isinstance(e, dict) and isinstance(e.get("stem"), str) and not NAME_RE.fullmatch(e["stem"]):
                _err(errors, f"pack.json decks[{i}] stem {e['stem']!r} is not a valid name")
    if isinstance(files, dict):
        for rel in files:
            if isinstance(rel, str) and rel.endswith(".ydk") and not _is_allowed(rel):
                stem = rel.split("/")[-1][:-4]
                _err(errors, f"pack file {rel!r} stem {stem!r} is not a valid name")

    # allowed paths: anything else is an error, before any file is read
    if isinstance(files, dict):
        for rel in files:
            if isinstance(rel, str) and not _is_allowed(rel):
                _err(errors, f"pack file {rel!r} is not an allowed pack path")
    have = {p.relative_to(pack).as_posix() for p in pack.rglob("*") if p.is_file()}
    have.discard("pack.json")
    for rel in sorted(have):
        if not _is_allowed(rel):
            _err(errors, f"pack file {rel!r} is not an allowed pack path")
        elif rel.endswith(".ydk") and not NAME_RE.fullmatch(rel.split("/")[-1][:-4]):
            _err(errors, f"pack file {rel!r} stem is not a valid name")

    # 2. sizes (bytes; JSON / JSONL must parse)
    def _over(rel: str, limit: int) -> None:
        p = pack / rel
        if p.is_file() and p.stat().st_size > limit:
            _err(errors, f"pack file {rel} is {p.stat().st_size} bytes, over the {limit} limit")

    if manifest_path.is_file() and manifest_path.stat().st_size > SIZE_PACK_JSON:
        _err(errors, f"pack.json is {manifest_path.stat().st_size} bytes, over the {SIZE_PACK_JSON} limit")
    _over("card.json", SIZE_CARD_JSON)
    _over("train.jsonl", SIZE_TRAIN_JSONL)
    _over(METRICS_REL, SIZE_POLICY_METRICS)
    _over(WEIGHTS_REL, SIZE_WEIGHTS)
    for rel in sorted(have):
        if rel.endswith(".ydk"):
            _over(rel, SIZE_YDK)
    if (pack / "card.json").is_file():
        try:
            json.loads((pack / "card.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as e:
            _err(errors, f"card.json unparsable: {e}")
    for rel in ("train.jsonl", METRICS_REL):
        p = pack / rel
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            _err(errors, f"{rel} unreadable: {e}")
            continue
        for n, raw in enumerate(lines, 1):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except (ValueError, RecursionError) as e:
                _err(errors, f"{rel} line {n} unparsable: {type(e).__name__}")
                break
            if not isinstance(rec, dict):
                _err(errors, f"{rel} line {n} must hold a JSON object")
                break

    # 3. contributor text fields (reach release notes / issue comments: reject, never sanitise)
    if isinstance(comp, dict):
        _check_text(errors, "name", comp.get("name"), min_len=1, max_len=80)
        if "contact" in comp:
            _check_text(errors, "contact", comp.get("contact"), min_len=0, max_len=120)

    # 4. licence
    lic = manifest.get("license")
    if isinstance(lic, str) and lic != "CC-BY-4.0":
        _err(errors, f'license {lic!r} != "CC-BY-4.0" (packs must be CC-BY-4.0)')

    # 6. hashes: every listed file present with its sha256; no unlisted file
    if isinstance(files, dict):
        for rel, want in files.items():
            if not isinstance(rel, str) or not isinstance(want, str):
                continue
            p = pack / rel
            if not p.is_file():
                _err(errors, f"listed file {rel} missing")
            elif sha256_file(p) != want:
                _err(errors, f"listed file {rel} sha256 mismatch (tampered?)")
        for rel in sorted(have):
            if rel not in files:
                _err(errors, f"extra unlisted file {rel}")

    # 5. decks (.ydk), data only
    if isinstance(decks, list):
        roles = [e.get("role") for e in decks if isinstance(e, dict)]
        if "train" not in roles:
            _err(errors, 'no deck with role "train"')
        if "heldout" not in roles:
            warnings.append("no held-out deck")
        db_ids = _cards_db_ids(cards_db) if cards_db is not None else None
        if cards_db is not None and db_ids is None:
            warnings.append("cards_db has no readable card table; code check skipped")
        for i, e in enumerate(decks):
            if not isinstance(e, dict):
                continue
            stem, role = e.get("stem"), e.get("role")
            if not isinstance(stem, str) or role not in ("train", "heldout"):
                continue
            dp = pack / "decks" / role / f"{stem}.ydk"
            if not dp.is_file():
                _err(errors, f"pack.json decks[{i}] file decks/{role}/{stem}.ydk missing")
                continue
            try:
                text = dp.read_text(encoding="utf-8")
            except OSError as ex:
                _err(errors, f"decks/{role}/{stem}.ydk unreadable: {ex}")
                continue
            sections, bad = _parse_ydk_raw(text)
            for b in bad:
                _err(errors, f"decks/{role}/{stem}.ydk {b}")
            n_main, n_extra, n_side = (len(sections["main"]), len(sections["extra"]), len(sections["side"]))
            if not 40 <= n_main <= 60:
                _err(errors, f"decks/{role}/{stem}.ydk main has {n_main} cards, need 40-60")
            if not 0 <= n_extra <= 15:
                _err(errors, f"decks/{role}/{stem}.ydk extra has {n_extra} cards, need 0-15")
            if not 0 <= n_side <= 15:
                _err(errors, f"decks/{role}/{stem}.ydk side has {n_side} cards, need 0-15")
            counts: dict[int, int] = {}
            for c in sections["main"] + sections["extra"] + sections["side"]:
                counts[c] = counts.get(c, 0) + 1
            for c in sorted(counts):
                if counts[c] > 3:
                    _err(errors, f"decks/{role}/{stem}.ydk code {c} appears {counts[c]}x, at most 3 copies")
            if db_ids is not None:
                for c in sorted(counts):
                    if c not in db_ids:
                        _err(errors, f"decks/{role}/{stem}.ydk code {c} not in cards_db")
            try:
                got = deck_hash(dp)
            except Exception as ex:  # noqa: BLE001 - load_ydk on raw text can only fail on I/O here
                _err(errors, f"decks/{role}/{stem}.ydk fails deck_hash: {ex!r}")
                continue
            if isinstance(e.get("sha256"), str) and got != e["sha256"]:
                _err(errors, f"decks/{role}/{stem}.ydk sha256 mismatch")

    # pack_id tracks the weights file
    wpath = pack / WEIGHTS_REL
    if isinstance(pid, str) and isinstance(arch, str) and wpath.is_file():
        try:
            want = _pack_id(arch, wpath)
        except OSError:
            want = None
        if want is not None and pid != want:
            _err(errors, f"pack_id {pid!r} != {want!r}")

    # 7. weights (only after 1-6 pass: never load weights before names, paths and sizes passed)
    if not errors:
        if not wpath.is_file():
            _err(errors, "policy/weights.safetensors missing")
        else:
            try:
                with safe_open(str(wpath), framework="numpy") as f:
                    names = f.keys()
                    wtensors = {k: f.get_slice(k)[:] for k in names}
            except Exception as e:  # noqa: BLE001 - a truncated / garbage file is an error, not an exception
                _err(errors, f"policy/weights.safetensors unreadable: {e!r}")
                wtensors = None
            if wtensors is not None:
                for k in sorted(wtensors):
                    dt = wtensors[k].dtype
                    if dt != np.float16:
                        _err(errors, f"policy/weights.safetensors tensor {k!r} dtype {dt}, want float16")
                        break
                else:
                    for k in sorted(wtensors):
                        if not bool(np.isfinite(wtensors[k]).all()):
                            _err(errors, f"policy/weights.safetensors tensor {k!r} has NaN or Inf values")
                            break
                ref_arch = ref_obs = None
                ref_t: dict[str, list] | dict | None = None
                is_release = False
                if release is not None:
                    is_release = True
                    try:
                        loaded = load_state(Path(release))
                    except Exception as e:  # noqa: BLE001
                        _err(errors, f"release {release} unreadable: {e!r}")
                        loaded = None
                    if loaded is not None and (
                        not isinstance(loaded, dict) or not loaded
                        or not all(isinstance(k, str) and torch.is_tensor(v)
                                   for k, v in loaded.items())
                    ):
                        _err(errors, f"release {release} is not a flat state dict of tensors")
                        loaded = None
                    if isinstance(loaded, dict):
                        ref_t = {k: list(v.shape) for k, v in loaded.items()}
                        rel_metrics = Path(release).parent / "metrics.jsonl"
                        ref_arch = _arch_of(_read_args(rel_metrics)) if rel_metrics.is_file() else None
                        ref_obs = None
                elif fingerprint is not None:
                    try:
                        fp = json.loads(Path(fingerprint).read_text(encoding="utf-8"))
                    except (OSError, ValueError) as e:
                        _err(errors, f"fingerprint {fingerprint} unreadable: {e}")
                        fp = None
                    if isinstance(fp, dict):
                        ok = (
                            isinstance(fp.get("release"), str)
                            and isinstance(fp.get("sha256"), str)
                            and isinstance(fp.get("obs_version"), int)
                            and isinstance(fp.get("arch"), dict)
                            and all(k in fp["arch"] for k in ARCH_KEYS)
                            and isinstance(fp.get("tensors"), dict)
                            and all(isinstance(k, str) and isinstance(v, list)
                                    and all(isinstance(d, int) for d in v)
                                    for k, v in fp["tensors"].items())
                        )
                        if not ok:
                            _err(errors, f"fingerprint {fingerprint} is not a fingerprint_of() file")
                        else:
                            ref_t = fp["tensors"]
                            ref_arch = fp["arch"]
                            ref_obs = fp["obs_version"]
                if isinstance(ref_t, dict):
                    if set(wtensors) != set(ref_t):
                        _err(errors, "policy key set differs from the reference's")
                    else:
                        bad_shape = [k for k in wtensors
                                     if list(wtensors[k].shape) != list(ref_t[k])]
                        if bad_shape:
                            _err(errors, f"policy tensor shapes differ from the reference's: {bad_shape[0]}")
                        elif is_release and isinstance(loaded, dict) and all(
                                np.array_equal(wtensors[k].astype(np.float32),
                                               np.asarray(loaded[k], dtype=np.float32))
                                for k in wtensors):
                            _err(errors, "policy equals the release exactly (nothing trained)")
                        if is_release and isinstance(loaded, dict) and not errors:
                            num = den = 0.0
                            for k in wtensors:
                                b = loaded[k]
                                if torch.is_tensor(b) and b.is_floating_point():
                                    d = wtensors[k].astype(np.float64) - np.asarray(
                                        b, dtype=np.float64)
                                    num += float(np.sum(d * d))
                                    bb = np.asarray(b, dtype=np.float64)
                                    den += float(np.sum(bb * bb))
                            rel = (num ** 0.5) / (den ** 0.5) if den > 0 else (
                                float("inf") if num > 0 else 0.0)
                            if rel > MAX_REL_DIST:
                                _err(errors, f"policy distance from release {rel:.3f} "
                                             f"> {MAX_REL_DIST} (over-trained?)")
                if (isinstance(compat, dict) and isinstance(compat.get("arch"), dict)
                        and isinstance(ref_arch, dict) and compat["arch"] != ref_arch):
                    _err(errors, f"compat.arch {compat['arch']!r} != reference arch {ref_arch!r}")
                if (isinstance(compat, dict) and isinstance(ref_obs, int)
                        and isinstance(compat.get("obs_version"), int)
                        and compat["obs_version"] != ref_obs):
                    _err(errors, f"compat.obs_version {compat['obs_version']} != reference {ref_obs}")

    # 9. compat obs version vs the running engine
    if (isinstance(compat, dict) and isinstance(compat.get("obs_version"), int)
            and compat["obs_version"] != OBS_VERSION):
        _err(errors, f"compat.obs_version {compat['obs_version']} != current {OBS_VERSION}")

    # warnings vs the given release / files
    if release is not None and isinstance(base, dict) and isinstance(base.get("sha256"), str):
        try:
            if sha256_file(Path(release)) != base["sha256"]:
                warnings.append(f"base.sha256 != release {release} sha256 (trained from an older release)")
        except OSError:
            pass
    if fingerprint is not None and isinstance(base, dict) and isinstance(base.get("sha256"), str):
        try:
            fp = json.loads(Path(fingerprint).read_text(encoding="utf-8"))
            if isinstance(fp, dict) and isinstance(fp.get("sha256"), str) and fp["sha256"] != base["sha256"]:
                warnings.append(f"base.sha256 != fingerprint {fingerprint} sha256 (trained from an older release)")
        except (OSError, ValueError):
            pass
    if emb is not None and isinstance(compat, dict) and isinstance(compat.get("emb_sha256"), str):
        ep = Path(emb)
        got = sha256_file(ep) if ep.is_file() else ""
        if got != compat["emb_sha256"]:
            warnings.append("emb_sha256 differs from the given emb file")
    if cards_db is not None and isinstance(compat, dict) and isinstance(compat.get("cards_db_sha256"), str):
        got = cards_db_hash(cards_db)
        if got != compat["cards_db_sha256"]:
            warnings.append("cards_db_sha256 differs from the given cards db file")
    return (errors, warnings)


def archive(pack: Path, dest_dir: Path) -> Path:
    pack, dest_dir = Path(pack), Path(dest_dir)
    manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    pack_id = manifest["pack_id"]
    if not isinstance(pack_id, str) or not pack_id:
        raise ValueError("pack.json has no pack_id string")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{pack_id}.tar.gz"
    rels = sorted(p.relative_to(pack).as_posix() for p in pack.rglob("*") if p.is_file())
    dirs = sorted({str(Path(r).parent) for r in rels if str(Path(r).parent) != "."})
    with tarfile.open(dest, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for d in [".", *dirs]:
            name = pack_id if d == "." else f"{pack_id}/{d}"
            ti = tarfile.TarInfo(name)
            ti.type = tarfile.DIRTYPE
            ti.mode = 0o755
            ti.mtime = 0
            ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, "", ""
            tf.addfile(ti)
        for rel in rels:
            data = (pack / rel).read_bytes()
            ti = tarfile.TarInfo(f"{pack_id}/{rel}")
            ti.size = len(data)
            ti.mode = 0o644
            ti.mtime = 0
            ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(data))
    return dest


def _archive_top(archive_path: Path) -> str:
    name = archive_path.name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    if name.endswith(".tgz"):
        return name[: -len(".tgz")]
    return name


def extract(archive_path: Path, dest_dir: Path) -> Path:
    archive_path, dest_dir = Path(archive_path), Path(dest_dir)
    allowed_dirs: set[str] = set()
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        if len(members) > MAX_MEMBERS:
            raise ValueError(f"archive has {len(members)} members, over the {MAX_MEMBERS} limit")
        total = sum(m.size for m in members if m.isfile())
        if total > MAX_UNCOMPRESSED:
            raise ValueError(f"archive uncompressed size {total} over the {MAX_UNCOMPRESSED} limit")
        tops: set[str] = set()
        for m in members:
            n = m.name
            if n.startswith(("/", "\\")):
                raise ValueError(f"archive member with absolute path: {n!r}")
            parts = n.split("/")
            if any(p == ".." for p in parts):
                raise ValueError(f"archive member with .. component: {n!r}")
            if m.issym() or m.islnk():
                raise ValueError(f"archive member is a link: {n!r}")
            if m.ischr() or m.isblk() or m.isfifo() or m.isdev():
                raise ValueError(f"archive member is a device file: {n!r}")
            tops.add(parts[0])
        tops.discard("")
        if len(tops) != 1:
            raise ValueError(f"archive must hold exactly one top-level directory, has {sorted(tops)!r}")
        (top,) = tops
        pj = next((m for m in members if m.name == f"{top}/pack.json"), None)
        if pj is None:
            raise ValueError("archive has no <top>/pack.json")
        if not pj.isfile() or pj.size > SIZE_PACK_JSON:
            raise ValueError("archive pack.json is not a regular file of at most 256 KiB")
        raw = tf.extractfile(pj)
        if raw is None:
            raise ValueError("archive pack.json unreadable")
        try:
            manifest = json.loads(raw.read().decode("utf-8"))
        except (ValueError, RecursionError) as e:
            raise ValueError(f"archive pack.json unparseable: {e}") from e
        if not isinstance(manifest, dict) or manifest.get("pack_id") != top:
            raise ValueError(f"archive top-level directory {top!r} != pack_id {manifest.get('pack_id')!r}")
        _ = _archive_top(archive_path)  # informational only; the pack_id check above is authoritative
        # v2: only regular files and directories, and every member must be an allowed pack path
        # (or a directory on the way to one). Reject before writing anything.
        allowed_dirs = {top, f"{top}/policy", f"{top}/decks",
                        f"{top}/decks/train", f"{top}/decks/heldout"}
        for m in members:
            n = m.name.rstrip("/")
            if m.isfile():
                if not n.startswith(top + "/") or not _is_allowed(n[len(top) + 1:]):
                    raise ValueError(f"archive member path not allowed: {m.name!r}")
            elif m.isdir():
                if n != top and n not in allowed_dirs:
                    raise ValueError(f"archive member path not allowed: {m.name!r}")
            else:
                raise ValueError(f"archive member is not a regular file or directory: {m.name!r}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / top
        if out.exists():
            raise ValueError(f"{out} already exists")
        tf.extractall(dest_dir, filter="data")
    return dest_dir / top
