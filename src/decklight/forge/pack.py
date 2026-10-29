"""Forge pack format v1: build, validate and archive a specialist fine-tune submission.

A pack is a directory (archived as ``<pack_id>.tar.gz``) holding the specialist's
policy weights plus the metadata the coordinator needs to verify and distill it.
See docs/plans/2026-10-29-deck-forge.md, section "Protocol: the pack".
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from decklight.decks.ydk import load_ydk
from decklight.engine.obs import OBS_VERSION

PROTOCOL = "forge-pack/1"

ARCH_DEFAULTS = {"d": 256, "layers": 4, "heads": 4, "v3": False, "hist": 0, "known": False}
ARCH_KEYS = ("d", "layers", "heads", "v3", "hist", "known")

MAX_MEMBERS = 200
MAX_UNCOMPRESSED = 64 * 1024 * 1024


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
                continue
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


def _load_state_flat(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


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
    release_metrics = release.parent / "metrics.jsonl"
    if not release_metrics.is_file():
        raise FileNotFoundError(f"{release_metrics} missing: cannot read the release architecture")

    out.mkdir(parents=True)
    (out / "policy").mkdir(parents=True)
    (out / "decks" / "train").mkdir(parents=True)
    (out / "decks" / "heldout").mkdir(parents=True)

    state = _load_state_flat(ft_policy)
    fp16 = {k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in state.items()}
    policy_pt = out / "policy" / "best.pt"
    torch.save(fp16, policy_pt)
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

    pack_id = f"{archetype}-{sha256_file(policy_pt)[:12]}"
    files = {}
    for p in sorted(f for f in out.rglob("*") if f.is_file()):
        rel = p.relative_to(out).as_posix()
        if rel == "pack.json":
            continue
        files[rel] = sha256_file(p)
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
        "files": files,
    }
    if compute is not None:
        manifest["compute"] = compute
    (out / "pack.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return out


def _err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def validate_pack(pack: Path, release: Path | None = None, *, emb: Path | None = None,
                  cards_db: Path | list[Path] | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    pack = Path(pack)
    manifest_path = pack / "pack.json"
    if not pack.is_dir():
        return ([f"pack {pack} is not a directory"], warnings)
    if not manifest_path.is_file():
        return ([f"{manifest_path} missing"], warnings)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return ([f"pack.json unparseable: {e}"], warnings)
    if not isinstance(manifest, dict):
        return (["pack.json must hold a JSON object"], warnings)

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

    # file presence / hashes
    if isinstance(files, dict):
        for rel, want in files.items():
            if not isinstance(rel, str) or not isinstance(want, str):
                continue
            p = pack / rel
            if not p.is_file():
                _err(errors, f"listed file {rel} missing")
            elif sha256_file(p) != want:
                _err(errors, f"listed file {rel} sha256 mismatch (tampered?)")
        have = {p.relative_to(pack).as_posix() for p in pack.rglob("*") if p.is_file()}
        have.discard("pack.json")
        for rel in sorted(have):
            if rel not in files:
                _err(errors, f"extra unlisted file {rel}")

    # policy weights
    policy_pt = pack / "policy" / "best.pt"
    state: dict | None = None
    if not policy_pt.is_file():
        _err(errors, "policy/best.pt missing")
    else:
        try:
            loaded = torch.load(policy_pt, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001 - any load failure is an error
            _err(errors, f"policy/best.pt does not load weights-only: {e!r}")
            loaded = None
        if loaded is not None and (
            not isinstance(loaded, dict) or not loaded
            or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in loaded.items())
        ):
            _err(errors, "policy/best.pt is not a flat state dict of tensors")
            loaded = None
        if loaded is not None and isinstance(loaded, dict):
            state = loaded

    # release comparison
    rel_state: dict | None = None
    if release is not None:
        release = Path(release)
        try:
            rel_loaded = torch.load(release, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            _err(errors, f"release {release} does not load weights-only: {e!r}")
            rel_loaded = None
        if isinstance(rel_loaded, dict):
            if "model" in rel_loaded and isinstance(rel_loaded["model"], dict):
                rel_loaded = rel_loaded["model"]
            if isinstance(rel_loaded, dict) and all(torch.is_tensor(v) for v in rel_loaded.values()):
                rel_state = rel_loaded
        if state is not None and rel_state is not None:
            if set(state) != set(rel_state):
                _err(errors, "policy key set differs from the release's")
            else:
                shapes = [k for k in state if tuple(state[k].shape) != tuple(rel_state[k].shape)]
                if shapes:
                    _err(errors, f"policy tensor shapes differ from the release's: {shapes[0]}")
                elif all(torch.equal(state[k].float(), rel_state[k].float()) for k in state):
                    _err(errors, "policy equals the release exactly (nothing trained)")
        rel_metrics = release.parent / "metrics.jsonl"
        if isinstance(compat, dict) and isinstance(compat.get("arch"), dict):
            want_arch = _arch_of(_read_args(rel_metrics)) if rel_metrics.is_file() else None
            if want_arch is not None and compat["arch"] != want_arch:
                _err(errors, f"compat.arch {compat['arch']!r} != release arch {want_arch!r}")

    # compat obs version
    if (isinstance(compat, dict) and isinstance(compat.get("obs_version"), int)
            and compat["obs_version"] != OBS_VERSION):
        _err(errors, f"compat.obs_version {compat['obs_version']} != current {OBS_VERSION}")

    # decks
    if isinstance(decks, list):
        roles = [e.get("role") for e in decks if isinstance(e, dict)]
        if "train" not in roles:
            _err(errors, 'no deck with role "train"')
        if "heldout" not in roles:
            warnings.append("no held-out deck")
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
                d = load_ydk(dp)
            except Exception as ex:  # noqa: BLE001
                _err(errors, f"decks/{role}/{stem}.ydk fails load_ydk: {ex!r}")
                continue
            if not d.main:
                _err(errors, f"decks/{role}/{stem}.ydk has an empty main deck")
                continue
            try:
                got = deck_hash(dp)
            except Exception as ex:  # noqa: BLE001
                _err(errors, f"decks/{role}/{stem}.ydk fails deck_hash: {ex!r}")
                continue
            if isinstance(e.get("sha256"), str) and got != e["sha256"]:
                _err(errors, f"decks/{role}/{stem}.ydk sha256 mismatch")

    # pack_id
    if isinstance(manifest.get("pack_id"), str) and isinstance(manifest.get("archetype"), str):
        if policy_pt.is_file() and state is not None:
            want = f"{manifest['archetype']}-{sha256_file(policy_pt)[:12]}"
            if manifest["pack_id"] != want:
                _err(errors, f"pack_id {manifest['pack_id']!r} != {want!r}")
        elif policy_pt.is_file():
            try:
                want = f"{manifest['archetype']}-{sha256_file(policy_pt)[:12]}"
            except OSError:
                want = None
            if want is not None and manifest["pack_id"] != want:
                _err(errors, f"pack_id {manifest['pack_id']!r} != {want!r}")

    # warnings vs the given release / files
    if release is not None and isinstance(base, dict) and isinstance(base.get("sha256"), str):
        try:
            if sha256_file(Path(release)) != base["sha256"]:
                warnings.append(f"base.sha256 != release {release} sha256 (trained from an older release)")
        except OSError:
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
        raw = tf.extractfile(pj)
        if raw is None:
            raise ValueError("archive pack.json unreadable")
        try:
            manifest = json.loads(raw.read().decode("utf-8"))
        except ValueError as e:
            raise ValueError(f"archive pack.json unparseable: {e}") from e
        if not isinstance(manifest, dict) or manifest.get("pack_id") != top:
            raise ValueError(f"archive top-level directory {top!r} != pack_id {manifest.get('pack_id')!r}")
        _ = _archive_top(archive_path)  # informational only; the pack_id check above is authoritative
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / top
        if out.exists():
            raise ValueError(f"{out} already exists")
        tf.extractall(dest_dir, filter="data")
    return dest_dir / top
