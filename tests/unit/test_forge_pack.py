"""Forge pack format v2: build, validate, archive (tiny tensors, no network, < 20 s)."""

import io
import json
import pickle
import re
import shutil
import sqlite3
import tarfile
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from decklight.engine.obs import OBS_VERSION
from decklight.forge.pack import (
    archive,
    build_pack,
    deck_hash,
    extract,
    fingerprint_of,
    migrate_pack,
    sha256_file,
    validate_pack,
)
from decklight.model.ckpt import load_state

ROOT = Path(__file__).resolve().parents[2]
REAL_DECKS = sorted((ROOT / "decks" / "forge" / "f0" / "train" / "elfnote").glob("*.ydk"))
ARGS = {"d": 8, "layers": 1, "heads": 2, "v3": False, "hist": 0, "known": False}

MAIN1 = list(range(10000001, 10000041))  # 40 codes
MAIN2 = list(range(10000041, 10000081))  # 40 other codes
EXTRA = [20000001, 20000002]
SIDE = [30000001]


def _write_metrics(path: Path, args: dict = ARGS) -> None:
    path.write_text(json.dumps({"args": args}) + "\n", encoding="utf-8")


def _write_ydk(path: Path, main, extra=(), side=()) -> None:
    lines = ["#created by test"]
    lines.append("#main")
    lines += [str(c) for c in main]
    lines.append("#extra")
    lines += [str(c) for c in extra]
    lines.append("!side")
    lines += [str(c) for c in side]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_cdb(path: Path, codes) -> None:
    db = sqlite3.connect(path)
    try:
        db.execute("CREATE TABLE datas (id INTEGER PRIMARY KEY)")
        db.executemany("INSERT INTO datas (id) VALUES (?)", [(c,) for c in codes])
        db.commit()
    finally:
        db.close()


def _setup(tmp_path: Path, *, ft_delta: float = 0.02, ft_scale: float | None = None,
           arch_b: dict | None = None, ft_extra_key: bool = False,
           ft_shape: tuple[int, ...] | None = None) -> dict:
    """Fake release dir + fake fine-tune dir (+ emb/cards_db stubs); tiny self-made .ydk lists."""
    rel = tmp_path / "release"
    rel.mkdir(parents=True)
    base = {"w1": torch.zeros(4, 4), "w2": torch.ones(3)}
    torch.save(base, rel / "best.pt")
    _write_metrics(rel / "metrics.jsonl")
    rel_b = tmp_path / "release_b"
    rel_b.mkdir(parents=True)
    # Near-identical release: base.sha256 differs (a warning), distance stays small (no error).
    torch.save({k: v + 0.02 for k, v in base.items()}, rel_b / "best.pt")
    _write_metrics(rel_b / "metrics.jsonl", arch_b or ARGS)

    ft = tmp_path / "ft"
    (ft / "policy").mkdir(parents=True)
    (ft / "rl").mkdir(parents=True)
    if ft_scale is not None:
        changed = {k: v * ft_scale for k, v in base.items()}
    else:
        changed = {k: (v + ft_delta if k == "w1" else v) for k, v in base.items()}
    if ft_extra_key:
        changed["w3"] = torch.zeros(2)
    if ft_shape is not None:
        changed["w1"] = torch.zeros(ft_shape)
    torch.save(changed, ft / "policy" / "best.pt")
    (ft / "rl" / "metrics.jsonl").write_text(
        json.dumps({"iter": 3, "samples": 64}) + "\n", encoding="utf-8")
    emb = tmp_path / "emb.npz"
    emb.write_bytes(b"emb")
    cdb = tmp_path / "cards.cdb"
    _write_cdb(cdb, MAIN1 + MAIN2 + EXTRA + SIDE)
    deck1 = tmp_path / "deck1.ydk"
    _write_ydk(deck1, MAIN1, EXTRA, SIDE)
    deck2 = tmp_path / "deck2.ydk"
    _write_ydk(deck2, MAIN2, EXTRA, SIDE)
    return {"release": rel / "best.pt", "release_b": rel_b / "best.pt", "ft": ft,
            "emb": emb, "cdb": cdb, "deck1": deck1, "deck2": deck2}


def _build(ctx: dict, out: Path, **kw) -> Path:
    args = {"ft_policy": ctx["ft"] / "policy" / "best.pt", "train_metrics": ctx["ft"] / "rl" / "metrics.jsonl",
            "release": ctx["release"], "release_name": "r0", "archetype": "elfnote",
            "train_decks": [ctx["deck1"]], "heldout_decks": [ctx["deck2"]], "out": out,
            "contributor": {"name": "t"}, "license": "CC-BY-4.0",
            "train": {"config": {"iterations": 3}, "seed": 0, "iterations": 3, "stop": "fixed"},
            "emb": ctx["emb"], "cards_db": ctx["cdb"], "core": ""}
    args.update(kw)
    return build_pack(**args)


def _manifest(pack: Path) -> dict:
    return json.loads((pack / "pack.json").read_text(encoding="utf-8"))


def _save_manifest(pack: Path, m: dict) -> None:
    (pack / "pack.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")


def _retouch(pack: Path, rel: str, data: bytes) -> None:
    """Rewrite a pack file and fix its manifest files entry (isolates the error under test)."""
    (pack / rel).write_bytes(data)
    m = _manifest(pack)
    m["files"][rel] = sha256_file(pack / rel)
    _save_manifest(pack, m)


def _retouch_deck(pack: Path, role: str, stem: str, text: str) -> None:
    rel = f"decks/{role}/{stem}.ydk"
    (pack / rel).write_text(text, encoding="utf-8")
    m = _manifest(pack)
    m["files"][rel] = sha256_file(pack / rel)
    for e in m["decks"]:
        if e["stem"] == stem and e["role"] == role:
            e["sha256"] = deck_hash(pack / rel)
    _save_manifest(pack, m)


def _reweight(pack: Path, state: dict) -> None:
    """Replace the pack weights and fix the manifest hashes + pack_id."""
    save_file(state, pack / "policy" / "weights.safetensors")
    m = _manifest(pack)
    m["files"]["policy/weights.safetensors"] = sha256_file(pack / "policy" / "weights.safetensors")
    m["pack_id"] = f"{m['archetype']}-{m['files']['policy/weights.safetensors'][:12]}"
    _save_manifest(pack, m)


def _valid(ctx: dict, pack: Path, **kw) -> tuple[list, list]:
    kw.setdefault("emb", ctx["emb"])
    kw.setdefault("cards_db", ctx["cdb"])
    return validate_pack(pack, ctx["release"], **kw)


def test_build_validates_clean(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    manifest = _manifest(pack)
    assert re.fullmatch(r"elfnote-[0-9a-f]{12}", manifest["pack_id"])
    assert manifest["pack_id"] == f"elfnote-{sha256_file(pack / 'policy' / 'weights.safetensors')[:12]}"
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["contributor"]["name"] == "t"
    assert not (pack / "policy" / "best.pt").exists()
    for rel, want in manifest["files"].items():
        assert sha256_file(pack / rel) == want
    with safe_open(str(pack / "policy" / "weights.safetensors"), framework="numpy") as f:
        assert set(f.keys()) == {"w1", "w2"}
        names = f.keys()
        assert all(f.get_slice(k)[:].dtype.name == "float16" for k in names)
    assert manifest["decks"][0]["sha256"] == deck_hash(pack / "decks" / "train" / "deck1.ydk")
    errors, warnings = _valid(ctx, pack)
    assert errors == [] and warnings == []


def test_real_decks_validate(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack", train_decks=[REAL_DECKS[0]], heldout_decks=[REAL_DECKS[1]])
    errors, warnings = validate_pack(pack, ctx["release"], emb=ctx["emb"])
    assert errors == [] and warnings == []


def test_tampered_and_extra_file_error(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    raw = (pack / "train.jsonl").read_bytes()
    with open(pack / "train.jsonl", "ab") as f:
        f.write(b"x")
    errors, _ = _valid(ctx, pack)
    assert errors
    (pack / "train.jsonl").write_bytes(raw)
    (pack / "extra.txt").write_text("unlisted", encoding="utf-8")
    errors, _ = _valid(ctx, pack)
    assert any("extra unlisted" in e for e in errors)


def test_identical_policy_and_shape_error(tmp_path):
    ctx = _setup(tmp_path)
    same_ft = tmp_path / "ft-same"
    shutil.copytree(ctx["ft"], same_ft)
    shutil.copyfile(ctx["release"], same_ft / "policy" / "best.pt")
    pack = _build({**ctx, "ft": same_ft}, tmp_path / "pack-same")
    errors, _ = _valid(ctx, pack)
    assert any("nothing trained" in e for e in errors)

    ctx2 = _setup(tmp_path / "other")
    st = {k: v.clone() for k, v in
          torch.load(ctx2["ft"] / "policy" / "best.pt", map_location="cpu", weights_only=True).items()}
    st["w1"] = torch.zeros(5, 5)
    torch.save(st, ctx2["ft"] / "policy" / "best.pt")
    pack2 = _build(ctx2, tmp_path / "pack-shape")
    errors, _ = validate_pack(pack2, ctx2["release"], emb=ctx2["emb"], cards_db=ctx2["cdb"])
    assert any("differ" in e for e in errors)


def test_protocol_retired_and_unknown(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    manifest = _manifest(pack)
    _save_manifest(pack, dict(manifest, protocol="forge-pack/1"))
    errors, _ = validate_pack(pack, ctx["release"])
    assert len(errors) == 1 and "retired" in errors[0] and "migrate" in errors[0]
    _save_manifest(pack, dict(manifest, protocol="forge-pack/999"))
    errors, _ = validate_pack(pack, ctx["release"])
    assert any("protocol" in e for e in errors)
    _save_manifest(pack, {k: v for k, v in manifest.items() if k != "train"})
    errors, _ = validate_pack(pack, ctx["release"])
    assert any("train" in e for e in errors)


def test_pickle_best_pt_disallowed_and_never_runs(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    (pack / "policy" / "best.pt").write_bytes(pickle.dumps(Evil()))
    errors, _ = _valid(ctx, pack)
    assert any("allowed" in e for e in errors)
    assert not marker.exists(), "validate executed the pickle"
    tar = archive(pack, tmp_path / "dist")
    with pytest.raises(ValueError, match="not allowed"):
        extract(tar, tmp_path / "x")
    assert not marker.exists(), "extract executed the pickle"


def test_bad_names_rejected(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    manifest = _manifest(pack)
    bad = dict(manifest, archetype="x;rm -rf")
    _save_manifest(pack, bad)
    errors, _ = _valid(ctx, pack)
    assert any("valid name" in e for e in errors)
    bad = dict(manifest)
    bad["decks"] = [*manifest["decks"],
                    {"stem": "$(touch pwn)", "role": "train", "sha256": "x" * 64}]
    _save_manifest(pack, bad)
    errors, _ = _valid(ctx, pack)
    assert any("valid name" in e for e in errors)
    assert list(tmp_path.rglob("pwn")) == []


def test_weights_nan_f32_garbage_error(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    bad = {"w1": torch.zeros(4, 4, dtype=torch.float16), "w2": torch.ones(3, dtype=torch.float16)}
    bad["w1"][0, 0] = float("nan")
    _reweight(pack, bad)
    errors, _ = _valid(ctx, pack)
    assert any("NaN" in e for e in errors)

    _reweight(pack, {"w1": torch.zeros(4, 4), "w2": torch.ones(3)})
    errors, _ = _valid(ctx, pack)
    assert any("float16" in e for e in errors)

    _reweight(pack, {"w1": torch.zeros(4, 4, dtype=torch.float16),
                     "w2": torch.ones(3, dtype=torch.float16)})
    data = (pack / "policy" / "weights.safetensors").read_bytes()
    _retouch(pack, "policy/weights.safetensors", data[: len(data) // 2])
    m = _manifest(pack)
    m["pack_id"] = f"{m['archetype']}-{m['files']['policy/weights.safetensors'][:12]}"
    _save_manifest(pack, m)
    errors, _ = _valid(ctx, pack)
    assert any("unreadable" in e for e in errors)

    _retouch(pack, "policy/weights.safetensors", b"not a safetensors file")
    m = _manifest(pack)
    m["pack_id"] = f"{m['archetype']}-{m['files']['policy/weights.safetensors'][:12]}"
    _save_manifest(pack, m)
    errors, _ = _valid(ctx, pack)
    assert any("unreadable" in e for e in errors)


def test_distance_from_release(tmp_path):
    ctx = _setup(tmp_path)
    far = _setup(tmp_path / "far", ft_scale=1.5)
    pack = _build({**far, "release": ctx["release"], "emb": far["emb"], "cdb": far["cdb"]},
                  tmp_path / "pack-far")
    errors, _ = validate_pack(pack, ctx["release"], emb=far["emb"], cards_db=far["cdb"])
    assert any("distance" in e for e in errors)

    near = _setup(tmp_path / "near", ft_scale=1.01)
    pack = _build({**near, "release": ctx["release"], "emb": near["emb"], "cdb": near["cdb"]},
                  tmp_path / "pack-near")
    errors, warnings = validate_pack(pack, ctx["release"], emb=near["emb"], cards_db=near["cdb"])
    assert errors == [] and warnings == []


def test_decks_data_only(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")

    stem = "deck1"
    lines = ["#created by test", "#main", *map(str, MAIN1), "#extra", *map(str, EXTRA),
             "!side", *map(str, SIDE)]

    _retouch_deck(pack, "train", stem, "\n".join([*lines, "Blue-Eyes"]) + "\n")
    errors, _ = _valid(ctx, pack)
    assert any("not a decimal card code" in e for e in errors)

    short = MAIN1[:39]
    _retouch_deck(pack, "train", stem,
                  "\n".join(["#main", *map(str, short), "#extra",
                             *map(str, EXTRA), "!side", *map(str, SIDE)]) + "\n")
    errors, _ = _valid(ctx, pack)
    assert any("need 40-60" in e for e in errors)

    four = [MAIN1[0]] * 4 + MAIN1[1:37]  # still 40 cards, one code 4x
    _retouch_deck(pack, "train", stem,
                  "\n".join(["#main", *map(str, four), "#extra",
                             *map(str, EXTRA), "!side", *map(str, SIDE)]) + "\n")
    errors, _ = _valid(ctx, pack)
    assert any("copies" in e for e in errors)

    unknown = [99999999] + MAIN1[1:]
    _retouch_deck(pack, "train", stem,
                  "\n".join(["#main", *map(str, unknown), "#extra",
                             *map(str, EXTRA), "!side", *map(str, SIDE)]) + "\n")
    errors, _ = _valid(ctx, pack)
    assert any("not in cards_db" in e for e in errors)


def test_sizes_and_text(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    _retouch(pack, "train.jsonl", b"x" * (4 * 1024 * 1024 + 1))
    errors, _ = _valid(ctx, pack)
    assert any("train.jsonl" in e and "over the" in e for e in errors)

    m = _manifest(pack)
    m["contributor"] = {"name": "<script>"}
    _save_manifest(pack, m)
    errors, _ = _valid(ctx, pack)
    assert any("forbidden" in e for e in errors)

    m["contributor"] = {"name": ""}
    _save_manifest(pack, m)
    errors, _ = _valid(ctx, pack)
    assert any("needs a 'name' string" in e for e in errors)


def test_licence(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    m = _manifest(pack)
    m["license"] = "MIT"
    _save_manifest(pack, m)
    errors, _ = _valid(ctx, pack)
    assert any("CC-BY-4.0" in e for e in errors)


def test_fingerprint(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    fp = tmp_path / "fp.json"
    fp.write_text(json.dumps(fingerprint_of(ctx["release"], "r0")), encoding="utf-8")
    errors, warnings = validate_pack(pack, emb=ctx["emb"], cards_db=ctx["cdb"], fingerprint=fp)
    assert errors == [] and warnings == []

    ctx2 = _setup(tmp_path / "other", ft_shape=(5, 5))
    pack2 = _build({**ctx2, "release": ctx["release"], "emb": ctx2["emb"], "cdb": ctx2["cdb"]},
                   tmp_path / "pack2")
    errors, _ = validate_pack(pack2, emb=ctx2["emb"], cards_db=ctx2["cdb"], fingerprint=fp)
    assert any("differ" in e for e in errors)


def test_archive_roundtrip_and_unsafe_rejected(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    tar = archive(pack, tmp_path / "dist")
    assert tar.name == f"{_manifest(pack)['pack_id']}.tar.gz"
    out = extract(tar, tmp_path / "x")
    errors, _ = _valid(ctx, out)
    assert errors == []

    def _craft(name: str, members: list[tuple[str, bytes]]) -> Path:
        p = tmp_path / name
        with tarfile.open(p, "w:gz") as tf:
            top = _manifest(pack)["pack_id"]
            for rel, data in members:
                ti = tarfile.TarInfo(f"{top}/{rel}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        return p

    with pytest.raises(ValueError, match="not allowed"):
        top = _manifest(pack)["pack_id"]
        pj = json.dumps({"pack_id": top}).encode()
        extract(_craft("evil-sh.tar.gz", [("pack.json", pj), ("decks/train/x.sh", b"evil")]),
                tmp_path / "x2")

    def _craft_raw(name: str, ti: tarfile.TarInfo, data: bytes = b"x") -> Path:
        p = tmp_path / name
        with tarfile.open(p, "w:gz") as tf:
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        return p

    before = set(tmp_path.iterdir())
    dotdot = tarfile.TarInfo("../x")
    with pytest.raises(ValueError):
        extract(_craft_raw("evil-dotdot.tar.gz", dotdot), tmp_path / "x3")
    link = tarfile.TarInfo("a/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    with pytest.raises(ValueError):
        extract(_craft_raw("evil-link.tar.gz", link), tmp_path / "x4")
    assert set(tmp_path.iterdir()) == before | {tmp_path / "evil-dotdot.tar.gz",
                                                tmp_path / "evil-link.tar.gz",
                                                tmp_path / "evil-sh.tar.gz"}


def test_base_mismatch_is_warning_only(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    errors, warnings = validate_pack(pack, ctx["release_b"], emb=ctx["emb"], cards_db=ctx["cdb"])
    assert errors == []
    assert any("base.sha256" in w for w in warnings)


def test_migrate_v1_roundtrip(tmp_path):
    ctx = _setup(tmp_path)
    v1 = tmp_path / "v1pack"
    v1.mkdir()
    (v1 / "policy").mkdir()
    torch.save(torch.load(ctx["ft"] / "policy" / "best.pt", map_location="cpu", weights_only=True),
               v1 / "policy" / "best.pt")
    shutil.copy(ctx["ft"] / "rl" / "metrics.jsonl", v1 / "train.jsonl")
    shutil.copy(ctx["release"].parent / "metrics.jsonl", v1 / "policy" / "metrics.jsonl")
    (v1 / "decks" / "train").mkdir(parents=True)
    (v1 / "decks" / "heldout").mkdir(parents=True)
    shutil.copy(ctx["deck1"], v1 / "decks" / "train" / "deck1.ydk")
    shutil.copy(ctx["deck2"], v1 / "decks" / "heldout" / "deck2.ydk")
    old_state = torch.load(v1 / "policy" / "best.pt", map_location="cpu", weights_only=True)
    old_id = f"elfnote-{sha256_file(v1 / 'policy' / 'best.pt')[:12]}"
    (v1 / "pack.json").write_text(json.dumps({
        "protocol": "forge-pack/1", "pack_id": old_id, "created": "2026-01-01T00:00:00+00:00",
        "archetype": "elfnote", "contributor": {"name": "t"}, "license": "unspecified",
        "base": {"release": "r0", "sha256": sha256_file(ctx["release"])},
        "compat": {"obs_version": OBS_VERSION, "arch": dict(ARGS),
                   "emb": "emb.npz", "emb_sha256": sha256_file(ctx["emb"]),
                   "cards_db_sha256": sha256_file(ctx["cdb"]), "core": ""},
        "decks": [{"stem": "deck1", "role": "train", "sha256": deck_hash(v1 / "decks" / "train"
                  / "deck1.ydk")},
                  {"stem": "deck2", "role": "heldout", "sha256": deck_hash(v1 / "decks" / "heldout"
                  / "deck2.ydk")}],
        "train": {"config": {"iterations": 3}, "seed": 0, "iterations": 3, "stop": "fixed"},
        "files": {"policy/best.pt": sha256_file(v1 / "policy" / "best.pt"),
                  "policy/metrics.jsonl": sha256_file(v1 / "policy" / "metrics.jsonl"),
                  "train.jsonl": sha256_file(v1 / "train.jsonl"),
                  "decks/train/deck1.ydk": sha256_file(v1 / "decks" / "train" / "deck1.ydk"),
                  "decks/heldout/deck2.ydk": sha256_file(v1 / "decks" / "heldout" / "deck2.ydk")},
    }, indent=1) + "\n", encoding="utf-8")

    out = migrate_pack(v1, tmp_path / "v2pack")
    m = _manifest(out)
    assert m["protocol"] == "forge-pack/2"
    assert m["license"] == "CC-BY-4.0"
    assert m["migrated_from"] == old_id
    assert m["contributor"] == {"name": "t"}
    assert (out / "policy" / "weights.safetensors").is_file()
    assert not (out / "policy" / "best.pt").exists()
    new_state = load_state(out / "policy" / "weights.safetensors")
    assert set(new_state) == set(old_state) == {"w1", "w2"}
    for k in old_state:
        assert torch.equal(new_state[k], old_state[k].half())
    errors, warnings = validate_pack(out, ctx["release"], emb=ctx["emb"], cards_db=ctx["cdb"])
    assert errors == [] and warnings == []


def test_deeply_nested_json_is_an_error_not_a_crash(tmp_path):
    pack = tmp_path / "p"
    pack.mkdir()
    (pack / "pack.json").write_text("[" * 100000 + "]" * 100000)
    errors, _ = validate_pack(pack)
    assert errors and "pack.json" in errors[0]
