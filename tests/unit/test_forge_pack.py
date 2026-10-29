"""Forge pack format v1: build, validate, archive (tiny tensors only, no real model, < 10 s)."""

import io
import json
import re
import shutil
import tarfile
from pathlib import Path

import pytest
import torch

from decklight.forge.pack import archive, build_pack, deck_hash, extract, sha256_file, validate_pack

ROOT = Path(__file__).resolve().parents[2]
DECKS = sorted((ROOT / "decks" / "forge" / "f0" / "train" / "elfnote").glob("*.ydk"))
ARGS = {"d": 8, "layers": 1, "heads": 2, "v3": False, "hist": 0, "known": False}


def _write_metrics(path: Path, args: dict = ARGS) -> None:
    path.write_text(json.dumps({"args": args}) + "\n", encoding="utf-8")


def _setup(tmp_path: Path, *, ft_delta: float = 5.0, arch_b: dict | None = None,
           ft_extra_key: bool = False, ft_shape: tuple[int, ...] | None = None) -> dict:
    """Fake release dir + fake fine-tune dir (+ emb/cards_db stubs); two real .ydk lists."""
    rel = tmp_path / "release"
    rel.mkdir(parents=True)
    base = {"w1": torch.zeros(4, 4), "w2": torch.ones(3)}
    torch.save(base, rel / "best.pt")
    _write_metrics(rel / "metrics.jsonl")
    rel_b = tmp_path / "release_b"
    rel_b.mkdir(parents=True)
    torch.save({k: v + 1.0 for k, v in base.items()}, rel_b / "best.pt")
    _write_metrics(rel_b / "metrics.jsonl", arch_b or ARGS)

    ft = tmp_path / "ft"
    (ft / "policy").mkdir(parents=True)
    (ft / "rl").mkdir(parents=True)
    changed = {k: (v + ft_delta if k == "w1" else v) for k, v in base.items()}
    if ft_extra_key:
        changed["w3"] = torch.zeros(2)
    if ft_shape is not None:
        changed["w1"] = torch.zeros(ft_shape)
    torch.save(changed, ft / "policy" / "best.pt")
    (ft / "rl" / "metrics.jsonl").write_text(
        json.dumps({"iter": 3, "samples": 64}) + "\n", encoding="utf-8")
    (ft / "FINETUNE.json").write_text(
        json.dumps({"iterations": 3, "seed": 0, "train_rl_cmd": ["train_rl.py"]}), encoding="utf-8")
    emb = tmp_path / "emb.npz"
    emb.write_bytes(b"emb")
    cdb = tmp_path / "cards.cdb"
    cdb.write_bytes(b"cardsdb")
    return {"release": rel / "best.pt", "release_b": rel_b / "best.pt", "ft": ft,
            "emb": emb, "cdb": cdb, "deck1": DECKS[0], "deck2": DECKS[1]}


def _build(ctx: dict, out: Path, **kw) -> Path:
    args = {"ft_policy": ctx["ft"] / "policy" / "best.pt", "train_metrics": ctx["ft"] / "rl" / "metrics.jsonl",
            "release": ctx["release"], "release_name": "r0", "archetype": "elfnote",
            "train_decks": [ctx["deck1"]], "heldout_decks": [ctx["deck2"]], "out": out,
            "contributor": {"name": "t"}, "license": "unspecified",
            "train": {"config": {"iterations": 3}, "seed": 0, "iterations": 3, "stop": "fixed"},
            "emb": ctx["emb"], "cards_db": ctx["cdb"], "core": ""}
    args.update(kw)
    return build_pack(**args)


def test_build_validates_clean(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"elfnote-[0-9a-f]{12}", manifest["pack_id"])
    assert manifest["pack_id"] == f"elfnote-{sha256_file(pack / 'policy' / 'best.pt')[:12]}"
    for rel, want in manifest["files"].items():
        assert sha256_file(pack / rel) == want
    state = torch.load(pack / "policy" / "best.pt", map_location="cpu", weights_only=True)
    assert all(v.dtype == torch.float16 for v in state.values() if v.is_floating_point())
    assert manifest["decks"][0]["sha256"] == deck_hash(pack / "decks" / "train" / f"{ctx['deck1'].stem}.ydk")
    errors, warnings = validate_pack(pack, ctx["release"], emb=ctx["emb"], cards_db=ctx["cdb"])
    assert errors == [] and warnings == []


def test_tampered_and_extra_file_error(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    raw = (pack / "train.jsonl").read_bytes()
    with open(pack / "train.jsonl", "ab") as f:
        f.write(b"x")
    errors, _ = validate_pack(pack, ctx["release"])
    assert errors
    (pack / "train.jsonl").write_bytes(raw)
    (pack / "extra.txt").write_text("unlisted", encoding="utf-8")
    errors, _ = validate_pack(pack, ctx["release"])
    assert any("extra unlisted" in e for e in errors)


def test_identical_policy_and_shape_error(tmp_path):
    ctx = _setup(tmp_path)
    same_ft = tmp_path / "ft-same"
    shutil.copytree(ctx["ft"], same_ft)
    shutil.copyfile(ctx["release"], same_ft / "policy" / "best.pt")
    pack = _build({**ctx, "ft": same_ft}, tmp_path / "pack-same")
    errors, _ = validate_pack(pack, ctx["release"])
    assert any("nothing trained" in e for e in errors)

    ctx2 = _setup(tmp_path / "other")
    pack2 = tmp_path / "pack-shape"
    ctx2["deck1"], ctx2["deck2"] = ctx["deck1"], ctx["deck2"]
    ft = ctx2["ft"] / "policy" / "best.pt"
    st = torch.load(ft, map_location="cpu", weights_only=True)
    st["w1"] = torch.zeros(5, 5)
    torch.save(st, ft)
    _build(ctx2, pack2)
    errors, _ = validate_pack(pack2, ctx2["release"])
    assert any("differ" in e for e in errors)


def test_protocol_and_missing_key_error(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    bad = dict(manifest, protocol="forge-pack/999")
    (pack / "pack.json").write_text(json.dumps(bad), encoding="utf-8")
    errors, _ = validate_pack(pack, ctx["release"])
    assert any("protocol" in e for e in errors)
    (pack / "pack.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "train"}),
                                    encoding="utf-8")
    errors, _ = validate_pack(pack, ctx["release"])
    assert any("train" in e for e in errors)


def test_archive_roundtrip_and_unsafe_rejected(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    tar = archive(pack, tmp_path / "dist")
    assert tar.name == f"{json.loads((pack / 'pack.json').read_text())['pack_id']}.tar.gz"
    out = extract(tar, tmp_path / "x")
    errors, _ = validate_pack(out, ctx["release"])
    assert errors == []

    def _craft(name: str, ti: tarfile.TarInfo, data: bytes = b"x") -> Path:
        p = tmp_path / name
        with tarfile.open(p, "w:gz") as tf:
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        return p

    before = set(tmp_path.iterdir())
    dotdot = tarfile.TarInfo("../x")
    with pytest.raises(ValueError):
        extract(_craft("evil-dotdot.tar.gz", dotdot), tmp_path / "x2")
    link = tarfile.TarInfo("a/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    with pytest.raises(ValueError):
        extract(_craft("evil-link.tar.gz", link), tmp_path / "x3")
    assert set(tmp_path.iterdir()) == before | {tmp_path / "evil-dotdot.tar.gz", tmp_path / "evil-link.tar.gz"}


def test_base_mismatch_is_warning_only(tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    errors, warnings = validate_pack(pack, ctx["release_b"], emb=ctx["emb"], cards_db=ctx["cdb"])
    assert errors == []
    assert any("base.sha256" in w for w in warnings)
