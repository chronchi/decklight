"""A5 card-pack builder: manifest/hashes, baseline diff, and that the pack mounts as extra cards.

Fake embeddings only: the Qwen model never loads on the laptop.
"""

import json
import sqlite3
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "app"))
import build_card_pack as BP  # noqa: E402

from decklight.engine import resources as R  # noqa: E402

CODE = 999999101


def _source(tmp_path: Path, desc: str = "Pack test card.") -> Path:
    d = tmp_path / "src"
    (d / "script").mkdir(parents=True)
    con = sqlite3.connect(d / "test.cdb")
    con.execute(
        "CREATE TABLE datas (id INT, ot INT, alias INT, setcode INT, type INT, atk INT, def INT, "
        "level INT, race INT, attribute INT, category INT)"
    )
    con.execute(
        "CREATE TABLE texts (id INT, name TEXT, desc TEXT, "
        + ", ".join(f"str{i} TEXT" for i in range(1, 17)) + ")"
    )
    con.execute("INSERT INTO datas VALUES (?, 1, 0, 0, 33, 1000, 1000, 4, 1, 16, 0)", (CODE,))
    con.execute(
        "INSERT INTO texts VALUES (" + ", ".join("?" * 19) + ")",
        (CODE, "Pack Test", desc, *["Pack string."] * 16),
    )
    con.commit()
    con.close()
    (d / "script" / f"c{CODE}.lua").write_text(
        "local s,id=GetID()\n"
        "function s.initial_effect(c)\n"
        "\tlocal e1=Effect.CreateEffect(c)\n"
        "\te1:SetDescription(aux.Stringid(id,0))\n"
        "\te1:SetType(EFFECT_TYPE_ACTIVATE)\n"
        "\te1:SetCode(EVENT_FREE_CHAIN)\n"
        "\tc:RegisterEffect(e1)\n"
        "end\n"
    )
    return d


def _build(tmp_path: Path, src: Path, name: str, **kw) -> tuple[Path, dict]:
    out = tmp_path / f"{name}.zip"
    manifest = BP.build_pack(
        out_zip=out, pack_id=name, source_dir=src, codes=[CODE], dim=8,
        fake_emb=True, **kw,
    )
    return out, manifest


def test_pack_manifest_and_files(tmp_path):
    src = _source(tmp_path)
    out, manifest = _build(tmp_path, src, "pack-a")
    assert manifest["format"] == BP.FORMAT
    assert manifest["codes"] == [CODE]
    assert manifest["counts"]["cards"] == 1
    assert manifest["counts"]["scripts"] == 1
    assert manifest["counts"]["embeddings"] == 1
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"manifest.json", "cards.cdb", "script/" + f"c{CODE}.lua", "emb.npz"} <= names
        m = json.loads(z.read("manifest.json"))
        for f in m["files"]:
            data = z.read(f["path"])
            assert len(data) == f["size"]
            import hashlib
            assert hashlib.sha256(data).hexdigest() == f["sha256"]


def test_baseline_diff_is_empty_then_one_changed_card(tmp_path):
    src = _source(tmp_path)
    base = tmp_path / "baseline.json"
    _, first = _build(tmp_path, src, "pack-a", write_baseline=base)
    assert first["counts"]["cards"] == 1

    out2 = tmp_path / "pack-b.zip"
    m2 = BP.build_pack(
        out_zip=out2, pack_id="pack-b", source_dir=src, baseline=json.loads(base.read_text()),
        dim=8, fake_emb=True,
    )
    assert m2.get("empty") is True and not out2.exists()

    con = sqlite3.connect(src / "test.cdb")
    con.execute("UPDATE texts SET desc='Changed text.' WHERE id=?", (CODE,))
    con.commit()
    con.close()
    out3 = tmp_path / "pack-c.zip"
    m3 = BP.build_pack(
        out_zip=out3, pack_id="pack-c", source_dir=src, baseline=json.loads(base.read_text()),
        dim=8, fake_emb=True,
    )
    assert m3["codes"] == [CODE]


def test_strings_only_change_builds_a_pack(tmp_path):
    src = _source(tmp_path)
    (src / "strings.conf").write_text("!system 1000 First string.\n")
    base = tmp_path / "baseline.json"
    _build(tmp_path, src, "pack-a", write_baseline=base)
    (src / "strings.conf").write_text("!system 1000 Changed system string.\n")
    out = tmp_path / "strings.zip"
    m = BP.build_pack(
        out_zip=out, pack_id="strings", source_dir=src,
        baseline=json.loads(base.read_text()), dim=8, fake_emb=True,
    )
    assert m.get("empty") is not True
    assert m["counts"]["cards"] == 0 and m["counts"]["strings"] == 1
    with zipfile.ZipFile(out) as z:
        assert "strings.conf" in z.namelist()


def test_pack_mounts_as_extra_cards(tmp_path, monkeypatch):
    src = _source(tmp_path)
    out, _ = _build(tmp_path, src, "pack-a")
    pack_dir = tmp_path / "mounted"
    pack_dir.mkdir()
    with zipfile.ZipFile(out) as z:
        z.extractall(pack_dir)
    monkeypatch.setenv(R.EXTRA_CARDS_ENV, str(pack_dir))
    cards = R.load_cards()
    assert cards[CODE].attack == 1000
    assert R.index_scripts()[f"c{CODE}.lua"] == pack_dir / "script" / f"c{CODE}.lua"

    z = np.load(pack_dir / "emb.npz")
    assert int(z["dim"]) == 8
    assert z["codes"].tolist() == [CODE]
    assert np.isfinite(z["card"]).all() and (np.abs(z["card"]).sum(1) > 0).all()
    lua = np.load(pack_dir / "lua_desc.npz")
    assert len(lua["keys"]) >= 1 and lua["feats"].shape[1] == len(lua["cols"])
