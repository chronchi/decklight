"""Provenance sidecar for the web's precomputed content (puzzles/primers)."""

import gzip
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from decklight.web import provenance

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_stamp_writes_entry_and_keeps_other_kind(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    value = tmp_path / "value.pt"
    ckpt.write_bytes(b"ckpt-bytes")
    value.write_bytes(b"value-bytes")
    src = tmp_path / "sources.json"
    old_puzzles = {
        "model": "P",
        "ckpt_sha256": "0" * 64,
        "value": "value_p2",
        "value_sha256": "1" * 64,
        "games": ["old"],
        "built": "2026-10-18",
        "note": "old",
    }
    src.write_text(json.dumps({"puzzles": old_puzzles}) + "\n")
    entry = provenance.stamp(
        "primers",
        model="r1",
        ckpt=ckpt,
        value=value,
        games=["run-a"],
        note="n",
        path=src,
    )
    assert entry["ckpt_sha256"] == _sha(ckpt)
    assert entry["value_sha256"] == _sha(value)
    assert entry["model"] == "r1" and entry["games"] == ["run-a"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["built"])
    data = json.loads(src.read_text())
    assert data["puzzles"] == old_puzzles  # other kind kept
    assert data["primers"]["ckpt_sha256"] == _sha(ckpt)
    with pytest.raises(ValueError):
        provenance.stamp("nope", model="x", ckpt=ckpt, value=value, games=["g"], path=src)


def test_check_live_branches(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    value = tmp_path / "value.pt"
    ckpt.write_bytes(b"live-ckpt")
    value.write_bytes(b"live-value")
    src = tmp_path / "sources.json"
    # matching -> []
    provenance.stamp("puzzles", model="P", ckpt=ckpt, value=value, games=["g"], path=src)
    provenance.stamp("primers", model="P", ckpt=ckpt, value=value, games=["g"], path=src)
    assert provenance.check_live(ckpt, value, path=src) == []
    # different ckpt for one kind -> one line naming the recorded model.
    # (both kinds share the live ckpt, so make puzzles match and primers stale.)
    other = tmp_path / "other.pt"
    other.write_bytes(b"other-ckpt")
    data = json.loads(src.read_text())
    data["primers"]["ckpt_sha256"] = _sha(other)
    src.write_text(json.dumps(data) + "\n")
    lines = provenance.check_live(ckpt, value, path=src)
    assert len(lines) == 1 and "P" in lines[0] and "primers" in lines[0]
    # missing kind -> the "no provenance" line
    src.write_text(json.dumps({"puzzles": data["puzzles"]}) + "\n")
    lines = provenance.check_live(ckpt, value, path=src)
    assert any("primers" in ln and "no provenance" in ln for ln in lines)
    # missing sources file -> two "no provenance" lines
    lines = provenance.check_live(ckpt, value, path=tmp_path / "does-not-exist.json")
    assert len(lines) == 2 and all("no provenance" in ln for ln in lines)
    # unreadable ckpt path -> a line, no exception
    lines = provenance.check_live(tmp_path / "nope.pt", value, path=src)
    assert lines and all(isinstance(ln, str) for ln in lines)


def test_build_primers_commit_copy_needs_provenance(tmp_path, monkeypatch, capsys):
    build_primers = _load("build_primers")
    fake_commit = tmp_path / "primers.json.gz"
    fake_sources = tmp_path / "sources.json"
    monkeypatch.setattr(build_primers, "COMMIT_COPY", fake_commit)
    monkeypatch.setattr(provenance, "SOURCES_PATH", fake_sources)

    def _boom(*a, **k):
        raise AssertionError("aggregation must not run before the provenance check")

    monkeypatch.setattr("decklight.analysis.primer.aggregate", _boom)
    summaries = tmp_path / "summaries.json.gz"
    summaries.write_bytes(gzip.compress(json.dumps([{"dummy": 1}]).encode()))
    rc = build_primers.main(
        ["--from-summaries", "--summaries", str(summaries), "--commit-copy"]
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "ALERT build_primers" in out and "--source-model" in out
    assert not fake_commit.exists()


def test_committed_sources_parse(tmp_path):
    del tmp_path  # committed file, not a tmp file
    data = json.loads((ROOT / "src" / "decklight" / "web" / "content_sources.json").read_text())
    assert set(data) == {"puzzles", "primers"}
    hex64 = re.compile(r"[0-9a-f]{64}")
    for kind in ("puzzles", "primers"):
        e = data[kind]
        assert hex64.fullmatch(e["ckpt_sha256"])
        assert hex64.fullmatch(e["value_sha256"])
