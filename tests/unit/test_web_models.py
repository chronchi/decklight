"""Model discovery for the web lobby (base + per-deck fine-tunes + CLI override).

No model loading (laptop has no GPU): the fixtures are empty best.pt files.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from decklight.web.models import discover_models

ROOT = Path(__file__).resolve().parents[2]
DECKS = [{"id": "branded"}, {"id": "blue-eyes"}]


def _write_tune(root: Path, name: str, deck_text: str, hours: float, *, mtime=None) -> Path:
    d = root / name
    (d / "policy").mkdir(parents=True)
    (d / "value").mkdir(parents=True)
    (d / "policy" / "best.pt").write_bytes(b"")
    (d / "value" / "best.pt").write_bytes(b"")
    lst = root / "lists" / f"{name}.ydk"
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text(deck_text)
    ft = d / "FINETUNE.json"
    ft.write_text(json.dumps({"deck": str(lst), "hours": hours}))
    if mtime is not None:
        os.utime(ft, (mtime, mtime))
    return ft


def _branded_text() -> str:
    return (ROOT / "decks" / "pool" / "branded.ydk").read_text()


def _blue_eyes_text() -> str:
    return (ROOT / "decks" / "pool" / "blue-eyes.ydk").read_text()


def test_base_is_always_first(tmp_path):
    models = discover_models([tmp_path], DECKS, None, None)
    assert models[0]["id"] == "base"
    assert models[0]["label"] == "Base model" and models[0]["kind"] == "base"


def test_missing_value_skipped_with_alert(tmp_path, capsys):
    _write_tune(tmp_path, "ft-novalue", _branded_text(), 1.0)
    noval = tmp_path / "ft-novalue" / "value" / "best.pt"
    noval.unlink()
    models = discover_models([tmp_path], DECKS, None, None)
    assert [m["id"] for m in models] == ["base"]
    out = capsys.readouterr().out
    assert "ALERT web: model" in out and "value/best.pt missing" in out


def test_deck_id_matched_by_content_not_name(tmp_path):
    # Same cards as the lobby's "branded" list, but a different file name.
    _write_tune(tmp_path, "ft-a", _branded_text(), 0.8)
    models = discover_models([tmp_path], DECKS, None, None)
    assert len(models) == 2
    tune = models[1]
    assert tune["deck_id"] == "branded"
    assert tune["deck_name"] == "ft-a" and tune["deck_name"] != "branded"
    assert tune["label"] == "Tuned for ft-a"
    assert tune["hours"] == 0.8 and tune["short"] == tune["id"][:4] and len(tune["id"]) == 10
    want = hashlib.sha256(
        str((tmp_path / "ft-a" / "FINETUNE.json").resolve()).encode()
    ).hexdigest()[:10]
    assert tune["id"] == want


def test_newest_first(tmp_path):
    now = time.time()
    _write_tune(tmp_path, "ft-old", _branded_text(), 0.5, mtime=now - 100000)
    _write_tune(tmp_path, "ft-new", _blue_eyes_text(), 1.2, mtime=now)
    models = discover_models([tmp_path], DECKS, None, None)
    assert [m["hours"] for m in models[1:]] == [1.2, 0.5]
    assert models[1]["deck_id"] == "blue-eyes" and models[2]["deck_id"] == "branded"


def test_cli_custom_entry(tmp_path):
    ckpt = tmp_path / "custom" / "policy" / "best.pt"
    value = tmp_path / "custom" / "value" / "best.pt"
    models = discover_models([tmp_path], DECKS, ckpt, value)
    cli = next(m for m in models if m["kind"] == "cli")
    assert cli["label"] == "Custom model"
    assert Path(cli["ckpt"]) == ckpt and Path(cli["value"]) == value


def test_cli_matching_discovered_entry_adds_nothing(tmp_path):
    ft = _write_tune(tmp_path, "ft-a", _branded_text(), 0.8)
    policy = ft.parent / "policy" / "best.pt"
    value = ft.parent / "value" / "best.pt"
    models = discover_models([tmp_path], DECKS, policy, value)
    assert [m["kind"] for m in models] == ["base", "finetune"]


def test_lobby_payload_strips_paths_and_default(tmp_path):
    from decklight.web.server import Server

    now = time.time()
    _write_tune(tmp_path, "ft-old", _branded_text(), 0.5, mtime=now - 100000)
    new_ft = _write_tune(tmp_path, "ft-new", _blue_eyes_text(), 1.2, mtime=now)
    srv = Server(token="t", models_dirs=[tmp_path])
    lobby = srv.lobby_message()
    assert lobby["t"] == "lobby" and lobby["decks"]
    for m in lobby["models"]:
        assert "ckpt" not in m and "value" not in m
    assert lobby["models"][0]["id"] == "base"
    # No CLI override: the default stays the base model; tunes sort newest first.
    assert lobby["model_default"] == "base"
    want = hashlib.sha256(str(new_ft.resolve()).encode()).hexdigest()[:10]
    assert lobby["models"][1]["id"] == want
