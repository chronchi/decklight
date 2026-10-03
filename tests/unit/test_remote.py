"""Remote verification: verdict serde, items jsonl, fake-box remote_verify, session CLI status."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

from ygo_zero.decks import Deck
from ygo_zero.demos.schema import ComboLine, Step
from ygo_zero.demos.verify import Verdict
from ygo_zero.engine.duel import DuelConfig, Placement
from ygo_zero.infra import remote, vast

ROOT = Path(__file__).resolve().parents[2]


def _verdict(config=None) -> Verdict:
    return Verdict(
        "line-1",
        True,
        "ok",
        [0, 3],
        ["stated", "auto"],
        7,
        12,
        [89631139],
        "2/2 steps",
        [1],
        deck_source="planned",
        executed_steps=1,
        inferred_frac=0.0,
        hand_inferred=False,
        config=config,
    )


def test_verdict_roundtrip_without_config():
    v = _verdict()
    assert remote.verdict_from_json(remote.verdict_to_json(v)) == v


def test_verdict_roundtrip_with_config_and_setup():
    cfg = DuelConfig(
        Deck((89631139, 8240199), (), (), "d0"),
        Deck((89631139,) * 3, (38033121,), (), "d1"),
        seed=3,
        setup=(
            Placement(0, 89631139, 2, 0, 0),
            Placement(1, 8240199, 4, 1, 0),
        ),
    )
    v = _verdict(cfg)
    back = remote.verdict_from_json(remote.verdict_to_json(v))
    assert back == v
    assert back.config is not None
    assert back.config.setup == cfg.setup
    assert back.config.deck0 == cfg.deck0 and back.config.deck1 == cfg.deck1


def test_items_jsonl_roundtrip(tmp_path):
    line = ComboLine(
        id="t1",
        deck="test",
        hand=["Sage with Eyes of Blue"],
        hand_codes=[8240199],
        steps=[Step(i=0, action="normal_summon", card="Sage with Eyes of Blue", card_code=8240199)],
        going="first",
    )
    items = [(line.to_json(), (8240199,), (), (), "test", 0, 100)]
    p = tmp_path / "items.jsonl"
    remote.write_items(p, items)
    assert remote.read_items(p) == items


def _fake_box(monkeypatch, tmp_path):
    """Fake the ssh/scp layer: the 'box' is a local dir; _run executes verify_items.py locally."""
    box = tmp_path / "box"
    box.mkdir()

    def _push(local: str, rmt: str) -> None:
        shutil.copy(local, box / Path(rmt).name)

    def _run(cmd: str, timeout_s) -> tuple[int, str]:
        cmd = cmd.replace("/tmp/", f"{box}/")
        p = subprocess.run(
            cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return p.returncode, p.stdout + p.stderr

    def _pull(rmt: str, local: str) -> None:
        shutil.copy(box / Path(rmt).name, local)

    monkeypatch.setattr(remote, "_push", _push)
    monkeypatch.setattr(remote, "_run", _run)
    monkeypatch.setattr(remote, "_pull", _pull)
    monkeypatch.setattr(remote, "require_session", lambda: {"instance_id": 1})


def _tiny_item() -> tuple:
    # Unresolvable names: the worker fails fast (no search), but the full path runs.
    line = ComboLine(
        id="tiny-0",
        deck="test",
        hand=["Not A Real Card"],
        hand_codes=[None],
        steps=[Step(i=0, action="normal_summon", card="Not A Real Card", card_code=None)],
        going="first",
    )
    return (line.to_json(), (89631139,), (), (), "test", 0, 100)


def test_remote_verify_over_fake_box(monkeypatch, tmp_path):
    _fake_box(monkeypatch, tmp_path)
    verdicts = remote.remote_verify([_tiny_item()], timeout_per_item=120, jobs=1)
    assert len(verdicts) == 1
    v = verdicts[0]
    assert (v.line_id, v.ok, v.reason) == ("tiny-0", False, "unresolved")


def _load_vast_session():
    spec = importlib.util.spec_from_file_location("vast_session", ROOT / "scripts" / "vast_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_session_status_without_session_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(vast, "SESSION_FILE", tmp_path / "vast_session.json")
    mod = _load_vast_session()
    monkeypatch.setattr("sys.argv", ["vast_session.py", "status"])
    mod.main()
    assert "no session" in capsys.readouterr().out
