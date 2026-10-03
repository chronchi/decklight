import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import datagen_loop as d


def test_session_names_and_labels():
    assert d.line_session_names(1) == ["lines"]
    assert d.line_session_names(3) == ["lines", "lines2", "lines3"]
    assert d.session_label("lines", "1200") == "verify-1200"
    assert d.session_label("games", "1200") == "games-1200"
    assert d.session_label("lines2", "1200") == "lines2-1200"
    assert d.session_command("1200", "lines2")[-2:] == ["--session", "lines2"]


def test_worker_commands_carry_session():
    assert d.planner_command("x", 1, None, "lines3")[-4:-2] == ["--session", "lines3"] or "lines3" in d.planner_command(
        "x", 1, None, "lines3")
    assert d.improve_command("data/demos/planned-s1", "x", "lines2")[-2:] == ["--session", "lines2"]
    assert d.expand_command(Path("a/verified/x.jsonl"), Path("a/lines"), Path("a/expanded-x"), [], "lines2")[-2:] == [
        "--session", "lines2"]
