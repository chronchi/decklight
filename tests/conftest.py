from pathlib import Path

import pytest

from ygo_zero.engine.resources import CORE_LIB

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config, items):
    """Tier marker = top-level test folder (unit / engine / gpu)."""
    for item in items:
        tier = Path(item.fspath).relative_to(Path(__file__).parent).parts[0]
        item.add_marker(getattr(pytest.mark, tier))


@pytest.fixture(scope="session")
def engine():
    if not CORE_LIB.exists():
        pytest.skip("core not built: run scripts/build_core.sh")
    from ygo_zero.engine.duel import Engine

    return Engine()


@pytest.fixture(scope="session")
def decks():
    from ygo_zero.decks import load_ydk

    return {p.stem: load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))}


@pytest.fixture(autouse=True)
def _no_real_opencode_log(monkeypatch, tmp_path):
    """llm reads opencode's log for rate/usage limits; tests must not depend on the machine's live log."""
    from ygo_zero.demos import llm

    monkeypatch.setattr(llm, "OPENCODE_LOG", tmp_path / "no-opencode.log")
    monkeypatch.setattr(llm, "QUOTA_MARKER", tmp_path / "QUOTA_EXCEEDED")
