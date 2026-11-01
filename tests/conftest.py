import os
from pathlib import Path

import pytest

# Under pytest-xdist each worker gets one intra-op thread: 4 workers x 4 torch/OpenMP threads on the laptop's 4 cores
# spun against each other and turned a 3 s collector test into 203 s (NOTEBOOK 2026-11-01, test budget). Set before
# torch is imported, so test subprocesses inherit it too.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

from decklight.engine.resources import CORE_LIB

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        import torch

        torch.set_num_threads(1)


def pytest_collection_modifyitems(config, items):
    """Tier marker = top-level test folder (unit / engine / gpu)."""
    for item in items:
        tier = Path(item.fspath).relative_to(Path(__file__).parent).parts[0]
        item.add_marker(getattr(pytest.mark, tier))


@pytest.fixture(scope="session")
def engine():
    if not CORE_LIB.exists():
        pytest.skip("core not built: run scripts/build_core.sh")
    from decklight.engine.duel import Engine

    return Engine()


@pytest.fixture(scope="session")
def decks():
    from decklight.decks import load_ydk

    return {p.stem: load_ydk(p) for p in sorted((FIXTURES / "decks").glob("*.ydk"))}


@pytest.fixture(autouse=True)
def _no_real_opencode_log(monkeypatch, tmp_path):
    """llm reads opencode's log for rate/usage limits; tests must not depend on the machine's live log."""
    from decklight.demos import llm

    monkeypatch.setattr(llm, "OPENCODE_LOG", tmp_path / "no-opencode.log")
    monkeypatch.setattr(llm, "QUOTA_MARKER", tmp_path / "QUOTA_EXCEEDED")
