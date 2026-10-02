"""Parallel helpers for the demo pipeline (docs/workflow.md cheap wins).

- scripts/verify_lines.py + scripts/expand_lines.py: ``map_ordered`` runs jobs in
  worker processes; results come back in serial (input) order.
- demos/llm.py: ``complete_many`` runs prompts concurrently on the same disk
  cache; the HTTP endpoint is skipped after its first failure in a process.
- demos/extract.py: per-section calls dispatch concurrently, assembled in order.
"""

import importlib.util
import json
import threading
import time
from pathlib import Path

from ygo_zero.demos import llm

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_verify = _load_script("verify_lines")
_expand = _load_script("expand_lines")


def _slow_square(x: int) -> int:
    """Fake worker: later items finish first, so naive gather would misorder."""
    time.sleep(0.01 * (8 - x))
    return x * x


def test_verify_map_ordered_identity():
    items = list(range(8))
    want = [x * x for x in items]
    assert _verify.map_ordered(_slow_square, items, 1) == want
    assert _verify.map_ordered(_slow_square, items, 2) == want
    assert _verify.map_ordered(_slow_square, [], 2) == []
    assert _verify.default_jobs() >= 1


def test_expand_map_ordered_identity():
    items = list(range(8))
    want = [x * x for x in items]
    assert _expand.map_ordered(_slow_square, items, 1) == want
    assert _expand.map_ordered(_slow_square, items, 2) == want
    assert _expand.default_jobs() >= 1


# --- demos/llm.py -------------------------------------------------------------


def _no_key():
    raise RuntimeError("no auth")


def test_complete_many_cache_and_order(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(llm, "_read_key", _no_key)
    calls: list[str] = []

    def fake_cli(model, system, prompt, timeout):
        calls.append(prompt)
        return f"resp:{prompt}"

    monkeypatch.setattr(llm, "_run_cli", fake_cli)
    prompts = ["p1", "p2", "p3"]
    out = llm.complete_many(prompts, system="S")
    assert out == [f"resp:{p}" for p in prompts]
    assert sorted(calls) == prompts  # each prompt ran exactly once
    cached = list(tmp_path.glob("*.json"))
    assert len(cached) == 3
    assert list(tmp_path.glob("*.tmp")) == []  # atomic writes leave no temp files
    for f in cached:
        assert "response" in json.loads(f.read_text())
    out2 = llm.complete_many(prompts, system="S")  # all cache hits now
    assert out2 == out
    assert len(calls) == 3


def test_complete_many_return_exceptions(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(llm, "_read_key", _no_key)

    def fake_cli(model, system, prompt, timeout):
        if prompt == "boom":
            raise RuntimeError("cli exit 1")
        return f"resp:{prompt}"

    monkeypatch.setattr(llm, "_run_cli", fake_cli)
    out = llm.complete_many(["ok1", "boom", "ok2"], system="S", return_exceptions=True)
    assert out[0] == "resp:ok1" and out[2] == "resp:ok2"
    assert isinstance(out[1], RuntimeError)
    try:
        llm.complete_many(["boom"], system="S")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected complete_many to raise without return_exceptions")


def test_http_skipped_after_first_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(llm, "http_dead", False)
    monkeypatch.setattr(llm, "_read_key", lambda: "k")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    posts: list[str] = []
    clis: list[str] = []

    def fake_post(model, system, prompt, key, timeout):
        posts.append(prompt)
        raise RuntimeError("http 400: bad request")

    def fake_cli(model, system, prompt, timeout):
        clis.append(prompt)
        return "cli-ok"

    monkeypatch.setattr(llm, "_post_http", fake_post)
    monkeypatch.setattr(llm, "_run_cli", fake_cli)
    assert llm.complete("a") == "cli-ok"
    assert len(posts) == 1 and llm.http_dead is True
    assert llm.complete("b") == "cli-ok"
    assert len(posts) == 1 and len(clis) == 2  # second call never tried HTTP


# --- demos/extract.py ----------------------------------------------------------

MOCK_JSON = json.dumps(
    {
        "lines": [
            {
                "title": "Sage opener",
                "hand": ["Sage with Eyes of Blue"],
                "requires": [],
                "steps": [
                    {
                        "i": 0,
                        "action": "normal_summon",
                        "card": "Sage with Eyes of Blue",
                        "text": "Normal Summon Sage with Eyes of Blue.",
                    }
                ],
                "end_board": ["Blue-Eyes Spirit Dragon"],
                "end_board_text": "Spirit Dragon pass",
                "going": "first",
            }
        ]
    }
)


def _guide(path: Path, n: int = 3) -> None:
    secs = "".join(f"COMBO {i}: thing{i}\nDo the thing with Sage.\n\n" for i in range(1, n + 1))
    path.write_text("URL: https://example.com/g\n" + "=" * 70 + "\n" + secs)


def test_extract_sections_run_concurrently(tmp_path):
    from ygo_zero.demos.extract import extract_lines
    from ygo_zero.demos.names import CardNames

    guide = tmp_path / "g.txt"
    _guide(guide)
    lock = threading.Lock()
    active = 0
    peak = 0
    seen: list[str] = []

    def mock(prompt, system=None, **kw):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            seen.append(prompt)
        try:
            time.sleep(0.2)
            return MOCK_JSON
        finally:
            with lock:
                active -= 1

    lines = extract_lines(guide, "blue-eyes", {8240199, 89631139}, names=CardNames(), complete_fn=mock)
    assert [line.id for line in lines] == [f"blue-eyes/g:{i}" for i in range(3)]
    assert len(seen) == 3  # one call per section, no repeats
    assert peak >= 2  # calls actually overlapped


def test_extract_skips_failed_section(tmp_path):
    from ygo_zero.demos.extract import extract_lines
    from ygo_zero.demos.names import CardNames

    guide = tmp_path / "g.txt"
    _guide(guide)

    def mock(prompt, system=None, **kw):
        if "COMBO 2" in prompt:
            raise RuntimeError("net down")
        return MOCK_JSON

    lines = extract_lines(guide, "blue-eyes", {8240199, 89631139}, names=CardNames(), complete_fn=mock)
    assert [line.id for line in lines] == ["blue-eyes/g:0", "blue-eyes/g:1"]
