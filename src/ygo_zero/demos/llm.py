"""Minimal LLM client for stage-2 extraction.

Primary path: OpenAI-compatible chat-completions endpoint (PLAN §4b), key
read from `~/.local/share/opencode/auth.json` (structure inspected, key never
printed or logged). Fallback: `opencode run --pure -m <model>` subprocess.
Both paths share the same disk cache, so a cache hit never touches the net.

`last_backend` reports which path produced the last result:
"cache" | "http" | "cli".
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "cache" / "llm"
ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
AUTH_FILE = Path.home() / ".local" / "share" / "opencode" / "auth.json"
MIN_INTERVAL_S = 2.0
REQ_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json",
}

last_backend: str | None = None
_last_call_ts = 0.0


def _cache_key(model: str, system: str | None, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update((system or "").encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    return h.hexdigest()


def _read_key() -> str:
    data = json.loads(AUTH_FILE.read_text())
    return data["opencode-go"]["key"]


def _throttle() -> None:
    global _last_call_ts
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()


def _post_http(model: str, system: str | None, prompt: str, key: str, timeout: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"model": model, "messages": messages}).encode(),
        headers={"Authorization": f"Bearer {key}", **REQ_HEADERS},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"http {e.code}: {e.read()[:200]!r}") from e
    return body["choices"][0]["message"]["content"]


def _run_cli(model: str, system: str | None, prompt: str, timeout: int) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    proc = subprocess.run(
        ["opencode", "run", "--pure", "-m", model, full],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cli exit {proc.returncode}: {proc.stderr[-500:]}")
    return proc.stdout


def complete(
    prompt: str,
    model: str = "opencode-go/muse-spark-1.3-contributor",
    system: str | None = None,
    timeout: int = 300,
) -> str:
    """Complete a prompt, using the disk cache when possible."""
    global last_backend
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(model, system, prompt)}.json"
    if path.exists():
        last_backend = "cache"
        return json.loads(path.read_text())["response"]

    errors: list[str] = []
    # Primary: HTTP endpoint, 3 attempts with backoff on 429/5xx.
    try:
        key = _read_key()
    except Exception as e:  # noqa: BLE001 - missing auth just disables the http path
        key = ""
        errors.append(f"auth: {e}")
    if key:
        for attempt in range(3):
            _throttle()
            try:
                text = _post_http(model, system, prompt, key, timeout)
                path.write_text(json.dumps({"model": model, "response": text}))
                last_backend = "http"
                return text
            except Exception as e:  # noqa: BLE001 - collect and retry/fall back
                errors.append(f"http try{attempt}: {e}")
                # retry with backoff on 429/5xx (or no status at all); other
                # 4xx mean the endpoint rejects this client -> go to CLI now.
                m = re.search(r"http (\d+)", str(e))
                if m and int(m.group(1)) not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2 ** (attempt + 1))
    # Fallback: CLI subprocess (same cache entry).
    _throttle()
    try:
        text = _run_cli(model, system, prompt, timeout)
        path.write_text(json.dumps({"model": model, "response": text}))
        last_backend = "cli"
        return text
    except Exception as e:  # noqa: BLE001 - report both paths in the error
        errors.append(f"cli: {e}")
    raise RuntimeError(f"llm.complete failed ({'; '.join(errors)})")
