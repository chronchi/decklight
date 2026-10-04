"""Minimal LLM client for stage-2 extraction.

Primary path: OpenAI-compatible chat-completions endpoint (PLAN §4b), key
read from `~/.local/share/opencode/auth.json` (structure inspected, key never
printed or logged). Fallback: `opencode run --pure -m <model>` subprocess.
Both paths share the same disk cache, so a cache hit never touches the net.

Fastest path (opt-in): a long-running `opencode serve` (scripts/llm_server.sh) set in `YGO_OPENCODE_SERVER`, e.g.
http://127.0.0.1:4097. One HTTP call per completion, no client process: about 1.5 s overhead instead of 7.5 s,
~0 client CPU instead of ~11 CPU-s and 600 MB per call (NOTEBOOK 2026-10-02 23:40).

`variant` = the provider's reasoning effort (muse: minimal|low|high|max; None = default). It is part of the cache
key only when set.

`last_backend` reports which path produced the last result:
"cache" | "server" | "http" | "cli".
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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
_throttle_lock = threading.Lock()
SERVER_URL = os.environ.get("YGO_OPENCODE_SERVER") or None
DEFAULT_VARIANT = os.environ.get("YGO_LLM_VARIANT") or None
# Once the HTTP endpoint has failed in this process it always fails here
# (workflow.md "Skip the dead HTTP attempt"), so stop trying it.
http_dead = False


def _cache_key(model: str, system: str | None, prompt: str, variant: str | None = None) -> str:
    h = hashlib.sha256()
    if model.startswith("opencode/") and model.endswith("-free"):  # same model on the free Zen tier: keep cache valid
        model = "opencode-go/" + model.removeprefix("opencode/").removesuffix("-free")
    h.update(model.encode() + (f"#{variant}".encode() if variant else b""))
    h.update(b"\x00")
    h.update((system or "").encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    return h.hexdigest()


OPENCODE_LOG = Path.home() / ".local" / "share" / "opencode" / "log" / "opencode.log"
QUOTA_MARKER = CACHE_DIR / "QUOTA_EXCEEDED"


class QuotaExceeded(RuntimeError):
    """The OpenCode Go plan's usage limit is hit: every backend will hang/retry, so fail fast (2026-10-03 20:30)."""


def quota_exceeded(window_s: float = 600) -> bool:
    """True when opencode logged "usage limit exceeded" in the last `window_s` seconds (tail of its log)."""
    try:
        with open(OPENCODE_LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            tail = f.read().decode(errors="ignore").splitlines()
    except OSError:
        return False
    now = time.time()
    for line in reversed(tail):
        if "usage limit exceeded" not in line.lower():
            continue
        m = re.match(r"timestamp=(\S+?)(?:\.\d+)?Z", line)
        if not m:
            return True
        t = time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) - time.timezone
        return now - t < window_s
    return False


def _flag_quota() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    QUOTA_MARKER.write_text(str(time.time()))
    raise QuotaExceeded("OpenCode Go usage limit exceeded (see ~/.local/share/opencode/log/opencode.log)")


def _read_key() -> str:
    data = json.loads(AUTH_FILE.read_text())
    return data["opencode-go"]["key"]


def _throttle() -> None:
    global _last_call_ts
    # Reserve the next free slot under the lock so concurrent threads stay MIN_INTERVAL_S apart.
    with _throttle_lock:
        now = time.monotonic()
        slot = max(now, _last_call_ts + MIN_INTERVAL_S)
        _last_call_ts = slot
    if slot > now:
        time.sleep(slot - now)


def _write_cache(path: Path, payload: dict) -> None:
    """Write a cache entry atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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


def _server_req(method: str, path: str, body: dict | None, timeout: float) -> dict | None:
    req = urllib.request.Request(
        SERVER_URL.rstrip("/") + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def _post_server(model: str, system: str | None, prompt: str, variant: str | None, timeout: int) -> str:
    """One completion through the opencode server: a throwaway session, tools off, then deleted."""
    provider, model_id = model.split("/", 1)
    sid = _server_req("POST", "/session", {"title": "ygo-zero"}, 30)["id"]
    try:
        # No "tools" override: OpenCode's free tier rejects requests without its tool set (403 "free tier can only be
        # used from within OpenCode"). Safety comes from the server's config instead (scripts/llm_server.sh:
        # OPENCODE_CONFIG=config/opencode-llm.json with bash/edit/webfetch/... = "ask"; nobody answers, so a tool call stalls to the timeout).
        body = {"model": {"providerID": provider, "modelID": model_id},
                "parts": [{"type": "text", "text": prompt}]}  # fmt: skip
        if system:
            body["system"] = system
        if variant:
            body["variant"] = variant
        r = _server_req("POST", f"/session/{sid}/message", body, timeout)
    finally:
        try:
            _server_req("DELETE", f"/session/{sid}", None, 30)
        except Exception:  # noqa: BLE001, S110 - cleanup only
            pass
    if r is None or r.get("info", {}).get("error"):
        raise RuntimeError(f"server error: {str((r or {}).get('info', {}).get('error'))[:300]}")
    text = "".join(p.get("text", "") for p in r.get("parts", []) if p.get("type") == "text")
    if not text.strip():
        raise RuntimeError("server: empty reply")
    return text


CLI_SLOTS = int(os.environ.get("YGO_LLM_CLI_SLOTS", "2"))  # machine-wide cap on `opencode run` (~450 MB each)


def _cli_slot(timeout: float) -> int:
    """Hold one of CLI_SLOTS lock files (flock, released when the fd closes); wait up to `timeout` s."""
    d = CACHE_DIR / "cli-slots"
    d.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + timeout
    while True:
        for i in range(CLI_SLOTS):
            fd = os.open(d / f"{i}.lock", os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError:
                os.close(fd)
        if time.monotonic() > end:
            raise RuntimeError(f"cli: no free slot of {CLI_SLOTS} in {timeout:.0f} s")
        time.sleep(3)


def _run_cli(model: str, system: str | None, prompt: str, timeout: int, variant: str | None = None) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    fd = _cli_slot(timeout)
    try:
        return _run_cli_locked(model, full, timeout, variant)
    finally:
        os.close(fd)


def _run_cli_locked(model: str, full: str, timeout: int, variant: str | None) -> str:
    env = dict(os.environ, OPENCODE_CONFIG=str(ROOT / "config" / "opencode-llm.json"))
    proc = subprocess.run(
        ["opencode", "run", "--pure", "-m", model, *(["--variant", variant] if variant else []), full],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cli exit {proc.returncode}: {proc.stderr[-500:]}")
    return proc.stdout


def complete(
    prompt: str,
    model: str = "opencode/muse-spark-1.3-contributor-free",
    system: str | None = None,
    timeout: int = 300,
    variant: str | None = None,
) -> str:
    """Complete a prompt, using the disk cache when possible."""
    global last_backend, http_dead
    variant = variant or DEFAULT_VARIANT
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(model, system, prompt, variant)}.json"
    if path.exists():
        last_backend = "cache"
        return json.loads(path.read_text())["response"]

    errors: list[str] = []
    if SERVER_URL:
        for attempt in range(2):
            _throttle()
            try:
                text = _post_server(model, system, prompt, variant, timeout)
                _write_cache(path, {"model": model, "variant": variant, "response": text})
                last_backend = "server"
                return text
            except Exception as e:  # noqa: BLE001 - retry once, then fall back to the CLI
                errors.append(f"server try{attempt}: {e}"[:300])
                if quota_exceeded():
                    _flag_quota()
                time.sleep(2 ** (attempt + 1))
    # Primary: HTTP endpoint, 3 attempts with backoff on 429/5xx.
    # Skipped entirely once it has failed in this process.
    try:
        key = _read_key()
    except Exception as e:  # noqa: BLE001 - missing auth just disables the http path
        key = ""
        errors.append(f"auth: {e}")
    if key and not http_dead:
        for attempt in range(3):
            _throttle()
            try:
                text = _post_http(model, system, prompt, key, timeout)
                _write_cache(path, {"model": model, "response": text})
                last_backend = "http"
                return text
            except Exception as e:  # noqa: BLE001 - collect and retry/fall back
                errors.append(f"http try{attempt}: {e}")
                http_dead = True
                # retry with backoff on 429/5xx (or no status at all); other
                # 4xx mean the endpoint rejects this client -> go to CLI now.
                m = re.search(r"http (\d+)", str(e))
                if m and int(m.group(1)) not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2 ** (attempt + 1))
    if quota_exceeded():
        _flag_quota()
    # Fallback: CLI subprocess (same cache entry), at most CLI_SLOTS at once machine-wide.
    if errors:
        print(f"llm: falling back to cli ({'; '.join(errors)[:300]})", file=sys.stderr, flush=True)
    _throttle()
    try:
        text = _run_cli(model, system, prompt, timeout, variant)
        _write_cache(path, {"model": model, "variant": variant, "response": text})
        last_backend = "cli"
        return text
    except Exception as e:  # noqa: BLE001 - report both paths in the error
        errors.append(f"cli: {e}")
    raise RuntimeError(f"llm.complete failed ({'; '.join(errors)})")


def complete_many(
    prompts: Sequence[str],
    model: str = "opencode/muse-spark-1.3-contributor-free",
    system: str | None = None,
    timeout: int = 300,
    max_workers: int = 6,
    return_exceptions: bool = False,
    variant: str | None = None,
) -> list[str | Exception]:
    """Complete many prompts concurrently, preserving input order.

    Each prompt runs through :func:`complete` (its own ``opencode``
    subprocess on a cache miss), so the disk cache behavior is identical
    to calling :func:`complete` serially. With ``return_exceptions``,
    per-prompt failures are returned in place (like ``asyncio.gather``)
    instead of raising.
    """
    if not prompts:
        return []
    if len(prompts) == 1 or max_workers <= 1:
        out: list[str | Exception] = []
        for p in prompts:
            try:
                out.append(complete(p, model=model, system=system, timeout=timeout, variant=variant))
            except Exception as e:
                if not return_exceptions:
                    raise
                out.append(e)
        return out
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        if return_exceptions:
            futs = [
                ex.submit(complete, p, model=model, system=system, timeout=timeout, variant=variant)
                for p in prompts
            ]
            res: list[str | Exception] = []
            for f in futs:
                try:
                    res.append(f.result())
                except Exception as e:  # noqa: BLE001 - captured in place when asked
                    res.append(e)
            return res
        return list(
            ex.map(
                lambda p: complete(p, model=model, system=system, timeout=timeout, variant=variant), prompts
            )
        )
