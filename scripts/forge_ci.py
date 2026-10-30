"""CI pack check for Forge submissions (runs without torch and without the engine).

Reads an issue body from the environment, downloads the single attached pack
tarball (if any), and validates it against a release fingerprint. Writes a
Markdown comment verdict; exit 0 carries the verdict, exit 1 an internal error.

Only stdlib + numpy + safetensors are used here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ATTACHMENT_RE = re.compile(
    r"https://github\.com/user-attachments/files/\d+/[A-Za-z0-9._-]{1,100}\.tar\.gz"
)
MAX_BYTES = 25 * 1024 * 1024
CHUNK = 1 << 20
MAX_LINES = 60
MAX_LINE = 200


def _host_allowed(host: str) -> bool:
    host = (host or "").lower()
    if host in ("github.com", "objects.githubusercontent.com"):
        return True
    return re.fullmatch(r"github-production-user-asset-[a-z0-9.-]+\.s3\.amazonaws\.com", host) is not None


class _TooLarge(Exception):
    pass


class _Unsafe(Exception):
    pass


class _RedirectGuard(urllib.request.HTTPRedirectHandler):
    """Refuse redirects to hosts outside the attachment allowlist."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlparse(newurl).hostname or ""
        if not _host_allowed(host):
            raise _Unsafe(f"redirect to {host!r} is not allowed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_attachment(url: str, dest: Path) -> str:
    """Stream *url* to *dest*, aborting above MAX_BYTES. Returns the final URL.

    Raises _TooLarge when the download exceeds the cap, _Unsafe when a
    redirect leaves the allowlist.
    """
    opener = urllib.request.build_opener(_RedirectGuard)
    req = urllib.request.Request(url, headers={"User-Agent": "decklight-forge-ci"})
    try:
        resp = opener.open(req, timeout=60)
    except _Unsafe:
        raise
    except Exception as e:
        raise _Unsafe(f"download failed: {type(e).__name__}") from e
    final_host = urllib.parse.urlparse(resp.geturl()).hostname or ""
    if not _host_allowed(final_host):
        raise _Unsafe(f"final URL host {final_host!r} is not allowed")
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise _TooLarge(f"attachment exceeds the {MAX_BYTES // (1 << 20)} MiB limit")
                f.write(chunk)
    finally:
        try:
            resp.close()
        except Exception:  # noqa: BLE001, S110 - best effort
            pass
    return resp.geturl()


def sanitise(line: str) -> str:
    """Make one message line safe for a public comment: no backticks, no @-mentions."""
    line = line.replace("`", "").replace("@", "(at)")
    line = "".join(ch for ch in line if unicodedata.category(ch) != "Cc")
    return line[:MAX_LINE]


def render_comment(header: str, details: list[str], pack_line: str | None = None) -> str:
    out = [header]
    if pack_line:
        out.append(pack_line)
    out.append("```")
    out.extend(sanitise(d) for d in details[:MAX_LINES])
    out.append("```")
    return "\n".join(out) + "\n"


def check_pack(archive_path: Path, fingerprint: Path) -> tuple[str, str | None]:
    """Validate one pack tarball; returns (comment, None) on a verdict, else raises."""
    from decklight.forge.pack import NAME_RE, PACK_RE, extract, validate_pack

    with tempfile.TemporaryDirectory(prefix="forge-ci-") as tmp:
        pack_dir = extract(archive_path, Path(tmp) / "pack")
        errors, warnings = validate_pack(pack_dir, fingerprint=fingerprint)
        pack_line = None
        try:
            manifest = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
            pid, arch = manifest.get("pack_id"), manifest.get("archetype")
            if (isinstance(pid, str) and isinstance(arch, str)
                    and PACK_RE.fullmatch(pid) and NAME_RE.fullmatch(arch)):
                pack_line = f"pack: {pid} ({arch})"
        except (OSError, ValueError):
            pack_line = None
        n = len(errors)
        header = "Validation: OK" if n == 0 else f"Validation: {n} errors"
        details = [f"ERROR {e}" for e in errors] + [f"WARN {w}" for w in warnings]
        return (render_comment(header, details, pack_line), None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--body-env", default=None, help="name of the env var holding the issue body")
    ap.add_argument("--file", default=None, help="local .tar.gz to check (skips the download)")
    ap.add_argument("--fingerprint", required=True, help="release fingerprint JSON")
    ap.add_argument("--out", required=True, help="where to write the Markdown comment")
    args = ap.parse_args()
    out = Path(args.out)

    def _fail(msg: str) -> int:
        out.write_text(render_comment("Validation: check failed to run", [msg]), encoding="utf-8")
        print(f"forge_ci: {msg}", file=sys.stderr)
        return 1

    try:
        fingerprint = Path(args.fingerprint)
        if not fingerprint.is_file():
            return _fail(f"fingerprint {fingerprint} missing")
        if args.file:
            comment, _ = check_pack(Path(args.file), fingerprint)
            out.write_text(comment, encoding="utf-8")
            return 0
        if not args.body_env:
            return _fail("no --body-env and no --file given")
        body = os.environ.get(args.body_env)
        if body is None:
            return _fail(f"env var {args.body_env} is not set")
        urls = ATTACHMENT_RE.findall(body)
        if not urls:
            out.write_text(
                "No pack attachment found (deck-list submissions are reviewed by hand).\n",
                encoding="utf-8",
            )
            return 0
        if len(urls) > 1:
            out.write_text(
                render_comment("Validation: 1 errors",
                               ["more than one pack attachment found; please submit one pack per issue"]),
                encoding="utf-8",
            )
            return 0
        with tempfile.TemporaryDirectory(prefix="forge-ci-dl-") as tmp:
            dest = Path(tmp) / "pack.tar.gz"
            try:
                download_attachment(urls[0], dest)
            except _TooLarge as e:
                out.write_text(render_comment("Validation: 1 errors",
                                              [f"pack attachment is too large (> 25 MiB), refusing to download: {e}"]),
                               encoding="utf-8")
                return 0
            except _Unsafe as e:
                out.write_text(render_comment("Validation: 1 errors", [f"download refused: {e}"]),
                               encoding="utf-8")
                return 0
            try:
                comment, _ = check_pack(dest, fingerprint)
            except ValueError as e:
                comment = render_comment("Validation: 1 errors", [f"archive rejected: {e}"])
            out.write_text(comment, encoding="utf-8")
            return 0
    except Exception as e:  # noqa: BLE001 - an internal error still writes a comment, then exit 1
        try:
            out.write_text(render_comment("Validation: check failed to run",
                                          [f"internal error: {type(e).__name__}"]),
                           encoding="utf-8")
        except OSError:
            pass
        print(f"forge_ci: internal error: {e!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
