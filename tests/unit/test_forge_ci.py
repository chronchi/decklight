"""Forge CI: issue-body packs, download caps, sanitised comments, workflow pinning (< 10 s, no network)."""

import importlib.util
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

import torch

from decklight.forge.pack import archive, build_pack, fingerprint_of

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("forge_ci_mod", ROOT / "scripts" / "forge_ci.py")
assert SPEC is not None and SPEC.loader is not None
fci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fci)

ARGS = {"d": 8, "layers": 1, "heads": 2, "v3": False, "hist": 0, "known": False}
MAIN = list(range(10000001, 10000041))
EXTRA = [20000001, 20000002]
SIDE = [30000001]


def _write_ydk(path: Path, main, extra=(), side=()) -> None:
    lines = ["#created by test", "#main", *map(str, main), "#extra", *map(str, extra),
             "!side", *map(str, side)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup(tmp_path: Path) -> dict:
    rel = tmp_path / "release"
    rel.mkdir(parents=True)
    torch.save({"w1": torch.zeros(4, 4), "w2": torch.ones(3)}, rel / "best.pt")
    (rel / "metrics.jsonl").write_text(json.dumps({"args": ARGS}) + "\n", encoding="utf-8")
    ft = tmp_path / "ft"
    (ft / "policy").mkdir(parents=True)
    (ft / "rl").mkdir(parents=True)
    torch.save({"w1": torch.zeros(4, 4) + 0.02, "w2": torch.ones(3)}, ft / "policy" / "best.pt")
    (ft / "rl" / "metrics.jsonl").write_text(json.dumps({"iter": 3}) + "\n", encoding="utf-8")
    emb = tmp_path / "emb.npz"
    emb.write_bytes(b"emb")
    db = sqlite3.connect(tmp_path / "cards.cdb")
    try:
        db.execute("CREATE TABLE datas (id INTEGER PRIMARY KEY)")
        db.executemany("INSERT INTO datas (id) VALUES (?)", [(c,) for c in MAIN + EXTRA + SIDE])
        db.commit()
    finally:
        db.close()
    d1, d2 = tmp_path / "deck1.ydk", tmp_path / "deck2.ydk"
    _write_ydk(d1, MAIN, EXTRA, SIDE)
    _write_ydk(d2, MAIN, EXTRA, SIDE)
    return {"release": rel / "best.pt", "ft": ft, "emb": emb, "cdb": tmp_path / "cards.cdb",
            "deck1": d1, "deck2": d2}


def _build(ctx: dict, out: Path) -> Path:
    return build_pack(
        ft_policy=ctx["ft"] / "policy" / "best.pt", train_metrics=ctx["ft"] / "rl" / "metrics.jsonl",
        release=ctx["release"], release_name="r0", archetype="elfnote",
        train_decks=[ctx["deck1"]], heldout_decks=[ctx["deck2"]], out=out,
        contributor={"name": "t"}, license="CC-BY-4.0",
        train={"config": {"iterations": 3}, "seed": 0, "iterations": 3, "stop": "fixed"},
        emb=ctx["emb"], cards_db=ctx["cdb"], core="")


def _pack_and_fp(tmp_path: Path) -> tuple[Path, Path]:
    ctx = _setup(tmp_path)
    tgz = archive(_build(ctx, tmp_path / "pack"), tmp_path / "dist")
    fp = tmp_path / "fp.json"
    fp.write_text(json.dumps(fingerprint_of(ctx["release"], "r0")), encoding="utf-8")
    return tgz, fp


def _run_main(monkeypatch, tmp_path: Path, fp: Path, env_body: str | None) -> tuple[int, str]:
    out = tmp_path / "comment.md"
    argv = ["forge_ci.py", "--fingerprint", str(fp), "--out", str(out)]
    if env_body is None:
        argv += ["--file", str(_pack_and_fp(tmp_path)[0])]
    else:
        argv += ["--body-env", "ISSUE_BODY"]
        monkeypatch.setenv("ISSUE_BODY", env_body)
    monkeypatch.setattr(sys, "argv", argv)
    return fci.main(), out.read_text(encoding="utf-8")


def test_url_extraction(monkeypatch, tmp_path):
    tgz, fp = _pack_and_fp(tmp_path)
    seen: list[str] = []

    def fake_download(url: str, dest: Path) -> str:
        seen.append(url)
        shutil.copy(tgz, dest)
        return url

    monkeypatch.setattr(fci, "download_attachment", fake_download)
    good = "https://github.com/user-attachments/files/12345/mypack.tar.gz"
    rc, comment = _run_main(monkeypatch, tmp_path, fp, f"please take {good}\n")
    assert rc == 0 and seen == [good] and comment.startswith("Validation: OK")

    seen.clear()
    rc, comment = _run_main(monkeypatch, tmp_path, fp,
                            "evil https://evil.example/x.tar.gz\n"
                            "raw https://github.com/foo/bar/raw/abc123/x.tar.gz\n")
    assert rc == 0 and seen == [] and "No pack attachment found" in comment

    seen.clear()
    rc, comment = _run_main(monkeypatch, tmp_path, fp,
                            f"{good}\nhttps://github.com/user-attachments/files/99/other.tar.gz\n")
    assert rc == 0 and seen == [] and "one pack per issue" in comment


def test_valid_pack_ok(monkeypatch, tmp_path):
    tgz, fp = _pack_and_fp(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["forge_ci.py", "--file", str(tgz), "--fingerprint", str(fp),
                         "--out", str(tmp_path / "c.md")])
    assert fci.main() == 0
    assert (tmp_path / "c.md").read_text(encoding="utf-8").startswith("Validation: OK")


def test_bad_payload_sanitised(monkeypatch, tmp_path):
    ctx = _setup(tmp_path)
    pack = _build(ctx, tmp_path / "pack")
    manifest = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    manifest["contributor"] = {"name": "evil`x@y"}
    (pack / "pack.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    tgz = archive(pack, tmp_path / "dist")
    fp = tmp_path / "fp.json"
    fp.write_text(json.dumps(fingerprint_of(ctx["release"], "r0")), encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["forge_ci.py", "--file", str(tgz), "--fingerprint", str(fp),
                         "--out", str(tmp_path / "c.md")])
    assert fci.main() == 0
    comment = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert comment.startswith("Validation: 1 errors")
    inner = comment.split("```")[1]
    assert "`" not in inner and "@" not in comment


def test_download_cap(monkeypatch, tmp_path):
    tgz, fp = _pack_and_fp(tmp_path)
    _ = tgz
    size = 26 * 1024 * 1024

    class _FakeResp:
        def __init__(self) -> None:
            self.left = size
            self.url = "https://github.com/user-attachments/files/1/big.tar.gz"

        def read(self, n: int = -1) -> bytes:
            if self.left <= 0:
                return b""
            n = min(n if n > 0 else self.left, self.left)
            self.left -= n
            return b"x" * n

        def geturl(self) -> str:
            return self.url

        def close(self) -> None:
            pass

    class _FakeOpener:
        def open(self, req, timeout=None) -> _FakeResp:
            return _FakeResp()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **k: _FakeOpener())
    rc, comment = _run_main(monkeypatch, tmp_path, fp,
                            "big https://github.com/user-attachments/files/1/big.tar.gz\n")
    assert rc == 0 and "too large" in comment


def test_workflow_pinned():
    import yaml

    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "validate-pack.yml").read_text())
    assert wf.get("on", wf.get(True)) == {"issues": {"types": ["labeled"]}}
    assert wf["permissions"] == {"contents": "read", "issues": "write"}
    steps = wf["jobs"]["validate"]["steps"]
    assert steps, "no steps in validate job"
    for s in steps:
        if "uses" in s:
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", s["uses"]), f"unpinned action: {s['uses']}"
        if "run" in s:
            assert "${{" not in s["run"], f"issue field interpolated in run: {s['run'][:60]}"


def test_no_torch_fingerprint_validate(tmp_path):
    tgz, fp = _pack_and_fp(tmp_path)
    script = (
        "import sys; sys.modules['torch'] = None; sys.path.insert(0, 'src');"
        "from pathlib import Path;"
        "from decklight.forge.pack import extract, validate_pack;"
        f"pack = extract(Path(r'{tgz}'), Path(r'{tmp_path / 'x'}'));"
        f"errors, warnings = validate_pack(pack, fingerprint=Path(r'{fp}'));"
        "assert errors == [], errors; print('fingerprint-validate ok')"
    )
    r = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
                       timeout=120, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "fingerprint-validate ok" in r.stdout
