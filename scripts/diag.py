"""Reusable policy diagnosis on a fixed window set (one checkpoint, or A vs B).

    uv run python scripts/diag.py --ckpt data/cache/forge_r1/best.pt --windows labels --max-starts 15 \
        --out /tmp/diag-smoke --sections behaviour,attention,interventions

Writes `report.md` plus JSON tables (behaviour, probes, attention, interventions) under `--out`. `--windows`:
labels (the label job), starts / starts-eval (training / held-out line starts), streams (recorded EDOPro client streams).
`--interpret` writes a DeepSeek brief from the report and launches scripts/delegate.sh (findings only).

Exit codes (fail loud): 2 = 0 windows; 3 = a probe target has one class; 4 = identical weights for --ckpt/--vs-ckpt;
5 = NaN in a table; 6 = more than 10% of windows failed to replay.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from decklight.diag.adapter import BCPolicyAdapter
from decklight.diag.attention import attention_table
from decklight.diag.behaviour import behaviour_table, pair_behaviour, score_windows
from decklight.diag.common import finite_tree
from decklight.diag.interventions import run_interventions
from decklight.diag.probes import ProbeConfig, compute_targets, run_probes
from decklight.diag.report import run_interpretation, write_report
from decklight.diag.windows import (
    build_label_windows,
    build_start_windows,
    build_stream_windows,
    cache_path,
    load_windows,
    materialize,
    save_windows,
)
from decklight.engine.duel import Engine
from decklight.eval.bc_policy import BCPolicy
from decklight.search.product import resolve_product_paths


def _progress(message: str) -> None:
    print(message, flush=True)


@contextmanager
def _stage(name: str):
    print(f"[{time.strftime('%H:%M:%S')}] stage {name} start", flush=True)
    t0 = time.monotonic()
    yield
    print(f"[{time.strftime('%H:%M:%S')}] stage {name} done in {time.monotonic() - t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--vs-ckpt", type=Path, default=None)
    ap.add_argument("--windows", choices=("labels", "starts", "starts-eval", "streams"), default="labels")
    ap.add_argument("--max-starts", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sections", default="behaviour,probes,attention,interventions")
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--model", default="opencode-go/deepseek-v4.1-flash")
    ap.add_argument("--emb", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--win", type=Path, default=ROOT / "data/cache/oppfilt_in", help="label-job records")
    ap.add_argument("--starts", type=Path, default=None, help="line starts file (labels / starts)")
    ap.add_argument("--stream-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--ref-ckpt", type=Path, default=None, help="policy that builds start/stream windows")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/diag/windows")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--min-windows", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=0, help="diagnose only the first N windows (smoke)")
    a = ap.parse_args()

    sections = [s.strip() for s in a.sections.split(",") if s.strip()]
    ckpt, _, emb = resolve_product_paths(a.ckpt, None, a.emb)
    ref_ckpt, _, _ = resolve_product_paths(a.ref_ckpt, None, emb)
    vs = a.vs_ckpt
    if vs is not None:
        from decklight.diag.windows import _audit

        LE = _audit("line_eval")
        if LE.sha(ckpt) == LE.sha(vs):
            print("ALERT diag: --ckpt and --vs-ckpt hold the same weights", flush=True)
            return 4

    engine = Engine()
    ref_pol = BCPolicy(ref_ckpt, emb, device=a.device, threads=a.threads)
    starts_path = a.starts
    if starts_path is None:
        starts_path = ROOT / (
            "data/cache/line_starts/starts-v2b.eval.jsonl" if a.windows == "starts-eval"
            else "data/cache/line_starts/starts-v2.jsonl"
        )  # fmt: skip
    key = {"source": a.windows, "max_starts": a.max_starts, "starts": str(starts_path), "win": str(a.win),
           "streams": str(a.stream_root), "ref": str(ref_ckpt)}  # fmt: skip
    cache = cache_path(a.cache_dir, a.windows, **key)
    if cache.is_file() and not a.no_cache:
        windows = load_windows(cache)
        print(f"windows: {len(windows)} (cache {cache})", flush=True)
    else:
        if a.windows == "labels":
            windows = build_label_windows(a.win, starts_path, max_starts=a.max_starts)
        elif a.windows in ("starts", "starts-eval"):
            src = "starts" if a.windows == "starts" else "starts-eval"
            windows = build_start_windows(engine, starts_path, ref_pol, source=src, max_starts=a.max_starts)
        else:
            windows = build_stream_windows(engine, a.stream_root, ref_pol, max_starts=a.max_starts)
        save_windows(cache, windows)
        print(f"windows: {len(windows)} (built -> {cache})", flush=True)
    with _stage("materialize"):
        windows = materialize(engine, windows, progress=_progress)
    if not a.no_cache and cache.is_file():
        save_windows(cache, windows)  # persist newly added metadata (e.g. activation names)
    if a.max_windows:
        windows = windows[: a.max_windows]
    if not windows:
        print("ALERT diag: 0 windows", flush=True)
        return 2
    print(f"windows: {len(windows)}, starts: {len({w.group for w in windows})}", flush=True)

    adapter_a = BCPolicyAdapter(ckpt, emb, device=a.device, threads=a.threads, name="A")
    adapter_b = None if vs is None else BCPolicyAdapter(vs, emb, device=a.device, threads=a.threads, name="B")
    with _stage("score A"):
        frames_a, bad_a = score_windows(engine, windows, adapter_a, progress=_progress)
    if adapter_b is not None:
        with _stage("score B"):
            frames_b, bad_b = score_windows(engine, windows, adapter_b, progress=_progress)
    else:
        frames_b, bad_b = [], 0
    bad = max(bad_a, bad_b)
    if bad > 0.1 * len(windows):
        print(f"ALERT diag: {bad}/{len(windows)} windows did not replay (replay diverged)", flush=True)
        return 6
    if not frames_a:
        print("ALERT diag: 0 windows replayed", flush=True)
        return 2

    names: dict[int, str] = {}
    try:
        from decklight.demos.names import CardNames

        names = CardNames().names
    except Exception as e:  # noqa: BLE001 - labels are cosmetic
        print(f"diag: card names unavailable ({e}); using codes", flush=True)

    sections_out: dict = {}
    targets: dict = {}
    target_info = {}
    if "behaviour" in sections:
        sections_out["behaviour"] = (
            pair_behaviour(frames_a, frames_b, names) if adapter_b is not None
            else behaviour_table(frames_a, names=names)
        )  # fmt: skip
    if "probes" in sections or "attention" in sections:
        with _stage("targets"):
            targets, target_info = compute_targets(engine, windows, ref_pol, progress=_progress)
    if "probes" in sections:
        cfg = ProbeConfig(permutations=a.permutations, bootstrap=a.bootstrap, min_windows=a.min_windows)
        def _pack(p):
            return {"tables": p.tables, "errors": p.errors, "targets": p.targets,
                    "supplied_columns": p.supplied_columns}

        with _stage("probes A"):
            pa = run_probes(frames_a, targets, cfg, progress=_progress)
        if adapter_b is not None:
            with _stage("probes B"):
                pb = run_probes(frames_b, targets, cfg, progress=_progress)
            sections_out["probes"] = {"A": _pack(pa), "B": _pack(pb)}
        else:
            sections_out["probes"] = _pack(pa)
    if "attention" in sections:
        with _stage("attention A"):
            aa = attention_table(frames_a, targets, progress=_progress)
        if adapter_b is not None:
            with _stage("attention B"):
                ab = attention_table(frames_b, targets, progress=_progress)
            sections_out["attention"] = {"A": aa, "B": ab}
        else:
            sections_out["attention"] = aa
    if "interventions" in sections:
        with _stage("interventions A"):
            ia = run_interventions(frames_a, adapter_a, progress=_progress)
        if adapter_b is not None:
            with _stage("interventions B"):
                ib = run_interventions(frames_b, adapter_b, progress=_progress)
            sections_out["interventions"] = {"A": ia, "B": ib}
        else:
            sections_out["interventions"] = ia

    for name, obj in sections_out.items():
        if not finite_tree(obj):
            print(f"ALERT diag: NaN/inf in the {name} table", flush=True)
            return 5

    meta = {
        "ckpt": str(ckpt),
        "vs_ckpt": None if vs is None else str(vs),
        "windows": a.windows,
        "windows_n": len(windows),
        "starts_n": len({w.group for w in windows}),
        "failed_replay": bad,
        "reference_policy": str(ref_ckpt),
        "targets": target_info,
        "sections": sections,
        "note": "Interventions are inference-only input edits; edited inputs can be off-distribution.",
    }
    with _stage("report"):
        report_md = write_report(a.out, meta, sections_out)
    print(f"report -> {report_md}", flush=True)
    for res in (sections_out.get("probes", {}),):
        if isinstance(res, dict) and res.get("errors"):
            print(f"ALERT diag: probe targets with one class: {res['errors']}", flush=True)
            return 3
        if isinstance(res, dict) and "A" in res and res["A"].get("errors"):
            print(f"ALERT diag: probe targets with one class: {res['A']['errors']}", flush=True)
            return 3
    if a.interpret:
        tag = run_interpretation(a.out, meta, report_md, model=a.model)
        print(f"interpretation launched: {tag}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
