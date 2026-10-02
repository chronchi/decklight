"""Latency / quality benchmark of OpenCode Go models on our real extraction prompts (no cache).

    uv run python scripts/bench_llm.py --models opencode-go/muse-spark-1.3-contributor opencode-go/deepseek-v4.1-flash \
        opencode-go/longcat-2.5-preview-free --reps 2 --parallel 4

Quality = step-card-sequence overlap of each extracted line with the best-matching line of the reference model's
answer (default: muse's cached answer for the same prompt), plus JSON validity and line counts.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ygo_zero.demos import llm
from ygo_zero.demos.extract import SYSTEM, _parse_json, _prompt, split_header, split_sections

GUIDES = [  # (deck, guide file, section title substring)
    ("blue-eyes", "data/guides/blue-eyes/02_game8.txt", None),
    ("salamangreat", "data/guides/salamangreat/01_roadoftheking-vol1.txt", None),
]


def prompts() -> list[tuple[str, str]]:
    out = []
    for deck, path, want in GUIDES:
        _, body = split_header(Path(path).read_text(errors="replace"))
        for title, chunk in split_sections(body):
            if (want is None or want in title) and (
                "combo" in title.lower() or "combo" in chunk[:400].lower()
            ):
                out.append((f"{deck}:{title[:40]}", _prompt(title, chunk, deck)))
                break
    return out


def seq(line: dict) -> list[str]:
    return [str(s.get("card", "")).lower() for s in line.get("steps", [])]


def overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def quality(lines: list[dict], ref: list[dict]) -> float:
    if not lines or not ref:
        return 0.0
    return statistics.mean(max(overlap(seq(r), seq(x)) for x in lines) for r in ref)


def call(model: str, prompt: str) -> tuple[float, str]:
    t0 = time.time()
    try:
        text = llm._run_cli(model, SYSTEM, prompt, timeout=600)
    except Exception as e:  # noqa: BLE001 - a failed call is a data point
        text = f"ERROR {e!r}"
    return time.time() - t0, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ref", default="opencode-go/muse-spark-1.3-contributor")
    ap.add_argument("--out", type=Path, default=Path("data/cache/bench_llm.json"))
    args = ap.parse_args()
    ps = prompts()
    print("prompts:", [(name, len(p)) for name, p in ps])
    refs = {}
    for name, p in ps:
        cached = llm.CACHE_DIR / f"{llm._cache_key(args.ref, SYSTEM, p)}.json"
        refs[name] = _parse_json(json.loads(cached.read_text())["response"]) if cached.exists() else []
        print(
            f"reference {name}: {len(refs[name])} lines from cache"
            if refs[name]
            else f"reference {name}: none cached"
        )
    results = {}
    for model in args.models:
        rows = []
        for name, p in ps:
            for _ in range(args.reps):
                dt, text = call(model, p)
                lines = _parse_json(text)
                q = quality(lines, refs[name]) if refs[name] else None
                rows.append({"prompt": name, "seconds": round(dt, 1), "chars": len(text), "json_ok": bool(lines),
                             "lines": len(lines), "quality": None if q is None else round(q, 2)})  # fmt: skip
                print(model, rows[-1], flush=True)
        name, p = ps[0]
        t0 = time.time()
        with ThreadPoolExecutor(args.parallel) as ex:
            par = list(ex.map(call, [model] * args.parallel, [p] * args.parallel))
        wall = time.time() - t0
        par_ok = sum(bool(_parse_json(t)) for _, t in par)
        print(
            f"{model}: {args.parallel} parallel calls wall {wall:.0f}s (each {[round(d) for d, _ in par]}), json_ok {par_ok}"
        )
        results[model] = {"serial": rows, "parallel": {"n": args.parallel, "wall": round(wall, 1),
                          "each": [round(d, 1) for d, _ in par], "json_ok": par_ok}}  # fmt: skip
    args.out.write_text(json.dumps(results, indent=1))
    print("\nSUMMARY (median seconds / json ok / mean quality vs ref):")
    for model, r in results.items():
        s = r["serial"]
        qs = [x["quality"] for x in s if x["quality"] is not None]
        print(f"  {model:45s} median {statistics.median(x['seconds'] for x in s):6.1f}s  "
              f"json {sum(x['json_ok'] for x in s)}/{len(s)}  quality {statistics.mean(qs) if qs else float('nan'):.2f}  "
              f"parallel x{r['parallel']['n']} wall {r['parallel']['wall']}s")  # fmt: skip


if __name__ == "__main__":
    main()
