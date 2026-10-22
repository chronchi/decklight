"""Turn a train_rl metrics.jsonl into PROGRESS lines for the per-deck fine-tune pilot.

    python scripts/audit/ft_progress.py <metrics.jsonl> --tag ft-BlueEyes [--seen N]
        [--ms-it 20 --ms-min 0.45]

For every new iteration (iter > --seen) divisible by 5, prints one line:
    PROGRESS <tag> it=<i> focus_wr_vs_bc=<last-5 mean> focus_share=<x> n=<sum last 5>[ kl=<v>]
kl is the kl_bc (or kl) value of iteration <i> when present. Missing file or no new
iterations: prints nothing, exits 0. With --ms-it, once iterations ms_it-4..ms_it are
all present, prints MILESTONE <tag> it=<ms> mean=<m> (or MILESTONE_FAIL ...) and exits
2 when the mean is below --ms-min (the caller alerts; it never kills the run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_recs(path: Path) -> dict[int, dict]:
    recs: dict[int, dict] = {}
    try:
        text = path.read_text()
    except OSError:
        return recs
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        if isinstance(r, dict) and isinstance(r.get("iter"), int) and "samples" in r:  # skip {"iter", "eval"} records
            recs[r["iter"]] = r
    return recs


def kl_of(rec: dict) -> float | None:
    for k in ("kl_bc", "kl"):
        v = rec.get(k)
        if isinstance(v, (int, float)):
            return v
    return None


def progress_lines(recs: dict[int, dict], tag: str, seen: int) -> list[str]:
    lines = []
    for i in sorted(recs):
        if i <= seen or i % 5 != 0:
            continue
        window = [recs.get(j) for j in range(i - 4, i + 1)]
        if any(w is None for w in window):
            continue
        assert all(w is not None for w in window)
        wrs = [float(w.get("focus_wr_vs_bc", 0.0)) for w in window]  # type: ignore[union-attr]
        share = window[-1].get("focus_seat_share", 0.0)  # type: ignore[union-attr]
        n = sum(int(w.get("focus_n_vs_bc", 0)) for w in window)  # type: ignore[union-attr]
        mean = sum(wrs) / len(wrs)
        kl = kl_of(window[-1])  # type: ignore[union-attr]
        line = (f"PROGRESS {tag} it={i} focus_wr_vs_bc={mean:.3f} "
                f"focus_share={share} n={n}")
        if kl is not None:
            line += f" kl={kl:.4f}"
        lines.append(line)
    return lines


def milestone_mean(recs: dict[int, dict], ms_it: int) -> float | None:
    window = [recs.get(j) for j in range(ms_it - 4, ms_it + 1)]
    if any(w is None for w in window):
        return None
    wrs = [float(w.get("focus_wr_vs_bc", 0.0)) for w in window]  # type: ignore[union-attr]
    return sum(wrs) / len(wrs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", type=Path)
    ap.add_argument("--tag", default="ft")
    ap.add_argument("--seen", type=int, default=0)
    ap.add_argument("--ms-it", type=int, default=0)
    ap.add_argument("--ms-min", type=float, default=0.45)
    args = ap.parse_args()
    recs = load_recs(args.metrics)
    for line in progress_lines(recs, args.tag, args.seen):
        print(line, flush=True)
    if args.ms_it > 0 and args.seen < args.ms_it:  # once: the watcher passes the last it= it printed as --seen
        m = milestone_mean(recs, args.ms_it)
        if m is not None:
            if m < args.ms_min:
                print(f"MILESTONE_FAIL {args.tag} it={args.ms_it} mean={m:.3f} < {args.ms_min}", flush=True)
                raise SystemExit(2)
            print(f"MILESTONE {args.tag} it={args.ms_it} mean={m:.3f}", flush=True)


if __name__ == "__main__":
    main()
