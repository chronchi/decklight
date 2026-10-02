"""Run one capped job on vast.ai and pull runs/<name>/ back (see ygo_zero.infra.vast).

uv run python scripts/vast_job.py --name smoke --hours 0.4 --cmd 'bash scripts/vast_smoke.sh smoke'
uv run python scripts/vast_job.py --cleanup        # destroy any leftover ygo-zero instance
"""

import argparse

from ygo_zero.infra import vast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--cmd")
    ap.add_argument("--hours", type=float, default=0.5)
    ap.add_argument("--max-dph", type=float, default=0.15)
    ap.add_argument("--min-cpus", type=int, default=8)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()
    if args.cleanup:
        print("destroyed:", vast.destroy_strays())
        return
    spec = vast.JobSpec(args.name, args.cmd, args.hours, args.max_dph, args.min_cpus, gpu=not args.no_gpu)
    res = vast.run_job(spec)
    print(res)


if __name__ == "__main__":
    main()
