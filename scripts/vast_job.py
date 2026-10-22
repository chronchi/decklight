"""Run one capped job on vast.ai and pull runs/<name>/ back (see decklight.infra.vast).

uv run python scripts/vast_job.py --name smoke --hours 0.4 --cmd 'bash scripts/vast_smoke.sh smoke'
uv run python scripts/vast_job.py --cleanup        # destroy any leftover ygo-zero instance
"""

import argparse
import signal
import sys

from decklight.infra import vast


def main() -> None:
    # SIGTERM (driver timeout) must run run_job's `finally`, which destroys the box; Python's default handler skips it
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--cmd")
    ap.add_argument("--hours", type=float, default=0.5)
    ap.add_argument("--max-dph", type=float, default=0.15)
    ap.add_argument("--min-cpus", type=int, default=8)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--num-gpus", type=int, default=1, help="GPUs per box (offer query num_gpus=N)")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--data", nargs="*", default=[], help="repo paths under data/ to ship with the code")
    ap.add_argument("--extra-query", default="", help='extra vast offer filter, e.g. "compute_cap>=800"')
    ap.add_argument("--min-gpu-ram", type=float, default=6)
    ap.add_argument("--setup", default=None, help="replace the box setup (default: scripts/vast_setup.sh)")
    args = ap.parse_args()
    if args.cleanup:
        print("destroyed:", vast.destroy_strays())
        return
    spec = vast.JobSpec(
        args.name,
        args.cmd,
        args.hours,
        args.max_dph,
        args.min_cpus,
        gpu=not args.no_gpu,
        num_gpus=args.num_gpus,
        data_paths=tuple(args.data),
        extra_query=args.extra_query,
        min_gpu_ram_gb=args.min_gpu_ram,
    )
    if args.setup is not None:
        spec.setup = args.setup
    res = vast.run_job(spec, max_hosts=5)
    print(res)


if __name__ == "__main__":
    main()
