"""Warm vast.ai session box for remote verification (see ygo_zero.infra.vast).

The planner LLM calls stay on the laptop; only CPU-heavy engine verification batches go to the box:

    uv run python scripts/vast_session.py start [--min-cpus 24 --max-dph 0.15 --hours-cap 6]
    uv run python scripts/vast_session.py status
    uv run python scripts/vast_session.py push    # re-push code after local changes
    uv run python scripts/vast_session.py stop    # destroy the box, log the spend
"""

from __future__ import annotations

import argparse
import time

from ygo_zero.infra import vast


def cmd_start(args: argparse.Namespace) -> None:
    spec = vast.JobSpec(
        name=args.name,
        command="true",  # unused: a session runs no job, only setup
        hours_cap=args.hours_cap,
        max_dph=args.max_dph,
        min_cpus=args.min_cpus,
        gpu=not args.cpu,
        data_paths=tuple(args.data),
    )
    s = vast.start_session(spec, name=args.session)
    left = s["deadline"] - time.time()
    print(f"session on {s['host']}:{s['port']} (instance {s['instance_id']}, ${s['dph']}/h)")
    print(f"hours-cap {args.hours_cap}h (~{left / 3600:.1f}h left); stop it with `vast_session.py stop`")


def cmd_status(args: argparse.Namespace) -> None:
    s = vast.load_session(args.session)
    if s is None:
        print(f"no session {args.session!r}")
        return
    up = time.time() - float(s["started"])
    est = float(s["dph"]) * up / 3600
    left = float(s["deadline"]) - time.time()
    print(f"instance {s['instance_id']} ({s['offer']}) at {s['host']}:{s['port']}")
    print(f"up {up / 3600:.2f}h, est cost ${est:.3f} at ${s['dph']}/h")
    if left < 0:
        print(f"WARNING: past hours-cap by {-left / 3600:.2f}h; run `vast_session.py stop`")
    else:
        print(f"hours-cap in {left / 3600:.2f}h")


def cmd_push(args: argparse.Namespace) -> None:
    vast.repush_code(args.session)
    print("pushed")


def cmd_stop(args: argparse.Namespace) -> None:
    vast.end_session(args.session)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start", help="rent a box, push code, run setup, keep it warm")
    st.add_argument("--name", default="session")
    st.add_argument("--min-cpus", type=int, default=24)
    st.add_argument("--max-dph", type=float, default=0.15)
    st.add_argument("--hours-cap", type=float, default=6)
    # CPU-only listings are stale (create -> no_such_ask, NOTEBOOK 2026-10-02): rent a cheap many-core GPU host
    st.add_argument("--cpu", action="store_true", default=False, help="CPU-only listing (usually stale)")
    st.add_argument("--gpu", action="store_false", dest="cpu", help="cheap GPU host for its cores (default)")
    st.add_argument("--data", nargs="*", default=[], help="repo paths under data/ to ship with the code")
    st.add_argument("--session", default="lines", help="named session (default: lines)")
    st.set_defaults(fn=cmd_start)
    sst = sub.add_parser("status", help="show the session box, uptime and est. cost")
    sst.add_argument("--session", default="lines", help="named session (default: lines)")
    sst.set_defaults(fn=cmd_status)
    pu = sub.add_parser("push", help="re-push the working tree to the session box")
    pu.add_argument("--session", default="lines", help="named session (default: lines)")
    pu.set_defaults(fn=cmd_push)
    sp = sub.add_parser("stop", help="destroy the session box and log the spend")
    sp.add_argument("--session", default="lines", help="named session (default: lines)")
    sp.set_defaults(fn=cmd_stop)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
