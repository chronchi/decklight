"""Run verifier jobs in forked children with a hard wall-clock limit.

Some card interactions make the core compute for minutes inside a single `OCG_DuelProcess` call (Kewl Tune: Synchro
checks over hand + field materials), which nothing in-process can interrupt. Each job therefore runs in its own forked
child (cheap: it inherits the parent's loaded Engine), and a child that overruns is killed and reported as failed.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from multiprocessing.connection import wait
from typing import Any

from decklight.demos.verify import Verdict


def _child(fn: Callable[[Any], Any], item: Any, conn) -> None:
    try:
        conn.send(fn(item))
    finally:
        conn.close()


def map_timeout(
    fn: Callable[[Any], Any],
    items: list,
    jobs: int,
    timeout: float,
    on_fail: Callable[[Any, str], Any],
) -> list:
    """`[fn(x) for x in items]` in input order, `jobs` forked children at a time; a child that runs longer than
    `timeout` seconds (or dies) yields `on_fail(item, "timeout" | "crashed")` instead. `fn` sees the parent's
    globals as they were at fork time (build the Engine before calling)."""
    ctx = mp.get_context("fork")
    results: list = [None] * len(items)
    pending = list(range(len(items)))
    running: dict[int, tuple] = {}  # index -> (process, receiving end, start time)
    while pending or running:
        while pending and len(running) < max(1, jobs):
            i = pending.pop(0)
            recv, send = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(fn, items[i], send), daemon=True)
            p.start()
            send.close()
            running[i] = (p, recv, time.monotonic())
        ready = wait([r for _, r, _ in running.values()], timeout=0.5)
        now = time.monotonic()
        for i, (p, recv, t0) in list(running.items()):
            if recv in ready:
                try:
                    results[i] = recv.recv()
                except EOFError:
                    results[i] = on_fail(items[i], "crashed")
                p.join()
            elif now - t0 > timeout:
                p.kill()
                p.join()
                results[i] = on_fail(items[i], "timeout")
            else:
                continue
            recv.close()
            del running[i]
    return results


def failed_verdict(line_id: str, seed: int, why: str, timeout: float) -> Verdict:
    """The Verdict for a job the pool killed (`why` = "timeout" | "crashed")."""
    return Verdict(
        line_id, False, why, [], [], seed, 0, [], f"{why}: wall-clock limit {timeout:.0f}s or worker died", []
    )
