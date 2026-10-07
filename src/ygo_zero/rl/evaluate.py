"""Concurrent evaluation for RL training (train_rl --eval-mode async).

The eval body used to run inside train_rl and pause training while 20+ CPU
cores sat idle. Now it lives here as :func:`run_eval` (used in-process for
``--eval-mode sync`` and iteration 0, and by ``scripts/eval_ckpt.py`` for
async subprocess evals), plus :class:`AsyncEvaluator`, which owns the
at-most-one-eval-subprocess bookkeeping so it can be unit-tested with fake
commands.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def format_eval_line(res: dict) -> str:
    """The ``[eval it N] ...`` line train_rl prints for an eval record."""
    it = res["iter"]
    parts = "  ".join(
        f"{k} {v['wr']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] n={v['n']} cut={v['cut']}"
        for k, v in res["eval"].items()
    )
    return f"[eval it {it}] {parts}  ({res['eval_seconds']} s)"


def run_eval(
    ckpt,
    init,
    emb,
    ev_names,
    hold,
    gauntlet_dir,
    eval_seeds: int,
    ga_seeds: int,
    ga_opps: int,
    no_ygoagent: bool,
    device,
    threads: int,
    it: int = 0,
    **kwargs,
) -> dict:
    """Greedy ckpt vs greedy BC (train + holdout decks) and vs ygo-agent.

    Returns exactly the record train_rl used to log inline:
    ``{"iter", "eval": {"vs_bc", "vs_bc_holdout", ["vs_ygoagent"]}, "eval_seconds"}``.
    ``iter`` is accepted as an alias for ``it``. Sections with no games get
    ``wr 0.0`` (callers that need a hard failure check ``n == 0`` themselves).
    """
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.duel import Engine
    from ygo_zero.eval.bc_policy import BCPolicy
    from ygo_zero.eval.match import play_match, ring_pairs, wilson

    if "iter" in kwargs:
        it = kwargs["iter"]
    t = time.time()
    hold = set(hold)
    ev_names = list(ev_names)
    ev_decks = [load_ydk(ROOT / "decks/pool" / f"{n}.ydk") for n in ev_names]
    g_decks = [load_ydk(p) for p in sorted(Path(gauntlet_dir).glob("*.ydk"))]

    engine = Engine()
    pol = BCPolicy(ckpt, emb, device=str(device))
    base_pol = BCPolicy(init, emb, device=str(device))
    ga = None
    if not no_ygoagent:
        from ygo_zero.eval.ygoagent import YgoAgentPolicy

        ga = YgoAgentPolicy(seed=3, workers=8)  # interpreter pool: same moves, ~3x faster (NOTEBOOK 2026-10-07)

    res: dict = {"iter": it, "eval": {}}
    pairs = [(i, j) for i in range(len(ev_decks)) for j in range(len(ev_decks)) if i != j]
    games = play_match(engine, ev_decks, pol, base_pol, pairs, range(eval_seeds),
                       num_envs=128, num_threads=threads)  # fmt: skip
    for name, gs in (("vs_bc", games), ("vs_bc_holdout", [g for g in games if ev_names[g.deck_a] in hold])):
        if len(gs) == 0:
            res["eval"][name] = {"n": 0, "wr": 0.0, "ci": [0.0, 1.0], "cut": 0}
            continue
        w = sum(g.winner_is_a for g in gs)
        lo, hi = wilson(w, len(gs))
        res["eval"][name] = {"n": len(gs), "wr": round(w / len(gs), 3), "ci": [round(lo, 3), round(hi, 3)],
                             "cut": sum(g.truncated for g in gs)}  # fmt: skip
    if ga is not None:
        ga._states.clear()  # LSTM states are keyed by schedule index, which restarts at 0 every match
        gp = ring_pairs(len(g_decks), ga_opps)
        games = play_match(engine, g_decks, pol, ga, gp, range(ga_seeds), num_envs=64, num_threads=threads)
        if len(games) == 0:
            res["eval"]["vs_ygoagent"] = {"n": 0, "wr": 0.0, "ci": [0.0, 1.0], "cut": 0}
        else:
            w = sum(g.winner_is_a for g in games)
            lo, hi = wilson(w, len(games))
            res["eval"]["vs_ygoagent"] = {"n": len(games), "wr": round(w / len(games), 3),
                                          "ci": [round(lo, 3), round(hi, 3)],
                                          "cut": sum(g.truncated for g in games)}  # fmt: skip
    res["eval_seconds"] = round(time.time() - t)
    return res


class AsyncEvaluator:
    """At most one eval subprocess at a time, backed by ``<out>/evals.jsonl``.

    ``cmd_builder(it, ckpt)`` returns the subprocess argv for that eval point.
    Completed records are read from the jsonl file and each is returned exactly
    once across :meth:`poll`/:meth:`wait`. A non-zero exit appends
    ``ALERT train_rl: eval it N failed (exit X), see eval-NNNN.txt`` to
    ``<out>/ALERT`` (and prints it) instead of returning a record.
    """

    def __init__(self, cmd_builder, out_dir) -> None:
        self.cmd_builder = cmd_builder
        self.out_dir = Path(out_dir)
        self.evals_path = self.out_dir / "evals.jsonl"
        self.proc: subprocess.Popen | None = None
        self.it: int | None = None
        self._logfile = None
        self._pending: list[dict] = []
        self._relayed = 0
        if self.evals_path.is_file():
            self._relayed = sum(1 for _ in self.evals_path.read_text().splitlines())

    def _collect_new(self) -> list[dict]:
        if not self.evals_path.is_file():
            return []
        lines = self.evals_path.read_text().splitlines()
        new = lines[self._relayed :]
        self._relayed = len(lines)
        recs = []
        for ln in new:
            ln = ln.strip()
            if ln:
                recs.append(json.loads(ln))
        return recs

    def _finish_current(self) -> None:
        """Reap a finished subprocess: ALERT on failure, stash records on success."""
        proc, it = self.proc, self.it
        assert proc is not None
        if self._logfile is not None:
            try:
                self._logfile.close()
            except OSError:
                pass
            self._logfile = None
        rc = proc.returncode
        self.proc, self.it = None, None
        if rc != 0:
            line = f"ALERT train_rl: eval it {it} failed (exit {rc}), see eval-{it:04d}.txt"
            print(line, flush=True)
            try:
                with open(self.out_dir / "ALERT", "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        else:
            self._pending.extend(self._collect_new())

    def maybe_start(self, it: int, ckpt) -> bool:
        """Start the eval subprocess for ``it``; False when the previous one runs."""
        if self.proc is not None:
            if self.proc.poll() is None:
                return False
            self._finish_current()
        cmd = self.cmd_builder(it, ckpt)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._logfile = open(self.out_dir / f"eval-{it:04d}.txt", "w")  # noqa: PTH123
        self.proc = subprocess.Popen(cmd, stdout=self._logfile, stderr=subprocess.STDOUT)
        self.it = it
        return True

    def poll(self) -> list[dict]:
        """Non-blocking reap: records of evals finished since the last call."""
        if self.proc is not None and self.proc.poll() is not None:
            self._finish_current()
        out, self._pending = self._pending, []
        return out

    def wait(self) -> list[dict]:
        """Block until the running eval (if any) finishes; return its records."""
        if self.proc is not None:
            self.proc.wait()
            self._finish_current()
        out, self._pending = self._pending, []
        return out

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
