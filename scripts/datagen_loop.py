"""Overnight data-generation driver (laptop, unsupervised).

Every 60 s one `tick()` maintains: the local LLM server, a warm vast session box
for remote verification, `--planners` muse `plan_lines.py --remote` children (one
per pool deck, fewest verified records first), `--games` `play_games.py` children,
and a BC scaling run every `--bc-every-h` hours. All progress lands in
`data/cache/runs/datagen.log`.

Laptop rule: this process pins itself to cores 0-3 with nice 19 at start, so
children inherit. The OpenCode key never leaves the laptop.

Usage:
    uv run python scripts/datagen_loop.py [--planners 3 --games 3 --variant low]
"""

from __future__ import annotations

import argparse
import ast
import datetime
import glob
import json
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ygo_zero.demos import llm
from ygo_zero.infra import vast

ROOT = Path(__file__).resolve().parents[1]
RUNS_CACHE = ROOT / "data" / "cache" / "runs"
DATAGEN_LOG = RUNS_CACHE / "datagen.log"
ALLOWED_FILE = RUNS_CACHE / "vast-allowed.txt"
STOP_FILE = RUNS_CACHE / "STOP_DATAGEN"
SCALING_TSV = RUNS_CACHE / "bc_scaling.tsv"
LLM_SERVER_URL = "http://127.0.0.1:4097"
PROBE_MODEL = "opencode/muse-spark-1.3-contributor-free"

TICK_S = 60
SUMMARY_S = 30 * 60
SESSION_RETRY_S = 30 * 60
REPO_BUDGET_BYTES = 45 * 1024**3
KILL_GRACE_S = 30

CHILD_LIMITS = {"judge": 2 * 3600, "planner": 3 * 3600, "improver": 3 * 3600, "expander": 2 * 3600, "game": 2 * 3600, "session": 30 * 60, "bc": 3 * 3600}


# ---------------------------------------------------------------- pure helpers


def pool_decks(root: Path = ROOT) -> list[str]:
    """Sorted pool deck slugs (`decks/pool/*.ydk`)."""
    return sorted(p.stem for p in (root / "decks" / "pool").glob("*.ydk"))


def holdout_decks(root: Path = ROOT) -> set[str]:
    """Decks never used for games/BC training: HOLDOUT + kewl-tune."""
    out = {"kewl-tune"}
    f = root / "decks" / "pool" / "HOLDOUT"
    if f.is_file():
        out.update(f.read_text().split())
    return out


def eligible_game_decks(root: Path = ROOT) -> list[str]:
    """Pool decks allowed in self-play games (no holdout, no kewl-tune)."""
    hold = holdout_decks(root)
    return sorted(d for d in pool_decks(root) if d not in hold)


def _verified_files(root: Path, deck: str) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    demos = root / "data" / "demos"
    for pat in (f"*/verified/{deck}*.jsonl", f"verified/{deck}*.jsonl"):
        for f in sorted(demos.glob(pat)):
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def count_verified_records(root: Path, deck: str) -> int:
    """Verified records for `deck` summed over every demos source."""
    n = 0
    for f in _verified_files(root, deck):
        try:
            with open(f) as fh:
                n += sum(1 for line in fh if line.strip())
        except OSError:
            continue
    return n


def verified_counts(root: Path = ROOT) -> dict[str, int]:
    return {d: count_verified_records(root, d) for d in pool_decks(root)}


def choose_planner_deck(root: Path, running: set[str] | frozenset = frozenset()) -> str | None:
    """Pool deck with the fewest verified records, skipping running planners."""
    best: str | None = None
    best_n = 0
    for d in pool_decks(root):
        if d in running:
            continue
        n = count_verified_records(root, d)
        if best is None or n < best_n:
            best, best_n = d, n
    return best


def next_planner_seed(root: Path, deck: str) -> int:
    """1 + the largest N with `data/demos/planned-sN/report/<deck>.json`, else 1."""
    best = 0
    for f in (root / "data" / "demos").glob(f"planned-s*/report/{deck}.json"):
        m = re.fullmatch(r"planned-s(\d+)", f.parents[1].name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def planner_command(deck: str, seed: int, variant: str | None, session: str = "lines") -> list[str]:
    cmd = ["uv", "run", "--no-sync", "python", "scripts/plan_lines.py",
           "--deck", deck, "--seed", str(seed), "--hands", "20", "--repairs", "2",
           "--workers", "6", "--remote", "--session", session, "--candidates", "3"]
    if variant is not None:
        cmd += ["--variant", variant]
    cmd += ["--out-dir", f"data/demos/planned-s{seed}"]
    return cmd


def choose_improve_job(root: Path, running: set[tuple[str, str]] = frozenset()) -> tuple[str, str] | None:
    """(src dir, deck) with the most verified planner lines not yet tried by improve_lines.py, or None."""
    best, best_n = None, 0
    for vf in sorted((root / "data" / "demos").glob("planned*/verified/*.jsonl")):
        src, deck = vf.parent.parent, vf.stem
        key = (str(src.relative_to(root)), deck)
        if key in running:
            continue
        ids = {json.loads(x)["line_id"] for x in vf.read_text().splitlines() if x.strip()}
        rp = src / "improved" / "report" / f"{deck}.json"
        tried = {r.get("old_id") for r in json.loads(rp.read_text())} if rp.is_file() else set()
        n = len(ids - tried)
        if n > best_n:
            best, best_n = key, n
    return best


def improve_command(src: str, deck: str, session: str = "lines") -> list[str]:
    # the quality pass keeps muse's default reasoning effort (NOTEBOOK 2026-10-03: 7/9 kept vs 2/6 at minimal)
    return ["uv", "run", "--no-sync", "python", "scripts/improve_lines.py", "--deck", deck, "--src", src,
            "--remote", "--workers", "6", "--session", session]


def expander_verified_files(root: Path = ROOT) -> list[Path]:
    """Candidate verified files for hand-variant expansion (claude + planned, incl. improved)."""
    demos = root / "data" / "demos"
    files: list[Path] = []
    for pat in ("claude/verified/*.jsonl", "planned*/verified/*.jsonl",
                "planned*/improved/verified/*.jsonl"):
        files.extend(sorted(demos.glob(pat)))
    return sorted(set(files))


def expander_target(vf: Path) -> tuple[Path, Path, list[str]]:
    """(lines_dir, out_dir, skip_done_globs) for a verified file `vf` (same absoluteness as `vf`).

    `<src>` is the dir holding `verified/` (`<src>/improved` for improved files); expansions live in
    `<src>/expanded-<deck>/`, so planned and improved lines never share a done-set (an improved `X+imp`
    expands to `X#hV`, the same base id a planned expansion of `X` would use).
    """
    src = vf.parent.parent
    deck = vf.stem
    return src / "lines", src / f"expanded-{deck}", [str(src / "expanded*" / "*.jsonl")]


def _expanded_done_ids(patterns: list[str]) -> set[str]:
    """Base line ids (`<id>` of `<id>#hV`) already expanded in files matching the globs."""
    done: set[str] = set()
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            p = Path(f)
            if not p.is_file():
                continue
            try:
                text = p.read_text()
            except OSError:
                continue
            for raw in text.splitlines():
                if not raw.strip():
                    continue
                try:
                    lid = json.loads(raw)["line_id"]
                except (json.JSONDecodeError, KeyError):
                    continue
                done.add(str(lid).split("#")[0])
    return done


def expander_backlog(vf: Path) -> int:
    """Verified records in `vf` with no expansion yet (mirrors expand_lines' --skip-done check)."""
    _, _, skips = expander_target(vf)
    done = _expanded_done_ids(skips)
    try:
        lines = vf.read_text().splitlines()
    except OSError:
        return 0
    n = 0
    for raw in lines:
        if not raw.strip():
            continue
        try:
            lid = str(json.loads(raw)["line_id"])
        except (json.JSONDecodeError, KeyError):
            continue
        if lid.split("#")[0] not in done and lid.split("+")[0] not in done:
            n += 1
    return n


def choose_expand_job(root: Path, running: set[str] | frozenset = frozenset()) -> Path | None:
    """Verified file with the most unexpanded records, skipping running expanders and empty backlogs."""
    best, best_n = None, 0
    for vf in expander_verified_files(root):
        try:
            key = str(vf.relative_to(root))
        except ValueError:
            key = str(vf)
        if key in running or str(vf) in running:
            continue
        n = expander_backlog(vf)
        if n > best_n:
            best, best_n = vf, n
    return best


def expand_command(vf: Path, lines_dir: Path, out_dir: Path, skips: list[str], session: str = "lines") -> list[str]:
    return ["uv", "run", "--no-sync", "python", "scripts/expand_lines.py",
            "--verified", str(vf), "--lines-dir", str(lines_dir), "--out", str(out_dir),
            "--variants", "3", "--skip-done", *skips, "--remote", "--session", session]  # mix exp: expansions add little


def game_command(a: str, b: str, seed: int, variant: str | None, remote: bool = False) -> list[str]:
    cmd = ["uv", "run", "--no-sync", "python", "scripts/play_games.py",
           "--pairs", f"{a}:{b}", "--games", "1", "--seed", str(seed), "--max-turns", "12"]
    # turn plans keep muse's default effort (realism); interruption asks run at minimal (play_games default)
    if remote:
        cmd += ["--remote", "--budget", "20000"]  # complex decks (Tearlaments) exhaust 4000 nodes every turn
    cmd += ["--opening-frac", "0.5"]  # half the games start from a verified turn-1 board (going-second data)
    cmd += ["--out", "data/demos/games"]
    return cmd


def judge_command() -> list[str]:
    return ["uv", "run", "--no-sync", "python", "scripts/judge_games.py", "--workers", "4", "--limit", "40"]


def unjudged_games(root: Path = ROOT) -> int:
    """Games recorded minus games judged (line counts; cheap)."""
    def count(d: Path) -> int:
        return sum(sum(1 for x in f.read_text().splitlines() if x.strip()) for f in d.glob("*.jsonl")) if d.is_dir() else 0
    return count(root / "data" / "demos" / "games") - count(root / "data" / "demos" / "games_judged")


def session_label(name: str, hhmm: str) -> str:
    """vast instance label suffix: lines -> verify-HHMM, games -> games-HHMM, linesK -> linesK-HHMM."""
    return f"verify-{hhmm}" if name == "lines" else f"{name}-{hhmm}"


def line_session_names(n: int) -> list[str]:
    return ["lines"] + [f"lines{k}" for k in range(2, n + 1)]


def session_command(hhmm: str, name: str = "lines") -> list[str]:
    label = session_label(name, hhmm)
    cmd = ["uv", "run", "--no-sync", "python", "scripts/vast_session.py",
           "start", "--name", label, "--hours-cap", "4"]
    if name != "lines":
        cmd += ["--session", name]
    return cmd


def bc_command(name: str) -> list[str]:
    return ["uv", "run", "--no-sync", "python", "scripts/vast_job.py",
            "--name", name, "--cmd", f"bash scripts/vast_bc_scale.sh {name} 12",
            "--hours", "3", "--max-dph", "0.5", "--min-cpus", "16",
            "--min-gpu-ram", "12", "--extra-query", "compute_cap>=800",
            "--data", "data/demos", "data/cards/emb", "decks"]


def count_games_per_deck(root: Path) -> dict[str, int]:
    """Games played per deck: files in `data/demos/games/*.jsonl` naming the deck in `decks`."""
    counts: dict[str, int] = {}
    for f in (root / "data" / "demos" / "games").glob("*.jsonl"):
        try:
            lines = f.read_text().splitlines()
        except OSError:
            continue
        for raw in lines:
            if not raw.strip():
                continue
            try:
                decks = json.loads(raw).get("decks", [])
            except json.JSONDecodeError:
                continue
            for d in decks:
                counts[str(d)] = counts.get(str(d), 0) + 1
    return counts


def game_weights(decks: list[str], counts: dict[str, int]) -> dict[str, float]:
    """1/(1 + games played): unplayed decks are picked first, played ones stay in rotation."""
    return {d: 1.0 / (1.0 + counts.get(d, 0)) for d in decks}


def pick_game_pair(decks: list[str], counts: dict[str, int], rng) -> tuple[str, str]:
    """Two distinct decks, weighted by `game_weights`; order (first player) is random."""
    if len(decks) < 2:
        raise ValueError(f"need >= 2 eligible decks, got {decks}")
    w = game_weights(decks, counts)
    a = rng.choices(decks, weights=[w[d] for d in decks])[0]
    rest = [d for d in decks if d != a]
    rw = game_weights(rest, counts)
    b = rng.choices(rest, weights=[rw[d] for d in rest])[0]
    return (b, a) if rng.random() < 0.5 else (a, b)


def flat_to_path(flat: str) -> Path:
    """`vast_bc_scale.sh` flat dir name -> repo path (`/` was replaced by `_`; slugs never contain `_`)."""
    return Path(*flat.split("_"))


def path_to_flat(path: str | Path) -> str:
    return str(path).replace("/", "_")


def parse_train_first_line(line: str) -> tuple[dict, list]:
    """First line of `runs/<name>/train.txt`: `{...counts...} decks: [...]`."""
    head, _, tail = line.partition(" decks:")
    return ast.literal_eval(head.strip()), ast.literal_eval(tail.strip())


def metrics_best(metrics_path: Path) -> dict:
    """Best val/holdout acc and their epochs over `bc/metrics.jsonl`."""
    best: dict = {"val_acc": None, "val_epoch": None, "holdout_acc": None, "holdout_epoch": None}
    if not metrics_path.is_file():
        return best
    for raw in metrics_path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for key in ("val", "holdout"):
            sub = rec.get(key)
            if (
                isinstance(sub, dict)
                and isinstance(sub.get("acc"), (int, float))
                and (best[f"{key}_acc"] is None or sub["acc"] > best[f"{key}_acc"])
            ):
                best[f"{key}_acc"] = sub["acc"]
                best[f"{key}_epoch"] = rec.get("epoch")
    return best


# ---------------------------------------------------------------- driver


@dataclass
class Child:
    kind: str  # planner | improver | expander | game | session | bc
    proc: subprocess.Popen
    log_file: object = None
    log_path: Path = field(default_factory=Path)
    started: float = 0.0
    term_sent: float | None = None
    desc: str = ""
    name: str = ""  # run name for bc jobs
    deck: str = ""  # deck for planner children


class DatagenLoop:
    def __init__(self, args) -> None:
        self.args = args
        self.children: list[Child] = []
        self.rng = random.Random()
        self.start_t = time.time()
        self.bc_next_due = self.start_t + 3600  # first BC run 1 h after start
        self.failed_starts: dict[str, int] = {}
        self.session_retry_after: dict[str, float] = {}
        self.last_summary = 0.0
        self.over_budget = False
        self.quota_since: float | None = None  # LLM quota pause start (llm.QUOTA_MARKER)
        self.quota_probe_at = 0.0

    # -- logging ------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        RUNS_CACHE.mkdir(parents=True, exist_ok=True)
        with open(DATAGEN_LOG, "a") as f:
            f.write(line + "\n")

    # -- children -----------------------------------------------------
    def _launch(self, kind: str, cmd: list[str], log_path: Path, **kw) -> Child:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "ab")  # noqa: SIM115 - handle stays open for the child's lifetime
        env = dict(os.environ)
        env["YGO_OPENCODE_SERVER"] = LLM_SERVER_URL
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env,
                                preexec_fn=_oom_first)  # noqa: PLW1509 - the driver is single-threaded
        child = Child(kind=kind, proc=proc, log_file=fh, log_path=log_path,
                      started=time.time(), **kw)
        self.children.append(child)
        self.log(f"start {kind} {child.desc} pid={proc.pid} log={log_path}")
        return child

    def _reap(self) -> None:
        for child in list(self.children):
            rc = child.proc.poll()
            if rc is None:
                continue
            try:
                child.log_file.close()
            except OSError:
                pass
            self.children.remove(child)
            self.log(f"exit {child.kind} {child.desc} rc={rc}")
            if child.kind == "session":
                sname = child.name or "lines"
                if rc == 0:
                    self.failed_starts[sname] = 0
                else:
                    self.failed_starts[sname] = self.failed_starts.get(sname, 0) + 1
                    if self.failed_starts[sname] >= 2:
                        self.session_retry_after[sname] = time.time() + SESSION_RETRY_S
                        self.log(f"session {sname}: 2 consecutive failed starts, retry after 30 min")
            elif child.kind == "bc":
                self._finish_bc(child)
                if not (ROOT / "runs" / str(child.name) / "bc" / "metrics.jsonl").is_file():
                    # host never ready / job cap hit: retry soon instead of waiting a whole slot (2 of 4 failed so)
                    self.bc_next_due = min(self.bc_next_due, time.time() + 900)
                    self.log(f"bc {child.name}: no metrics, retry in 15 min")

    def _enforce_limits(self, now: float) -> None:
        for child in self.children:
            limit = CHILD_LIMITS[child.kind]
            if now - child.started < limit:
                continue
            if child.term_sent is None:
                child.proc.terminate()
                child.term_sent = now
                self.log(f"timeout {child.kind} {child.desc}: SIGTERM after {limit / 3600:.1f}h")
            elif now - child.term_sent > KILL_GRACE_S:
                child.proc.kill()
                self.log(f"timeout {child.kind} {child.desc}: SIGKILL")

    def _live(self, kind: str) -> list[Child]:
        return [c for c in self.children if c.kind == kind]

    # -- pieces -------------------------------------------------------
    def _ensure_llm_server(self) -> None:
        try:
            out = subprocess.run(["bash", "scripts/llm_server.sh", "status"],
                                 capture_output=True, text=True, cwd=ROOT, timeout=30, check=False)
            text = out.stdout + out.stderr
        except (OSError, subprocess.SubprocessError) as e:
            self.log(f"llm server status failed: {e}")
            return
        if "not running" in text:
            self.log("llm server not running, starting")
            try:
                subprocess.run(["bash", "scripts/llm_server.sh", "start"],
                               capture_output=True, cwd=ROOT, timeout=60, check=False)
            except (OSError, subprocess.SubprocessError) as e:
                self.log(f"llm server start failed: {e}")

    def _credit(self) -> float | None:
        try:
            return vast.credit()
        except Exception as e:  # noqa: BLE001 - network/CLI failure: skip paid launches this tick
            self.log(f"vast credit check failed: {e}")
            return None

    def _allow(self, label: str) -> None:
        """Record an approved rental BEFORE creating it (vast-guard.sh destroys the rest)."""
        ALLOWED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALLOWED_FILE, "a") as f:
            f.write(label + "\n")

    def _ensure_session(self, now: float, credit: float | None, name: str = "lines") -> None:
        sess = vast.load_session(name)
        if sess is not None and sess.get("deadline", float("inf")) < now:
            self.log(f"vast session {name} past deadline, stopping it first")
            try:
                cmd = ["uv", "run", "--no-sync", "python", "scripts/vast_session.py", "stop"]
                if name != "lines":
                    cmd += ["--session", name]
                subprocess.run(cmd, capture_output=True, cwd=ROOT, timeout=600, check=False)
            except (OSError, subprocess.SubprocessError) as e:
                self.log(f"session stop failed: {e}")
            sess = None
        live_names = {c.name or "lines" for c in self._live("session")}
        if sess is not None and name not in live_names:
            try:
                alive = any(i["id"] == sess["instance_id"] for i in vast.our_instances())
            except Exception as e:  # noqa: BLE001 - vast CLI hiccup: assume alive this tick
                self.log(f"session liveness check failed: {e}")
                alive = True
            if not alive:
                self.log(f"vast session instance {sess['instance_id']} is gone: "
                         f"dropping {vast.session_file(name).name}")
                vast.session_file(name).unlink(missing_ok=True)
                sess = None
        if sess is not None:
            return
        if credit is None or credit < self.args.credit_floor:
            return
        if name in live_names:
            return
        if now < self.session_retry_after.get(name, 0.0):
            return
        hhmm = time.strftime("%H%M")
        label = session_label(name, hhmm)
        self._allow(f"ygo-zero-{label}")
        logname = f"vast-session-{hhmm}.log" if name == "lines" else f"vast-session-{name}-{hhmm}.log"
        self._launch("session", session_command(hhmm, name),
                     RUNS_CACHE / logname, desc=label, name=name)

    def _line_session(self) -> str | None:
        """The live "lines" box with the fewest remote workers (planners/improvers/expanders), or None."""
        live = [n for n in line_session_names(self.args.line_sessions) if vast.load_session(n) is not None]
        if not live:
            return None
        load = {n: 0 for n in live}
        for c in self.children:
            if c.kind in ("planner", "improver", "expander") and c.name in load:
                load[c.name] += 1
        return min(live, key=lambda n: (load[n], live.index(n)))

    def _ensure_planners(self) -> None:
        if self._line_session() is None:
            return  # planners verify --remote on a lines box
        running = {c.deck for c in self._live("planner")}
        orphans = max(0, running_scripts("plan_lines.py") - len(self._live("planner")))
        while len(self._live("planner")) + orphans < self.args.planners:
            deck = choose_planner_deck(ROOT, running)
            if deck is None:
                return
            seed = next_planner_seed(ROOT, deck)
            sess = self._line_session()
            self._launch("planner", planner_command(deck, seed, self.args.variant, sess),
                         RUNS_CACHE / f"plan-s{seed}-{deck}.log",
                         desc=f"{deck} s{seed} @{sess}", deck=deck, name=sess)
            running.add(deck)

    def _ensure_improvers(self) -> None:
        if self._line_session() is None:
            return
        running = {c.deck for c in self._live("improver")}
        while len(self._live("improver")) < self.args.improvers:
            job = choose_improve_job(ROOT, running)
            if job is None:
                return
            src, deck = job
            sess = self._line_session()
            self._launch("improver", improve_command(src, deck, sess),
                         RUNS_CACHE / f"improve-{src.replace('/', '_')}-{deck}.log", desc=f"{src} {deck} @{sess}",
                         deck=job, name=sess)
            running.add(job)

    def _ensure_expanders(self) -> None:
        if self._line_session() is None:
            return  # expansion verifies --remote on a lines box
        running = {c.deck for c in self._live("expander")}
        while len(self._live("expander")) < self.args.expanders:
            vf = choose_expand_job(ROOT, running)
            if vf is None:
                return
            vf_rel = vf.relative_to(ROOT)
            src_rel = vf_rel.parent.parent
            lines_dir, out_dir, skips = expander_target(vf_rel)
            key = str(vf_rel)
            sess = self._line_session()
            self._launch("expander", expand_command(vf_rel, lines_dir, out_dir, skips, sess),
                         RUNS_CACHE / f"expand-{path_to_flat(src_rel)}-{vf_rel.stem}.log",
                         desc=f"{src_rel} {vf_rel.stem} @{sess}", deck=key, name=sess)
            running.add(key)

    def _ensure_judge(self) -> None:
        """One judge worker whenever >= 10 games are unjudged (realism scores -> BC weights, NOTEBOOK 2026-10-03)."""
        if self.args.judge and not self._live("judge") and unjudged_games(ROOT) >= 10:
            self._launch("judge", judge_command(), RUNS_CACHE / "judge.log", desc="judge_games")

    def _ensure_games(self, remote: bool = False) -> None:
        orphans = max(0, running_scripts("play_games.py") - len(self._live("game")))
        while len(self._live("game")) + orphans < self.args.games:
            decks = eligible_game_decks(ROOT)
            if len(decks) < 2:
                self.log("games: fewer than 2 eligible decks")
                return
            hold = sorted(holdout_decks(ROOT) & set(pool_decks(ROOT)))
            if len(hold) >= 2 and self.rng.random() < self.args.holdout_game_frac:
                # evaluation-only games: holdout vs holdout, so every sample lands in the holdout split and no
                # training sample ever sees a holdout deck's cards (not even as the opponent's)
                a, b = pick_game_pair(hold, count_games_per_deck(ROOT), self.rng)
            else:
                a, b = pick_game_pair(decks, count_games_per_deck(ROOT), self.rng)
            seed = self.rng.randint(0, 10**6)
            self._launch("game", game_command(a, b, seed, self.args.variant, remote=remote),
                         RUNS_CACHE / f"game-{a}-vs-{b}-{seed}.log", desc=f"{a}-vs-{b} seed={seed}")

    def _ensure_bc(self, now: float, credit: float | None) -> None:
        if now < self.bc_next_due or self._live("bc"):
            return
        if credit is None or credit < self.args.credit_floor:
            return
        name = "bcs-" + time.strftime("%Y%m%d-%H%M")
        self._allow(f"ygo-zero-{name}")
        self._launch("bc", bc_command(name), RUNS_CACHE / f"vast-{name}.log",
                     desc=name, name=name)
        self.bc_next_due = now + self.args.bc_every_h * 3600

    def _finish_bc(self, child: Child) -> None:
        """Copy expansions back, drop the flat dir, append the scaling row. Runs/<name>/ is local."""
        name = child.name
        rundir = ROOT / "runs" / name
        exp = rundir / "expanded"
        if exp.is_dir():
            for flat in sorted(exp.iterdir()):
                if not flat.is_dir():
                    continue
                dest = ROOT / flat_to_path(flat.name)
                try:
                    shutil.copytree(flat, dest, dirs_exist_ok=True)
                    self.log(f"bc {name}: copied {flat.name} -> {dest}")
                except OSError as e:
                    self.log(f"bc {name}: copy {flat.name} failed: {e}")
            try:
                shutil.rmtree(exp)
            except OSError as e:
                self.log(f"bc {name}: rmtree expanded failed: {e}")
        train = rundir / "train.txt"
        metrics = rundir / "bc" / "metrics.jsonl"
        if not train.is_file() or not metrics.is_file():
            self.log(f"bc {name}: no train.txt/metrics.jsonl, skipping scaling row")
            return
        try:
            counts, decks = parse_train_first_line(train.read_text().splitlines()[0])
            best = metrics_best(metrics)
            row = [name, datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M"),
                   str(counts.get("train", "")), str(counts.get("val", "")),
                   str(counts.get("holdout", "")), str(len(decks)),
                   str(best["val_acc"]), str(best["val_epoch"]),
                   str(best["holdout_acc"]), str(best["holdout_epoch"])]
        except (ValueError, SyntaxError, IndexError, KeyError) as e:
            self.log(f"bc {name}: could not parse results: {e}")
            return
        if not SCALING_TSV.is_file():
            SCALING_TSV.write_text("name\ttime_utc\ttrain\tval\tholdout\tn_decks\t"
                                   "best_val_acc\tbest_val_epoch\tbest_holdout_acc\tbest_holdout_epoch\n")
        with open(SCALING_TSV, "a") as f:
            f.write("\t".join(row) + "\n")
        self.log(f"bc {name}: scaling row: {' '.join(row)}")

    def _repo_size(self) -> tuple[int, str]:
        try:
            out = subprocess.run(["du", "-sb", str(ROOT)], capture_output=True, text=True,
                                   timeout=300, check=False)
            by = int(out.stdout.split()[0])
            hr = subprocess.run(["du", "-sh", str(ROOT)], capture_output=True, text=True,
                                timeout=300, check=False).stdout.split()[0]
            return by, hr
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
            self.log(f"du failed: {e}")
            return 0, "?"

    def _summary(self, now: float) -> None:
        counts = verified_counts(ROOT)
        ranked = sorted(counts.items(), key=lambda kv: kv[1])
        top = ", ".join(f"{d}={n}" for d, n in ranked[-3:][::-1])
        bottom = ", ".join(f"{d}={n}" for d, n in ranked[:3])
        games = sum(count_games_per_deck(ROOT).values())
        credit = self._credit()
        by, hr = self._repo_size()
        self.log(f"summary verified top3 [{top}] bottom3 [{bottom}] games={games} "
                 f"credit={credit} du={hr} children={len(self.children)}")
        if by >= REPO_BUDGET_BYTES:
            self.over_budget = True
            self.log(f"WARNING: repo {hr} >= 45 GB, stop launching (AGENTS.md storage rule)")

    def _quota_paused(self, now: float) -> bool:
        """LLM quota hit (marker written by llm.complete): launch nothing that calls muse; probe every 10 min with one
        tiny call; after 30 min paused, end the vast sessions (no verification work while muse is down)."""
        if not llm.QUOTA_MARKER.exists():
            if self.quota_since is not None:
                self.log(f"llm quota: resumed after {(now - self.quota_since) / 60:.0f} min")
                self.quota_since = None
            return False
        if self.quota_since is None:
            self.quota_since = now
            self.quota_probe_at = now + 600
            self.log("llm quota exceeded: pausing LLM workers (probe every 10 min)")
        if now >= self.quota_probe_at:
            self.quota_probe_at = now + 600
            os.environ["YGO_OPENCODE_SERVER"] = LLM_SERVER_URL
            try:
                llm.complete(f"Reply with the word ok. {now:.0f}", model=PROBE_MODEL, timeout=90)
                llm.QUOTA_MARKER.unlink(missing_ok=True)
                self.log("llm quota probe ok: resuming")
                return False
            except Exception as e:  # noqa: BLE001 - still limited (or down): stay paused
                self.log(f"llm quota probe failed: {str(e)[:120]}")
                llm.QUOTA_MARKER.write_text(str(now))
        if now - self.quota_since > 1800:
            for name in [*line_session_names(self.args.line_sessions), "games"]:
                if vast.load_session(name) is not None:
                    self.log(f"llm quota pause > 30 min: ending vast session {name}")
                    cmd = ["uv", "run", "--no-sync", "python", "scripts/vast_session.py", "stop"]
                    if name != "lines":
                        cmd += ["--session", name]
                    subprocess.run(cmd, capture_output=True, cwd=ROOT, timeout=600, check=False)
        return True

    # -- main tick ----------------------------------------------------
    def tick(self, now: float | None = None) -> bool:
        """One 60 s step. Returns True when the driver should exit."""
        now = time.time() if now is None else now
        self._reap()
        self._enforce_limits(now)
        if STOP_FILE.exists():
            if not self.children:
                self.log("STOP_DATAGEN set, no children left: exiting")
                return True
            self.log(f"STOP_DATAGEN set: not launching, waiting for {len(self.children)} children")
            return False
        if self.over_budget:
            return False
        if now - self.last_summary >= SUMMARY_S:
            self._summary(now)
            self.last_summary = now
        self._ensure_llm_server()
        credit = self._credit()
        if self._quota_paused(now):
            self._ensure_bc(now, credit)  # training needs no LLM
            return False
        for name in line_session_names(self.args.line_sessions):
            self._ensure_session(now, credit, name)
        games_remote = False
        if self.args.game_session:
            self._ensure_session(now, credit, "games")
            games_remote = vast.load_session("games") is not None
        self._ensure_planners()
        self._ensure_improvers()
        self._ensure_expanders()
        self._ensure_games(remote=games_remote)
        self._ensure_judge()
        self._ensure_bc(now, credit)
        return False


def running_scripts(script: str) -> int:
    """Python processes running `scripts/<script>` machine-wide (uv wrappers not counted). Includes workers orphaned
    by a previous driver, so a restarted driver does not oversubscribe the laptop or the LLM server."""
    n = 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argv and Path(argv[0].decode(errors="ignore")).name.startswith("python") and \
                any(a.decode(errors="ignore").endswith(f"scripts/{script}") for a in argv[1:3]):
            n += 1
    return n


def _oom_first() -> None:
    """Children volunteer for the OOM killer before the driver (2026-10-03: an OOM took the driver down)."""
    try:
        Path("/proc/self/oom_score_adj").write_text("500")
    except OSError:
        pass


def pin_laptop() -> None:
    """Pin to cores 0-3, nice 19 (children inherit). Best effort: log and continue."""
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except (AttributeError, OSError) as e:
        print(f"pin: sched_setaffinity failed: {e}", flush=True)
    try:
        os.nice(19)
    except OSError as e:
        print(f"pin: nice failed: {e}", flush=True)


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planners", type=int, default=3)
    ap.add_argument("--improvers", type=int, default=2)
    ap.add_argument("--expanders", type=int, default=1)
    ap.add_argument("--holdout-game-frac", type=float, default=0.15,
                    help="share of games played holdout-vs-holdout (evaluation of interaction on unseen decks)")
    ap.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True, help="rate new games for realism")
    ap.add_argument("--line-sessions", type=int, default=1, help="number of warm 'lines' boxes (lines, lines2, ...)")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--variant", default=None,
                    help="muse reasoning effort (minimal|low|high|max; default: provider's)")
    ap.add_argument("--credit-floor", type=float, default=3.0)
    ap.add_argument("--game-session", action=argparse.BooleanOptionalAction, default=True,
                    help="keep a second warm box (session 'games') for game verification")
    ap.add_argument("--bc-every-h", type=float, default=3.0)
    ap.add_argument("--once", action="store_true", help="one tick then exit (for tests)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pin_laptop()
    loop = DatagenLoop(args)
    loop.log(f"datagen start planners={args.planners} improvers={args.improvers} "
               f"expanders={args.expanders} games={args.games} variant={args.variant} "
               f"credit_floor={args.credit_floor} bc_every_h={args.bc_every_h} "
               f"game_session={args.game_session}")
    while True:
        done = loop.tick()
        if done or args.once:
            break
        time.sleep(TICK_S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())