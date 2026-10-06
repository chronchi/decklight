"""vast.ai job launcher: rent → ship code → run under a hard time cap → pull results → always destroy.

Rules (AGENTS.md): every job has a time cap, results land on the laptop, the instance is destroyed even on
error, and every spend is logged (data/vast_ledger.jsonl here; the total goes into STATUS.md by hand).
Uses the `vastai` CLI (`uvx vastai`, key in ~/.config/vastai) through subprocess with --raw JSON output.
"""

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "data" / "vast_ledger.jsonl"
BAD_MACHINES = ROOT / "data" / "vast_bad_machines.json"  # machine_id -> reason; skipped by find_offer
LABEL_PREFIX = "ygo-zero-"
REMOTE_DIR = "/workspace/ygo-zero"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]  # fmt: skip
# Shipped to the instance: tracked files + submodules, minus anything large or derived.
EXCLUDES = [".git", ".venv", "data", "runs", "build", "__pycache__", "*.so", ".pytest_cache", ".ruff_cache"]


class VastError(RuntimeError):
    pass


class HostFailed(VastError):
    """The machine could not start our container (e.g. broken GPU passthrough); try another offer."""


def cli(*args: str, raw: bool = True, timeout: int = 120):
    cmd = ["uvx", "vastai", *args] + (["--raw"] if raw else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if out.returncode != 0:
        raise VastError(f"{' '.join(cmd)}: {out.stderr.strip() or out.stdout.strip()}")
    if not raw:
        return out.stdout
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"error": True, "msg": out.stdout.strip()[:300]}


def credit() -> float:
    return float(cli("show", "user")["credit"])


def our_instances() -> list[dict]:
    return [i for i in cli("show", "instances") if (i.get("label") or "").startswith(LABEL_PREFIX)]


def destroy(instance_id: int, verify_s: int = 60) -> None:
    """Destroy and confirm it is gone. `-y` matters: without it the CLI asks for confirmation, reads EOF,
    prints "Aborted." and exits 0, destroying nothing (cost us a stray instance on 2026-10-01)."""
    cli("destroy", "instance", str(instance_id), "-y", raw=False)
    t0 = time.time()
    while time.time() - t0 < verify_s:
        if all(i["id"] != instance_id for i in cli("show", "instances")):
            return
        time.sleep(5)
    raise VastError(f"instance {instance_id} still listed {verify_s}s after destroy")


ALLOWED_FILE = ROOT / "data" / "cache" / "runs" / "vast-allowed.txt"  # labels approved by the datagen driver / Claude
STRAY_AGE_S = 6 * 3600  # older than any job cap (4.5 h): an allowed instance this old has lost its launcher


def destroy_strays() -> list[int]:
    """Destroy instances we labelled that no live launcher owns. Spared: the warm session boxes (every existing
    session file) and allowed labels younger than STRAY_AGE_S (another launcher's running job). 2026-10-03 a BC
    launch destroyed the live session box here, so the planners crash-looped for an hour."""
    keep: set = set()
    for f in _session_files():
        try:
            sess = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(sess, dict) and sess.get("instance_id") is not None:
            keep.add(sess["instance_id"])
    allowed = set(ALLOWED_FILE.read_text().split()) if ALLOWED_FILE.exists() else set()
    now = time.time()
    ids = []
    for i in our_instances():
        if i["id"] in keep or (i.get("label") in allowed and now - float(i.get("start_date") or now) < STRAY_AGE_S):
            continue
        ids.append(i["id"])
    for i in ids:
        destroy(i)
    return ids


@dataclass
class JobSpec:
    name: str
    command: str  # run in REMOTE_DIR after setup; write outputs under runs/<name>/
    hours_cap: float = 0.5  # hard wall-clock limit of the whole job on the instance
    max_dph: float = 0.15  # $/h cap for the offer
    min_cpus: int = 8
    min_gpu_ram_gb: float = 6
    disk_gb: int = 30
    image: str = "python:3.12-bookworm"  # has gcc/g++/git/curl; torch + CUDA libs come from PyPI via uv
    gpu: bool = True
    extra_query: str = ""
    setup: str = "bash scripts/vast_setup.sh"
    data_paths: tuple[str, ...] = ()  # extra repo paths to ship despite EXCLUDES (e.g. "data/demos/lines")
    min_mb_s: float = 15.0  # measured download speed a host must reach (0 = skip the check); vast's inet_down lies


@dataclass
class JobResult:
    name: str
    instance_id: int | None = None
    offer: dict = field(default_factory=dict)
    exit_code: int | None = None
    seconds: float = 0.0
    est_cost: float = 0.0
    credit_before: float = 0.0
    credit_after: float | None = None
    error: str = ""


def load_bad_machines() -> dict[str, str]:
    return json.loads(BAD_MACHINES.read_text()) if BAD_MACHINES.exists() else {}


def mark_bad_machine(machine_id, reason: str) -> None:
    bad = load_bad_machines()
    bad[str(machine_id)] = f"{time.strftime('%Y-%m-%d')}: {reason}"[:200]
    BAD_MACHINES.parent.mkdir(parents=True, exist_ok=True)
    BAD_MACHINES.write_text(json.dumps(bad, indent=1))


def find_offer(
    spec: JobSpec, exclude_machines: set[int] = frozenset(), exclude_offers: set[int] = frozenset()
) -> dict:
    q = (f"reliability>0.98 rentable=true verified=true disk_space>={spec.disk_gb} inet_down>=200 "
         f"cpu_cores_effective>={spec.min_cpus} dph<={spec.max_dph}")  # fmt: skip
    q += f" num_gpus=1 gpu_ram>={spec.min_gpu_ram_gb}" if spec.gpu else " num_gpus=0"
    q += f" {spec.extra_query}" if spec.extra_query else ""
    offers = [
        o
        for o in cli("search", "offers", q, "-o", "dph")
        if o.get("machine_id") not in exclude_machines
        and str(o.get("machine_id")) not in load_bad_machines()
        and o["id"] not in exclude_offers
    ]
    if not offers:
        raise VastError(f"no offer matches: {q}")
    return offers[0]


def _ssh_target(inst: dict) -> tuple[str, int]:
    try:
        return inst["public_ipaddr"], int(inst["ports"]["22/tcp"][0]["HostPort"])
    except (KeyError, TypeError, IndexError, ValueError):
        return inst["ssh_host"], int(inst["ssh_port"])


def disable_auto_tmux(host: str, port: int) -> None:
    try:
        subprocess.run(
            ["ssh", "-tt", *SSH_OPTS, "-p", str(port), f"root@{host}"],
            input="touch ~/.no_auto_tmux; tmux kill-server; exit\n",
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pass


def wait_ready(instance_id: int, timeout_s: int = 900, ssh_grace_s: int = 300) -> tuple[str, int]:
    """Wait until the instance runs and answers ssh. A box that runs but refuses ssh for `ssh_grace_s` is failed
    early (val-20261005-1820: publickey denied for the full 15 min)."""
    t0 = time.time()
    running_since: float | None = None
    while time.time() - t0 < timeout_s:
        inst = cli("show", "instance", str(instance_id))
        msg = inst.get("status_msg") or ""
        if inst.get("intended_status") == "stopped" or "Error" in msg or "failed" in msg:
            raise HostFailed(f"instance {instance_id}: {msg.strip()[:300]}")
        if inst.get("actual_status") == "running" and inst.get("ssh_host"):
            host, port = _ssh_target(inst)
            disable_auto_tmux(host, port)
            ok = subprocess.run(
                ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", "true"], capture_output=True, check=False
            )
            if ok.returncode == 0:
                return host, port
            running_since = running_since or time.time()
            if time.time() - running_since > ssh_grace_s:
                raise HostFailed(f"instance {instance_id}: running but ssh refused for {ssh_grace_s}s")
        time.sleep(15)
    raise HostFailed(f"instance {instance_id} not ready after {timeout_s}s")


def ssh(host: str, port: int, command: str, timeout_s: float | None = None, log: Path | None = None) -> int:
    cmd = ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", command]
    with open(log, "a") if log else open("/dev/null", "w") as out:
        try:
            return subprocess.run(
                cmd, stdout=out, stderr=subprocess.STDOUT, timeout=timeout_s, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            return 124


def run_detached(host: str, port: int, name: str, job: str, cap: int, log: Path, poll_s: int = 30) -> int:
    """Start the job with nohup on the box and poll for its exit-code file with short ssh calls, so a dropped
    connection (seen 2026-10-02: "closed by remote host" mid-setup) does not kill or lose the job."""
    rdir = f"{REMOTE_DIR}/runs/{name}"
    start = (
        f"mkdir -p {rdir} && test ! -e {rdir}/STARTED && touch {rdir}/STARTED && "
        f"setsid nohup bash -c {shlex.quote(f'timeout {cap} bash -lc {shlex.quote(job)}; echo $? > {rdir}/EXIT')}"
        f" > {rdir}/job.log 2>&1 < /dev/null & disown; echo started"
    )
    # Judge by the marker, not ssh's exit code: some hosts return 255 when the session closes (2026-10-02).
    for _ in range(3):
        try:
            out = subprocess.run(
                ["ssh", "-n", *SSH_OPTS, "-p", str(port), f"root@{host}", start],
                capture_output=True, text=True, timeout=60, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            continue
        with open(log, "a") as f:
            f.write(out.stdout + out.stderr)
        if "started" in out.stdout:
            break
        probe = subprocess.run(
            ["ssh", "-n", *SSH_OPTS, "-p", str(port), f"root@{host}", f"test -e {rdir}/STARTED && echo yes"],
            capture_output=True, text=True, timeout=60, check=False,
        )  # fmt: skip
        if "yes" in probe.stdout:
            break  # the first attempt did start it (its ssh just did not return cleanly)
        time.sleep(10)
    else:
        raise VastError("could not start the job on the instance")
    deadline = time.time() + cap + 300
    alerts_seen = 0
    progress_seen = 0
    polls, stalled = 0, False
    while time.time() < deadline:
        time.sleep(poll_s)
        alerts_seen = relay_alerts(host, port, rdir, log, alerts_seen)
        progress_seen = relay_alerts(host, port, rdir, log, progress_seen, "PROGRESS", "PROGRESS ")
        polls += 1
        if polls % max(1, 300 // poll_s) == 0:  # every ~5 min
            stalled = stall_check(host, port, rdir, log, stalled)
        try:
            out = subprocess.run(
                ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", f"cat {rdir}/EXIT 2>/dev/null"],
                capture_output=True, text=True, timeout=60, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            continue  # transient: keep polling until the deadline
        if out.returncode == 0 and out.stdout.strip():
            relay_alerts(host, port, rdir, log, alerts_seen)
            relay_alerts(host, port, rdir, log, progress_seen, "PROGRESS", "PROGRESS ")
            return int(out.stdout.strip())
    return 124


STALL_MIN = 40  # longer than any quiet stage (data loading ~15 min, one training epoch ~25 min)


def stall_check(host: str, port: int, rdir: str, log: Path, stalled: bool, minutes: int = STALL_MIN) -> bool:
    """Fail loud on a hang: ALERT once when no file under runs/<name> (except the usage logger's) changed for
    `minutes` (sw-20261005-2323 sat 20 min in a bare `wait` with nothing in the launcher log); a note when output
    resumes. Returns the current stalled state."""
    cmd = f"find {rdir} -type f ! -name usage.txt -mmin -{minutes} 2>/dev/null | head -1"
    try:
        out = subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
                             capture_output=True, text=True, timeout=60, check=False)
    except subprocess.TimeoutExpired:
        return stalled
    if out.returncode != 0:
        return stalled  # ssh trouble is not a stall
    now = not out.stdout.strip()
    if now != stalled:
        with open(log, "a") as f:
            f.write(f"ALERT stall: no output under {rdir} for {minutes} min\n" if now else "stall over: output resumed\n")
    return now


def relay_alerts(
    host: str, port: int, rdir: str, log: Path, seen: int, filename: str = "ALERT", prefix: str = "ALERT "
) -> int:
    """Fail loud: copy new lines of the job's runs/<name>/<filename> file into the launcher log as they appear,
    each prefixed with `prefix` (unless already prefixed), so a broken variant (ALERT) or training progress
    (PROGRESS, eval lines tailed by vast_rl.sh) is visible within one poll instead of at the end of the job.
    A missing file is normal (other job types have no PROGRESS): no error. Returns the number of lines
    relayed so far."""
    try:
        out = subprocess.run(
            ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", f"cat {rdir}/{filename} 2>/dev/null"],
            capture_output=True, text=True, timeout=60, check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return seen
    lines = [x for x in out.stdout.splitlines() if x.strip()]
    if len(lines) > seen:
        tag = prefix.strip()
        with open(log, "a") as f:
            f.writelines(f"{x}\n" if x.startswith(tag) else f"{prefix}{x}\n" for x in lines[seen:])
    return max(seen, len(lines))


def push_code(host: str, port: int, data_paths: tuple[str, ...] = ()) -> None:
    excl = " ".join(f"--exclude={shlex.quote(e)}" for e in EXCLUDES)
    remote = f"mkdir -p {REMOTE_DIR} && tar xzf - -C {REMOTE_DIR}"
    ssh = f"ssh {' '.join(SSH_OPTS)} -p {port} root@{host} {shlex.quote(remote)}"
    subprocess.run(f"tar czf - {excl} -C {shlex.quote(str(ROOT))} . | {ssh}", shell=True, check=True)
    if data_paths:  # shipped separately: EXCLUDES drops data/ from the code tarball
        paths = " ".join(shlex.quote(p) for p in data_paths)
        subprocess.run(f"tar czf - -C {shlex.quote(str(ROOT))} {paths} | {ssh}", shell=True, check=True)


def pull_results(host: str, port: int, name: str) -> Path:
    dest = ROOT / "runs" / name
    dest.mkdir(parents=True, exist_ok=True)
    remote = f"cd {REMOTE_DIR}/runs/{shlex.quote(name)} 2>/dev/null && tar czf - . || true"
    pipe = f"ssh {' '.join(SSH_OPTS)} -p {port} root@{host} {shlex.quote(remote)} | tar xzf - -C {shlex.quote(str(dest))}"
    subprocess.run(pipe, shell=True, check=False)
    return dest


def _rentable_offer(spec: JobSpec, bad_machines: set[int], tries: int = 10) -> dict:
    """Cheapest offer that can actually be rented: listings can be stale (CLI answers `no_such_ask` with exit 0
    and an error object, seen 2026-10-02 on CPU-only offers). Creates the instance; the result is in `_created`."""
    stale: set[int] = set()
    for _ in range(tries):
        offer = find_offer(spec, bad_machines, stale)
        worst = offer["dph_total"] * (spec.hours_cap + 0.25)
        if worst > credit() - 0.25:
            raise VastError(f"worst-case ${worst:.2f} exceeds credit - reserve")
        out = cli("create", "instance", str(offer["id"]), "--image", spec.image, "--disk", str(spec.disk_gb),
                  "--ssh", "--direct", "--label", LABEL_PREFIX + spec.name)  # fmt: skip
        if isinstance(out, dict) and out.get("new_contract"):
            offer["_created"] = out
            return offer
        print(f"offer {offer['id']} not rentable: {out}", flush=True)
        stale.add(offer["id"])
    raise VastError(f"no rentable offer after {tries} tries")


def _destroy_retrying(instance_id: int) -> None:
    for attempt in range(5):
        try:
            destroy(instance_id)
            return
        except VastError as e:
            print(f"destroy {instance_id} attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10)


SESSION_FILE = ROOT / "data" / "vast_session.json"


def session_file(name: str = "lines") -> Path:
    """Session-file path for a named session: `data/vast_session.json` for "lines" (unchanged),
    `data/vast_session-<name>.json` otherwise."""
    if name == "lines":
        return SESSION_FILE
    return ROOT / "data" / f"vast_session-{name}.json"


def _session_files() -> list[Path]:
    """Every existing session file (the default one plus any named ones)."""
    files = sorted((ROOT / "data").glob("vast_session-*.json")) if (ROOT / "data").exists() else []
    if SESSION_FILE.exists() and SESSION_FILE not in files:
        files = [SESSION_FILE, *files]
    return files


def load_session(name: str = "lines") -> dict | None:
    """The warm-box record written by `start_session`, or None when there is no session."""
    path = session_file(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def require_session(name: str = "lines") -> dict:
    s = load_session(name)
    if s is None:
        raise VastError(
            f"no vast session {name!r} (no {session_file(name).name}); start one with "
            "`uv run python scripts/vast_session.py start` on the laptop"
        )
    return s


def start_session(spec: JobSpec, max_hosts: int = 3, name: str = "lines") -> dict:
    """Rent a cheap (usually CPU-only) box exactly like `run_job` does, push the code, run `spec.setup`
    once, and keep the box warm: write the session file for `name` and return it. Runs no job, destroys
    nothing. The hours cap is recorded as a deadline in the file (`vast_session.py status` warns past it); the
    box itself is stopped by `vast_session.py stop` (i.e. `end_session`)."""
    path = session_file(name)
    if load_session(name) is not None:
        raise VastError(f"a session already exists ({path}); stop it first")
    strays = destroy_strays()
    if strays:
        print(f"destroyed stray instances {strays}")
    credit_before = credit()
    created: list[tuple[int, float, float]] = []  # (instance id, $/h, start time)
    bad_machines: set[int] = set()
    t0 = time.time()
    for _ in range(max_hosts):
        offer = _rentable_offer(spec, bad_machines)
        selected = {
            k: offer.get(k)
            for k in ("id", "machine_id", "gpu_name", "dph_total", "cpu_cores_effective", "geolocation")
        }
        out = offer.pop("_created")  # worst-case cost was checked before creating
        instance_id = int(out["new_contract"])
        created.append((instance_id, offer["dph_total"], time.time()))
        print(f"instance {instance_id}: {selected}", flush=True)
        try:
            host, port = wait_ready(instance_id)
            break
        except HostFailed as e:
            print(f"host failed, trying another machine: {e}", flush=True)
            mark_bad_machine(offer.get("machine_id"), str(e))
            _destroy_retrying(instance_id)
            bad_machines.add(offer.get("machine_id"))
    else:
        raise VastError(f"{max_hosts} hosts failed to start the container")
    print(f"ready after {time.time() - t0:.0f}s; pushing code", flush=True)
    for attempt in range(3):
        try:
            push_code(host, port, spec.data_paths)
            break
        except subprocess.CalledProcessError:
            if attempt == 2:
                mark_bad_machine(selected.get("machine_id"), "ssh transfers keep failing (exit 255)")
                _destroy_retrying(instance_id)
                raise
            time.sleep(10)
    print(f"running setup: {spec.setup}", flush=True)
    setup_log = ROOT / "runs" / "session" / "setup.log"
    setup_log.parent.mkdir(parents=True, exist_ok=True)
    job = f"export PATH=$HOME/.local/bin:$PATH && cd {REMOTE_DIR} && ({spec.setup})"
    rc = ssh(host, port, job, timeout_s=1800, log=setup_log)
    if rc != 0:
        _destroy_retrying(instance_id)
        raise VastError(f"session setup failed (exit {rc}); see {setup_log}")
    started = time.time()
    session = {
        "instance_id": instance_id,
        "host": host,
        "port": port,
        "started": started,
        "dph": offer["dph_total"],
        "offer": selected,
        "hours_cap": spec.hours_cap,
        "deadline": started + spec.hours_cap * 3600,
        "name": spec.name,
        "setup": spec.setup,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=1))
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(
            json.dumps(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "event": "session_start",
                    "name": spec.name,
                    "instance_id": instance_id,
                    "offer": selected,
                    "dph": offer["dph_total"],
                    "credit_before": credit_before,
                }
            )
            + "\n"
        )
    print(f"session ready on {host}:{port} (cap {spec.hours_cap}h); {path} written", flush=True)
    return session


def end_session(name: str = "lines") -> dict:
    """Destroy the session box, log the spend like `run_job`, delete the session file. Returns a summary."""
    s = require_session(name)
    instance_id = int(s["instance_id"])
    _destroy_retrying(instance_id)
    now = time.time()
    seconds = now - float(s.get("started", now))
    est_cost = float(s.get("dph", 0.0)) * seconds / 3600
    summary = {"instance_id": instance_id, "seconds": seconds, "est_cost": est_cost}
    try:
        summary["credit_after"] = credit()
    except VastError:
        pass
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(
            json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "event": "session_end", **summary}) + "\n"
        )
    session_file(name).unlink(missing_ok=True)
    print(f"session {instance_id} destroyed; spent ≈ ${est_cost:.3f} ({seconds / 60:.1f} min)", flush=True)
    return summary


def session_run(cmd: str, timeout_s: float | None = None, name: str = "lines") -> tuple[int, str]:
    """Run `cmd` in REMOTE_DIR on the session box. Returns (exit code, combined output)."""
    s = require_session(name)
    full = f"cd {REMOTE_DIR} && ({cmd})"
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, "-p", str(s["port"]), f"root@{s['host']}", full],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def session_push(local_paths: list[str | Path], remote_dir: str, name: str = "lines") -> None:
    """Copy local files/dirs into `remote_dir` on the session box (scp -r over the same ssh options)."""
    s = require_session(name)
    rc, _ = session_run(f"mkdir -p {shlex.quote(remote_dir)}", name=name)
    if rc != 0:
        raise VastError(f"could not create {remote_dir} on the session box")
    for p in local_paths:
        subprocess.run(
            ["scp", "-r", *SSH_OPTS, "-P", str(s["port"]), str(p), f"root@{s['host']}:{remote_dir}/"],
            check=True,
        )


def session_pull(remote_path: str, local_path: str | Path, name: str = "lines") -> None:
    """Copy `remote_path` from the session box to `local_path` (scp -r over the same ssh options)."""
    s = require_session(name)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["scp", "-r", *SSH_OPTS, "-P", str(s["port"]), f"root@{s['host']}:{remote_path}", str(local_path)],
        check=True,
    )


def repush_code(name: str = "lines") -> None:
    """Re-push the working tree to the session box (for code changes mid-session; .venv and build stay)."""
    s = require_session(name)
    push_code(s["host"], int(s["port"]))


SPEED_URL = "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"


def download_mb_s(host: str, port: int, seconds: int = 15) -> float:
    """Measured download speed of the box (MB/s) on the biggest file every job fetches (the CUDA torch wheel).
    Three slow hosts in a row on 2026-10-05 advertised inet_down >= 800 Mbps and delivered 1-4 MB/s."""
    cmd = ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}",
           f"curl -s -o /dev/null --max-time {seconds} -w '%{{size_download}}' '{SPEED_URL}' || true"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 30, check=False).stdout
        return float(out.strip().split()[-1]) / seconds / 1e6
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return 0.0


def run_job(spec: JobSpec, reserve: float = 0.25, max_hosts: int = 3) -> JobResult:
    """The whole lifecycle. Every instance created is destroyed in `finally`, whatever happens. A host that
    cannot start the container is dropped and the next-cheapest offer on another machine is tried."""
    res = JobResult(spec.name)
    log = ROOT / "runs" / spec.name / "launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # register this job's label first, so another launcher's stray sweep spares our box (2026-10-05: a launch
    # destroyed the running bct-20261005-1431 box because its label had not been added by hand)
    ALLOWED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALLOWED_FILE, "a") as f:
        f.write(LABEL_PREFIX + spec.name + "\n")
    strays = destroy_strays()
    if strays:
        print(f"destroyed stray instances {strays}")
    res.credit_before = credit()
    created: list[tuple[int, float, float]] = []  # (instance id, $/h, start time)
    bad_machines: set[int] = set()
    t0 = time.time()
    try:
        for _ in range(max_hosts):
            offer = _rentable_offer(spec, bad_machines)
            res.offer = {
                k: offer.get(k)
                for k in ("id", "machine_id", "gpu_name", "dph_total", "cpu_cores_effective", "geolocation")
            }
            out = offer.pop("_created")  # worst-case cost was checked before creating
            res.instance_id = int(out["new_contract"])
            created.append((res.instance_id, offer["dph_total"], time.time()))
            print(f"instance {res.instance_id}: {res.offer}", flush=True)
            try:
                host, port = wait_ready(res.instance_id)
                if spec.min_mb_s > 0:
                    mb_s = download_mb_s(host, port)
                    print(f"download speed {mb_s:.1f} MB/s", flush=True)
                    if mb_s < spec.min_mb_s:
                        raise HostFailed(f"slow network: {mb_s:.1f} MB/s < {spec.min_mb_s}")
                break
            except HostFailed as e:
                print(f"host failed, trying another machine: {e}", flush=True)
                mark_bad_machine(offer.get("machine_id"), str(e))
                _destroy_retrying(res.instance_id)
                bad_machines.add(offer.get("machine_id"))
        else:
            raise VastError(f"{max_hosts} hosts failed to start the container")
        print(f"ready after {time.time() - t0:.0f}s; pushing code", flush=True)
        for attempt in range(3):
            try:
                push_code(host, port, spec.data_paths)
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    mark_bad_machine(res.offer.get("machine_id"), "ssh transfers keep failing (exit 255)")
                    raise
                time.sleep(10)
        cap = int(spec.hours_cap * 3600 - (time.time() - t0))
        if cap < 60:
            raise VastError("time cap used up before the job could start")
        job = f"export PATH=$HOME/.local/bin:$PATH && cd {REMOTE_DIR} && ({spec.setup}) && ({spec.command})"
        res.exit_code = run_detached(host, port, spec.name, job, cap, log)
        print(f"job finished (exit {res.exit_code}) after {time.time() - t0:.0f}s; pulling", flush=True)
        pull_results(host, port, spec.name)
    except Exception as e:  # recorded, then re-raised after cleanup
        res.error = repr(e)
        raise
    finally:
        now = time.time()
        for iid, dph, start in created:
            _destroy_retrying(iid)
            res.est_cost += dph * (now - start) / 3600
        res.seconds = now - t0
        try:
            time.sleep(5)
            res.credit_after = credit()
            left = [i["id"] for i in our_instances()]
            if left:
                res.error += f" STILL RUNNING: {left}"
        except VastError:
            pass
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), **asdict(res)}) + "\n")
        print(
            f"spent ≈ ${res.est_cost:.3f} ({res.seconds / 60:.1f} min); credit ${res.credit_after}",
            flush=True,
        )
    return res
