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
    return json.loads(out.stdout) if raw else out.stdout


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


def destroy_strays() -> list[int]:
    """Destroy every instance we labelled (left over from a crashed launcher)."""
    ids = [i["id"] for i in our_instances()]
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


def find_offer(spec: JobSpec, exclude_machines: set[int] = frozenset()) -> dict:
    q = (f"reliability>0.98 rentable=true verified=true disk_space>={spec.disk_gb} inet_down>=200 "
         f"cpu_cores_effective>={spec.min_cpus} dph<={spec.max_dph}")  # fmt: skip
    q += f" num_gpus=1 gpu_ram>={spec.min_gpu_ram_gb}" if spec.gpu else ""
    q += f" {spec.extra_query}" if spec.extra_query else ""
    offers = [
        o for o in cli("search", "offers", q, "-o", "dph") if o.get("machine_id") not in exclude_machines
    ]
    if not offers:
        raise VastError(f"no offer matches: {q}")
    return offers[0]


def _ssh_target(inst: dict) -> tuple[str, int]:
    return inst["ssh_host"], int(inst["ssh_port"])


def wait_ready(instance_id: int, timeout_s: int = 900) -> tuple[str, int]:
    """Wait until the instance runs and answers ssh."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        inst = cli("show", "instance", str(instance_id))
        msg = inst.get("status_msg") or ""
        if inst.get("intended_status") == "stopped" or "Error" in msg or "failed" in msg:
            raise HostFailed(f"instance {instance_id}: {msg.strip()[:300]}")
        if inst.get("actual_status") == "running" and inst.get("ssh_host"):
            host, port = _ssh_target(inst)
            ok = subprocess.run(
                ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", "true"], capture_output=True, check=False
            )
            if ok.returncode == 0:
                return host, port
        time.sleep(15)
    raise VastError(f"instance {instance_id} not ready after {timeout_s}s")


def ssh(host: str, port: int, command: str, timeout_s: float | None = None, log: Path | None = None) -> int:
    cmd = ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", command]
    with open(log, "a") if log else open("/dev/null", "w") as out:
        try:
            return subprocess.run(
                cmd, stdout=out, stderr=subprocess.STDOUT, timeout=timeout_s, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            return 124


def push_code(host: str, port: int) -> None:
    excl = " ".join(f"--exclude={shlex.quote(e)}" for e in EXCLUDES)
    remote = f"mkdir -p {REMOTE_DIR} && tar xzf - -C {REMOTE_DIR}"
    pipe = f"tar czf - {excl} -C {shlex.quote(str(ROOT))} . | ssh {' '.join(SSH_OPTS)} -p {port} root@{host} {shlex.quote(remote)}"
    subprocess.run(pipe, shell=True, check=True)


def pull_results(host: str, port: int, name: str) -> Path:
    dest = ROOT / "runs" / name
    dest.mkdir(parents=True, exist_ok=True)
    remote = f"cd {REMOTE_DIR}/runs/{shlex.quote(name)} 2>/dev/null && tar czf - . || true"
    pipe = f"ssh {' '.join(SSH_OPTS)} -p {port} root@{host} {shlex.quote(remote)} | tar xzf - -C {shlex.quote(str(dest))}"
    subprocess.run(pipe, shell=True, check=False)
    return dest


def _destroy_retrying(instance_id: int) -> None:
    for attempt in range(5):
        try:
            destroy(instance_id)
            return
        except VastError as e:
            print(f"destroy {instance_id} attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10)


def run_job(spec: JobSpec, reserve: float = 0.25, max_hosts: int = 3) -> JobResult:
    """The whole lifecycle. Every instance created is destroyed in `finally`, whatever happens. A host that
    cannot start the container is dropped and the next-cheapest offer on another machine is tried."""
    res = JobResult(spec.name)
    log = ROOT / "runs" / spec.name / "launcher.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    strays = destroy_strays()
    if strays:
        print(f"destroyed stray instances {strays}")
    res.credit_before = credit()
    created: list[tuple[int, float, float]] = []  # (instance id, $/h, start time)
    bad_machines: set[int] = set()
    t0 = time.time()
    try:
        for _ in range(max_hosts):
            offer = find_offer(spec, bad_machines)
            res.offer = {
                k: offer.get(k)
                for k in ("id", "machine_id", "gpu_name", "dph_total", "cpu_cores_effective", "geolocation")
            }
            worst = offer["dph_total"] * (spec.hours_cap + 0.25)  # + boot/pull margin
            if worst > res.credit_before - reserve:
                raise VastError(
                    f"worst-case ${worst:.2f} exceeds credit ${res.credit_before:.2f} - reserve ${reserve}"
                )
            out = cli("create", "instance", str(offer["id"]), "--image", spec.image, "--disk", str(spec.disk_gb),
                      "--ssh", "--direct", "--label", LABEL_PREFIX + spec.name)  # fmt: skip
            res.instance_id = int(out["new_contract"])
            created.append((res.instance_id, offer["dph_total"], time.time()))
            print(f"instance {res.instance_id}: {res.offer}", flush=True)
            try:
                host, port = wait_ready(res.instance_id)
                break
            except HostFailed as e:
                print(f"host failed, trying another machine: {e}", flush=True)
                _destroy_retrying(res.instance_id)
                bad_machines.add(offer.get("machine_id"))
        else:
            raise VastError(f"{max_hosts} hosts failed to start the container")
        print(f"ready after {time.time() - t0:.0f}s; pushing code", flush=True)
        push_code(host, port)
        cap = int(spec.hours_cap * 3600 - (time.time() - t0))
        if cap < 60:
            raise VastError("time cap used up before the job could start")
        job = f"cd {REMOTE_DIR} && ({spec.setup}) && ({spec.command})"
        rlog = f"{REMOTE_DIR}/runs/{spec.name}/job.log"
        full = f"mkdir -p {REMOTE_DIR}/runs/{spec.name} && timeout {cap} bash -lc {shlex.quote(job)} > {rlog} 2>&1"
        res.exit_code = ssh(host, port, full, timeout_s=cap + 120, log=log)
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
