"""A2b-2 gates: Rust client = Python client, Rust host = Python host, two remote seats, real Multirole.

    uv run python scripts/app/edopro_net_gate.py --gate 1 --duels 24
    uv run python scripts/app/edopro_net_gate.py --gate 2 --duels 24
    uv run python scripts/app/edopro_net_gate.py --gate 3 --duels 20
    uv run python scripts/app/edopro_net_gate.py --gate 4 --duels 10

Every gate streams and gzips its logs under `data/cache/app/` (override with --out). Exit 2 with an `ALERT` line
on any mismatch, unfinished duel or missing artifact. Ports are OS-picked (except Multirole's fixed 7911/7922).

Gates:
1. Python host (StreamBot) + Python client vs Python host + Rust client: the client-side packet logs (sent and
   received sequences) are identical after sorting CARD_HINT/PLAYER_HINT runs (heap order, NOTEBOOK 2026-10-01).
2. Python host + Python client vs Rust host (--no-replay) + Python client: the same comparison.
3. Rust host with two Remote seats + two Rust clients (and once two Python clients): every duel finishes, both
   clients get the same `.yrpX`, its `to_record` (Rust and Python) is equal, passes `match_record.check`, and its
   responses equal the host's match record responses.
4. Real Multirole: two Rust clients (one creates, one joins by room id from the listing), every duel finishes and
   both replays import (`to_record` + `match_record.check`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PAIRS = [
    ("snake-eye", "branded"),
    ("blue-eyes", "burn"),
    ("despia", "dark-world"),
    ("drytron", "eldlich"),
    ("exodia", "burn"),
    ("centur-ion", "blackwing"),
    ("abc-dragon", "altergeist"),
    ("chaos-turbo", "adamancipator"),
]

HINT_TYPES = {0xA0, 0xA5}
STOC_GAME_MSG = 0x1
STOC_NEW_REPLAY = 0x30


def alert(text: str) -> None:
    print(f"ALERT edopro_net_gate: {text}", flush=True)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def deck(name: str) -> Path:
    p = ROOT / "decks" / "pool" / f"{name}.ydk"
    if not p.is_file():
        raise SystemExit(f"deck not found: {p}")
    return p


def start(cmd: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w")  # noqa: SIM115 - the child process owns it
    print("RUN", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def stop(p: subprocess.Popen, timeout: float = 180.0) -> int:
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=10)
        return -9


def load_log(path: Path) -> dict[str, list[dict]]:
    out = {"s": [], "r": []}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["dir"]].append(row)
    return out


def canon(rows: list[dict]) -> list[dict]:
    """Sort each run of consecutive hint messages (their order follows heap addresses)."""
    out: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        nonlocal run
        out.extend(sorted(run, key=lambda r: r["x"]))
        run = []

    for r in rows:
        x = r["x"]
        hint = r["p"] == STOC_GAME_MSG and len(x) >= 2 and int(x[:2], 16) in HINT_TYPES
        if hint:
            run.append(r)
        else:
            flush()
            out.append(r)
    flush()
    return out


def compare_logs(a_path: Path, b_path: Path) -> tuple[bool, dict]:
    a, b = load_log(a_path), load_log(b_path)
    num = {"sent": len(a["s"]), "received": len(a["r"])}
    for d in ("s", "r"):
        ca, cb = canon(a[d]), canon(b[d])
        if len(ca) != len(cb):
            alert(f"{d} frame counts differ: {len(ca)} vs {len(cb)} ({a_path.name} vs {b_path.name})")
            return False, num
        for i, (x, y) in enumerate(zip(ca, cb)):
            if x != y:
                alert(f"{a_path.name} vs {b_path.name}: first {d} difference at frame {i}: {x} != {y}")
                return False, num
    return True, num


def latest(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


# ---------------------------------------------------------------- gate 1/2


def run_one_host_client(
    kind: str, host_kind: str, pair: tuple[str, str], seed: int, host_agent: int, client_agent: int,
    work: Path, tag: str, rust_bin: Path, bot_second: bool = False,
) -> tuple[Path, Path, dict]:
    port = free_port()
    hlog = work / f"{tag}-host.log"
    climit = work / f"{tag}-client.jsonl.gz"
    crec = work / f"{tag}-crec"
    hrec = work / f"{tag}-hrec"
    if host_kind == "py":
        host_cmd = [
            sys.executable, str(ROOT / "scripts/app/edopro_net_py.py"), "host",
            "--deck", str(deck(pair[0])), "--port", str(port), "--seed", str(seed),
            "--agent-seed", str(host_agent), "--once", "--no-discovery",
            "--record-root", str(hrec), "--packet-log", str(work / f"{tag}-host.jsonl.gz"),
            *(["--bot-second"] if bot_second else []),
        ]
        wait = 2.0
    else:
        host_cmd = [
            str(rust_bin / "edopro-host"), "--deck", str(deck(pair[0])), "--port", str(port),
            "--seed", str(seed), "--agent-seed", str(host_agent), "--opponent", "stream",
            "--once", "--no-replay", "--no-discovery", "--record-root", str(hrec),
            *(["--bot-second"] if bot_second else []),
        ]
        wait = 1.0
    hp = start(host_cmd, hlog)
    try:
        time.sleep(wait)
        if kind == "py":
            client_cmd = [
                sys.executable, str(ROOT / "scripts/app/edopro_net_py.py"), "join",
                "--deck", str(deck(pair[1])), "--port", str(port), "--agent-seed", str(client_agent),
                "--record-dir", str(crec), "--packet-log", str(climit),
            ]
        else:
            client_cmd = [
                str(rust_bin / "edopro-join"), "--deck", str(deck(pair[1])), "--port", str(port),
                "--agent", "stream", "--agent-seed", str(client_agent), "--record-dir", str(crec),
                "--packet-log", str(climit),
            ]
        cp = start(client_cmd, work / f"{tag}-client.log")
        cc = stop(cp, timeout=240)
        hc = stop(hp, timeout=30)
    finally:
        stop(hp, timeout=5)
    info = {"host_exit": hc, "client_exit": cc, "port": port}
    recs = sorted(crec.glob("*.json"), key=lambda p: p.stat().st_mtime) if crec.is_dir() else []
    if recs:
        rec = json.loads(recs[-1].read_text())
        info.update({"seat": rec.get("seat"), "winner": rec.get("winner"),
                     "retries": rec.get("retries"), "responses": rec.get("n_responses")})
    if cc != 0:
        alert(f"{tag}: client exit {cc} (host {hc}); see {work / f'{tag}-client.log'}")
    if not climit.is_file():
        alert(f"{tag}: no client packet log")
    return climit, hlog, info


def gate_12(which: int, duels: int, work: Path, rust_bin: Path) -> dict:
    ok, runs = True, 0
    seats = {0: 0, 1: 0}
    totals = {"duels": 0, "frames": 0, "mismatches": 0}
    for i in range(duels):
        pair = PAIRS[i % len(PAIRS)]
        seed = 70_000 + 7 * i
        host_agent = 1000 + i
        client_agent = 2000 + i
        # Alternate the bot's turn-order preference so both who-goes-first cases appear even if the bot wins RPS.
        bot_second = i % 2 == 1
        tag = f"g{which}-{i:02d}-{pair[0]}-{pair[1]}"
        # Reference: Python host + Python client (also the gate 1 variant a and gate 2 variant a).
        ref_log, _, ref_info = run_one_host_client(
            "py", "py", pair, seed, host_agent, client_agent, work, tag + "-py", rust_bin, bot_second)
        if which == 1:
            # variant b: Python host + Rust client
            cand_log, _, cand_info = run_one_host_client(
                "rs", "py", pair, seed, host_agent, client_agent, work, tag + "-rs", rust_bin, bot_second)
        else:
            # variant b: Rust host + Python client
            cand_log, _, cand_info = run_one_host_client(
                "py", "rs", pair, seed, host_agent, client_agent, work, tag + "-rshost", rust_bin, bot_second)
        runs += 1  # noqa: SIM113 - counts duels run, not loop positions
        totals["duels"] += 1
        if ref_log.is_file() and cand_log.is_file():
            same, num = compare_logs(ref_log, cand_log)
            totals["frames"] += num["sent"] + num["received"]
            if not same:
                totals["mismatches"] += 1
                ok = False
        else:
            ok = False
        for info in (ref_info, cand_info):
            if info.get("client_exit") not in (0, None):
                ok = False
            if info.get("seat") in (0, 1):
                seats[info["seat"]] += 1
        # A vetoed RPS throw from the same seeds would tie forever; different agent seeds avoid it.
        print(json.dumps({"i": i, "pair": pair, "seed": seed, "ref": ref_info, "cand": cand_info}), flush=True)
    totals["seats"] = seats
    totals["ok"] = ok and seats[0] > 0 and seats[1] > 0
    if not ok:
        alert(f"gate {which}: {totals['mismatches']} mismatches over {totals['duels']} duels")
    if not (seats[0] > 0 and seats[1] > 0):
        alert(f"gate {which}: did not cover both who-goes-first cases: {seats}")
    return totals


# ---------------------------------------------------------------- gate 3


def rust_host_two(port: int, seed: int, work: Path, tag: str, rust_bin: Path, replay: bool = True) -> subprocess.Popen:
    cmd = [
        str(rust_bin / "edopro-host"), "--port", str(port), "--remote", "2", "--seed", str(seed),
        "--once", "--no-discovery", "--record-root", str(work / f"{tag}-hrec"),
        "--packet-log", str(work / f"{tag}-host.jsonl.gz"),
    ]
    if not replay:
        cmd.append("--no-replay")
    return start(cmd, work / f"{tag}-host.log")


def rust_client(port: int, deck_name: str, agent_seed: int, work: Path, tag: str, rust_bin: Path) -> subprocess.Popen:
    cmd = [
        str(rust_bin / "edopro-join"), "--deck", str(deck(deck_name)), "--port", str(port),
        "--agent", "stream", "--agent-seed", str(agent_seed), "--record-dir", str(work / f"{tag}-rec"),
        "--packet-log", str(work / f"{tag}.jsonl.gz"),
    ]
    return start(cmd, work / f"{tag}.log")


def python_client(port: int, deck_name: str, agent_seed: int, work: Path, tag: str) -> subprocess.Popen:
    cmd = [
        sys.executable, str(ROOT / "scripts/app/edopro_net_py.py"), "join",
        "--deck", str(deck(deck_name)), "--port", str(port), "--agent-seed", str(agent_seed),
        "--record-dir", str(work / f"{tag}-rec"), "--packet-log", str(work / f"{tag}.jsonl.gz"),
        "--save-replay", str(work / f"{tag}.yrpx"),
    ]
    return start(cmd, work / f"{tag}.log")


def read_json_record(rec_dir: Path) -> dict:
    if not rec_dir.is_dir():
        return {}
    files = sorted(rec_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(files[-1].read_text()) if files else {}


def read_host_record(rec_dir: Path) -> dict | None:
    if not rec_dir.is_dir():
        return None
    files = sorted(rec_dir.glob("*.match.json.gz"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    with gzip.open(files[-1], "rt", encoding="utf-8") as f:
        return json.load(f)


def normalize_record(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k not in ("id", "created")}
    return out


def check_replay(path: Path) -> tuple[dict | None, list[str], dict | None]:
    """Python `yrp.to_record` + `match_record.check` for one `.yrpX`."""
    from decklight.play import match_record, yrp

    engine = check_replay.engine  # type: ignore[attr-defined]
    x = yrp.parse_yrpx(path.read_bytes())
    rec, problems = yrp.to_record(x, engine, viewer=None, app="decklight-edopro-client")
    if rec is None:
        return None, problems, None
    return rec, match_record.check(rec, engine), x


def gate_3(duels: int, work: Path, rust_bin: Path) -> dict:
    from decklight.engine.duel import Engine

    check_replay.engine = Engine()  # type: ignore[attr-defined]
    ok, seats = True, {0: 0, 1: 0}
    stats = {"duels": 0, "yrpx_ok": 0, "check_ok": 0, "same_yrpx": 0, "host_responses_ok": 0}
    for i in range(duels):
        pair = PAIRS[i % len(PAIRS)]
        port = free_port()
        seed = 80_000 + 11 * i
        tag = f"g3-{i:02d}"
        hrec = work / f"{tag}-hrec"
        hp = rust_host_two(port, seed, work, tag, rust_bin)
        time.sleep(1.0)
        a = rust_client(port, pair[0], 3000 + i, work, tag + "-a", rust_bin)
        b = rust_client(port, pair[1], 4000 + i, work, tag + "-b", rust_bin)
        ha = stop(a, timeout=240)
        hb = stop(b, timeout=240)
        hh = stop(hp, timeout=60)
        rec_a = read_json_record(work / f"{tag}-a-rec")
        rec_b = read_json_record(work / f"{tag}-b-rec")
        stats["duels"] += 1
        ra = latest(work / f"{tag}-a-rec", "*.yrpx")
        rb = latest(work / f"{tag}-b-rec", "*.yrpx")
        host_rec = read_host_record(hrec)
        problems = []
        add_order = None
        if not (ha == 0 and hb == 0 and hh == 0):
            problems.append(f"exit codes a={ha} b={hb} host={hh}")
        if rec_a.get("winner") != rec_b.get("winner"):
            problems.append(f"winners differ: {rec_a.get('winner')} vs {rec_b.get('winner')}")
        for r in (rec_a, rec_b):
            if r.get("seat") in (0, 1):
                seats[r["seat"]] += 1
        if ra is None or rb is None:
            problems.append("a client kept no .yrpX")
        elif ra.read_bytes() != rb.read_bytes():
            problems.append(f"the two clients got different .yrpX ({ra.stat().st_size} vs {rb.stat().st_size})")
        else:
            stats["same_yrpx"] += 1
            rec_py, check, x = check_replay(ra)
            if rec_py is None:
                problems.append(f"python to_record failed: {check}")
            else:
                add_order = rec_py.get("add_order")
                stats["yrpx_ok"] += 1
                if check:
                    problems.append(f"match_record.check: {check}")
                else:
                    stats["check_ok"] += 1
                # Rust client's imported record next to the replay.
                rust_rec_path = latest(work / f"{tag}-a-rec", "*.match.json.gz")
                if rust_rec_path is None:
                    problems.append("no Rust imported record")
                else:
                    with gzip.open(rust_rec_path, "rt", encoding="utf-8") as f:
                        rust_rec = json.load(f)
                    if normalize_record(rust_rec) != normalize_record(rec_py):
                        problems.append("Rust and Python to_record JSON differ")
                if host_rec is None:
                    problems.append("no host match record")
                else:
                    hres = [e["r"] for e in host_rec.get("responses", [])]
                    yres = [r.hex() for r in x.yrp.responses]
                    if hres != yres:
                        problems.append(f"host responses {len(hres)} != replay responses {len(yres)}")
                    else:
                        stats["host_responses_ok"] += 1
                    if host_rec.get("result", {}).get("winner") != rec_py["result"]["winner"]:
                        problems.append("host winner != replay winner")
        if problems:
            ok = False
            alert(f"gate 3 duel {i}: " + "; ".join(problems))
        print(json.dumps({"i": i, "pair": pair, "seed": seed, "a": rec_a.get("winner"),
                          "b": rec_b.get("winner"), "seat_a": rec_a.get("seat"),
                          "seat_b": rec_b.get("seat"), "problems": problems,
                          "add_order": add_order}), flush=True)

    # Once with two Python clients (they only need to finish, and both receive the same replay).
    for j in range(2):
        pair = PAIRS[j % len(PAIRS)]
        port = free_port()
        seed = 85_000 + j
        tag = f"g3-py-{j}"
        hp = rust_host_two(port, seed, work, tag, rust_bin)
        time.sleep(1.0)
        a = python_client(port, pair[0], 5000 + j, work, tag + "-a")
        b = python_client(port, pair[1], 6000 + j, work, tag + "-b")
        ha = stop(a, timeout=240)
        hb = stop(b, timeout=240)
        hh = stop(hp, timeout=60)
        ya = work / f"{tag}-a.yrpx"
        yb = work / f"{tag}-b.yrpx"
        ya = ya if ya.is_file() else None
        yb = yb if yb.is_file() else None
        problems = []
        if ha != 0 or hb != 0 or hh != 0:
            problems.append(f"python clients exit a={ha} b={hb} host={hh}")
        if ya is None or yb is None or ya.read_bytes() != yb.read_bytes():
            problems.append("python clients did not both keep the same replay")
        if problems:
            ok = False
            alert(f"gate 3 python clients {j}: " + "; ".join(problems))
        stats["duels"] += 1

    stats["seats"] = seats
    stats["ok"] = ok and seats[0] > 0 and seats[1] > 0
    if not (seats[0] > 0 and seats[1] > 0):
        alert(f"gate 3: did not cover both who-goes-first cases: {seats}")
    return stats


# ---------------------------------------------------------------- gate 4


def multirole_running() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:7922/", timeout=2) as r:
            json.load(r)
        return True
    except Exception:  # noqa: BLE001
        return False


def find_room(note: str, timeout: float = 30.0) -> int:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:7922/", timeout=3) as r:
                rooms = json.load(r).get("rooms", [])
        except Exception:  # noqa: BLE001
            rooms = []
        for room in rooms:
            if room.get("roomnotes") == note:
                return int(room["roomid"])
        time.sleep(0.2)
    raise TimeoutError(f"room {note!r} never appeared")


def gate_4(duels: int, work: Path, rust_bin: Path) -> dict:
    import urllib.request  # noqa: F401  (multirole_running/find_room use it)

    from decklight.engine.duel import Engine
    from decklight.play import match_record, yrp

    engine = Engine()
    if not multirole_running():
        alert("gate 4: Multirole is not running on 127.0.0.1:7911/7922 (see scripts/multirole_local.sh)")
        return {"ok": False, "reason": "multirole not running"}
    ok, seats = True, {0: 0, 1: 0}
    stats = {"duels": 0, "add_order_found": 0, "check_ok": 0}
    build = ROOT / "data/cache/multirole/Multirole/build/multirole"
    stats["build_mb"] = round(build.stat().st_size / 1e6, 1) if build.is_file() else 0
    for i in range(duels):
        pair = PAIRS[i % len(PAIRS)]
        note = f"decklight-a2b2-{os.getpid()}-{i}"
        tag = f"g4-{i:02d}"
        # The creator starts first and waits for the joiner; it sends HS_START once the joiner is ready.
        host_cmd = [
            str(rust_bin / "edopro-join"), "--deck", str(deck(pair[0])), "--port", "7911",
            "--agent", "stream", "--agent-seed", str(7000 + i), "--create", "--rule", "3",
            "--room-notes", note, "--record-dir", str(work / f"{tag}-hrec"),
            "--packet-log", str(work / f"{tag}-host.jsonl.gz"),
        ]
        hp = start(host_cmd, work / f"{tag}-host.log")
        problems = []
        try:
            room = find_room(note, 60)
        except TimeoutError as e:
            problems.append(str(e))
            room = None
        if room is not None:
            jcmd = [
                str(rust_bin / "edopro-join"), "--deck", str(deck(pair[1])), "--port", "7911",
                "--roomid", str(room), "--agent", "stream", "--agent-seed", str(8000 + i),
                "--record-dir", str(work / f"{tag}-jrec"),
                "--packet-log", str(work / f"{tag}-join.jsonl.gz"),
            ]
            jp = start(jcmd, work / f"{tag}-join.log")
            hj = stop(jp, timeout=300)
            hh = stop(hp, timeout=120)
        else:
            hj, hh = None, stop(hp, timeout=10)
        recs = {}
        for name, d in (("host", work / f"{tag}-hrec"), ("join", work / f"{tag}-jrec")):
            r = read_json_record(d)
            recs[name] = r
            if r.get("seat") in (0, 1):
                seats[r["seat"]] += 1
        stats["duels"] += 1
        if hh != 0 or hj != 0:
            problems.append(f"exit codes host={hh} join={hj}")
        if recs["host"].get("winner") != recs["join"].get("winner"):
            problems.append(f"winners differ host={recs['host'].get('winner')} join={recs['join'].get('winner')}")
        for name, d in (("host", work / f"{tag}-hrec"), ("join", work / f"{tag}-jrec")):
            y = latest(d, "*.yrpx")
            if y is None:
                problems.append(f"{name} kept no replay")
                continue
            try:
                x = yrp.parse_yrpx(y.read_bytes())
                rec, to_problems = yrp.to_record(x, engine, viewer=None, app="gate4")
            except Exception as e:  # noqa: BLE001
                problems.append(f"{name} replay failed: {type(e).__name__}: {e}")
                continue
            if rec is None:
                problems.append(f"{name} to_record: {to_problems}")
                continue
            if rec.get("add_order"):
                stats["add_order_found"] += 1
            check = match_record.check(rec, engine)
            if check:
                problems.append(f"{name} check: {check}")
            else:
                stats["check_ok"] += 1
        if problems:
            ok = False
            alert(f"gate 4 duel {i}: " + "; ".join(problems))
        print(json.dumps({"i": i, "pair": pair, "room": room, "problems": problems,
                          "host": recs["host"].get("winner"), "join": recs["join"].get("winner")}), flush=True)
    stats["seats"] = seats
    stats["ok"] = ok and seats[0] > 0 and seats[1] > 0
    # The brief's gate 4 ends with Multirole stopped (its PID file is written by scripts/multirole_local.sh).
    pid_file = ROOT / "data/cache/multirole/run/multirole.pid"
    if pid_file.is_file():
        try:
            os.kill(int(pid_file.read_text().strip()), 15)
            print("stopped Multirole", flush=True)
        except (OSError, ValueError):
            pass
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", type=int, choices=[1, 2, 3, 4], required=True)
    ap.add_argument("--duels", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "data/cache/app")
    ap.add_argument("--rust-bin-dir", type=Path, default=ROOT / "rust/target/debug")
    a = ap.parse_args()
    n = a.duels if a.duels is not None else (10 if a.gate == 4 else 20)
    work = a.out / f"gate{a.gate}-{time.strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True, exist_ok=True)
    if a.gate in (1, 2):
        stats = gate_12(a.gate, n, work, a.rust_bin_dir)
    elif a.gate == 3:
        stats = gate_3(n, work, a.rust_bin_dir)
    else:
        stats = gate_4(n, work, a.rust_bin_dir)
    print(json.dumps({"gate": a.gate, "out": str(work), "stats": stats}), flush=True)
    return 0 if stats.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
