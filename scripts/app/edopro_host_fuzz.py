"""Fuzz a running `edopro-host` with malformed CTOS traffic; the host must stay up (no panic) and still serve a real
duel afterwards (run `edopro-join` against it). A2b-2 review, NOTEBOOK 2026-11-01.

    rust/target/release/edopro-host --deck decks/pool/branded.ydk --opponent stream --port 47911 --no-discovery \
        --seed 5 --record-root data/cache/app/claude/fuzz/rec &
    uv run python scripts/app/edopro_host_fuzz.py 47911 <client packet log .jsonl.gz> 400        # lobby junk
    uv run python scripts/app/edopro_host_fuzz.py 47911 <client packet log .jsonl.gz> 120 deep   # junk mid-duel

The packet log is any client-side `--packet-log` (its first sent packets give a valid lobby prefix).
"""
import gzip
import json
import random
import socket
import struct
import sys
import time

port, log, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
with gzip.open(log, "rt") as f:
    pre = [(r["p"], bytes.fromhex(r["x"])) for r in map(json.loads, f) if r["dir"] == "s"][:5]
rng = random.Random(5)
PROTOS = [1, 2, 3, 4, 0x10, 0x11, 0x12, 0x13, 0x14, 0x16, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x22]
def frame(p, x): return struct.pack("<HB", len(x) + 1, p) + x
stats = {"conns": 0, "frames": 0, "refused": 0}
for i in range(n):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=3)
    except OSError:
        stats["refused"] += 1; time.sleep(0.2); continue
    stats["conns"] += 1
    try:
        deep = len(sys.argv) > 4
        for p, x in (pre[:4] if deep else pre[: rng.randrange(len(pre) + 1)]):
            s.sendall(frame(p, x))
        if deep:
            pre4 = [q for q in pre if q[0] != 3]
            buf, started, t0 = b"", False, time.time()
            s.settimeout(0.5)
            while not started and time.time() - t0 < 8:
                try:
                    chunk = s.recv(65536)
                except TimeoutError:
                    continue
                if not chunk:
                    break
                buf += chunk
                while len(buf) >= 3 and len(buf) >= 2 + struct.unpack_from("<H", buf)[0]:
                    ln = struct.unpack_from("<H", buf)[0]; pr = buf[2]; buf = buf[2 + ln:]
                    if pr == 3: s.sendall(frame(3, bytes([rng.randrange(1, 4)])))
                    elif pr == 4: s.sendall(frame(4, bytes([rng.randrange(2)])))
                    elif pr == 1: started = True
            stats["started"] = stats.get("started", 0) + started
            for _ in range(rng.randrange(1, 60)):
                ln = rng.choice([0, 1, 2, 3, 4, 4, 4, 8, 16, 33])
                s.sendall(frame(1, bytes(rng.randrange(256) for _ in range(ln))))
                stats["frames"] += 1
                time.sleep(0.005)
        for _ in range(rng.randrange(1, 40)):
            k = rng.random()
            if k < 0.1:
                s.sendall(bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64))))
            elif k < 0.15:
                s.sendall(struct.pack("<H", rng.choice([0, 1, 2, 65535])) + b"\x01")
            else:
                p = rng.choice(PROTOS) if rng.random() < 0.9 else rng.randrange(256)
                x = bytes(rng.randrange(256) for _ in range(rng.choice([0, 1, 2, 4, 8, 40, 80, 300])))
                s.sendall(frame(p, x))
            stats["frames"] += 1
        s.settimeout(0.3)
        try:
            while s.recv(65536): pass
        except OSError: pass
    except OSError:
        pass
    finally:
        s.close()
print(json.dumps(stats))
