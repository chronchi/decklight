"""Record a full WebSocket transcript from the Python or Rust server (A2c parity harness).

    uv run python scripts/app/ws_transcript.py --server python --games 40 --seed 21000 \\
        --pairs blue-eyes:branded branded:burn ... --out data/cache/app/py.jsonl.gz

    uv run python scripts/app/ws_transcript.py --server rust --rust-bin rust/target/debug/decklight-serve ...

For every game: `new_game` with the pair's decks as ydk text, `first` alternating me/bot, `chain` alternating
auto/all, and every prompt answered with the client's own splitmix64 stream (seed ^ 0xA5A5 ^ g). Once per run the
client also does a bad-token connection, an unknown message and a `resync`, and surrenders one game mid-game.
Timer-driven ticker frames (`thinking` not null) are not recorded. Frames are streamed to a gzip JSONL file.

`--learn` (A4, product mode): every game is a practice game; the client asks for a `hint` at its 2nd and 5th prompt
and sends one `undo` at its 4th (after it has chosen), and surrenders at its `--max-prompts`th prompt so a gate game
stays short. Win chance is on from game 1, so game 0's hints carry no estimate and later ones do.

`--review` (with `--learn`): after every game the client asks for its `review` (progress frames, then the review),
`review_frame` at the first and last decision, two errors (an unknown decision, a surrender after the end); after
game 0 also a `whatif` at the middle decision with that prompt's last option, played (same splitmix stream) to a
surrender at its 6th prompt and reviewed too.
"""

from __future__ import annotations

import argparse
import gzip
import json
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.app.duel_hash import SplitMix


def resolve_deck(name: str) -> Path:
    p = Path(name)
    if p.is_file():
        return p
    q = ROOT / "decks" / "pool" / f"{name}.ydk"
    return q if q.is_file() else p


def wait_listening(proc: subprocess.Popen, log: Path, timeout: float = 60.0) -> int:
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = proc.stdout.readline() if proc.stdout is not None else b""
        if not line:
            if proc.poll() is not None:
                tail = log.read_text(errors="replace")[-2000:] if log.is_file() else ""
                raise RuntimeError(f"server exited before listening (rc={proc.returncode})\n{tail}")
            time.sleep(0.05)
            continue
        text = line.decode(errors="replace").strip()
        if "listening on" in text:
            return int(text.rsplit(" ", 1)[1])
    raise RuntimeError("server did not print 'listening on <port>'")


def start_server(args, run_dir: Path) -> tuple[subprocess.Popen, int, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    py_records = run_dir / "py_runs"
    rs_records = run_dir / "records"
    pics = run_dir / "pics"
    log = run_dir / "server.log"
    if args.server == "python":
        wrapper = "web_play_onnx.py" if args.mode == "product" else "py_server.py"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/app" / wrapper),
            "--port",
            "0",
            "--token",
            token,
            "--seed",
            str(args.seed),
            "--records",
            str(py_records),
            "--pics",
            str(pics),
        ]
        if args.mode == "product":
            cmd += ["--replay-threads", str(args.replay_threads)]
    elif args.server == "rust":
        cmd = [
            str(args.rust_bin),
            "--port",
            "0",
            "--token",
            token,
            "--records",
            str(rs_records),
            "--pics-dir",
            str(pics),
            "--root",
            str(ROOT),
        ]
        if not args.no_test_hooks:
            cmd += ["--fixed-seed", str(args.seed)]
            if args.mode != "product":
                cmd += ["--test-opponent", "stream"]
        if args.mode == "product":
            cmd += ["--replay-threads", str(args.replay_threads)]
    else:
        cmd = shlex.split(args.server)
    fh = log.open("wb")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=fh)
    try:
        port = wait_listening(proc, log)
    except Exception:
        proc.kill()
        fh.close()
        raise
    proc._log_fh = fh
    proc._cmd = cmd
    # Keep draining stdout (the Python server prints its ALERTs there): a full pipe would stall the server, and the
    # lines are the evidence when a run goes wrong.
    out_fh = (run_dir / "server.out").open("wb")
    threading.Thread(target=shutil.copyfileobj, args=(proc.stdout, out_fh), daemon=True).start()
    proc._out_fh = out_fh
    return proc, port, token


async def bad_token(port: int, token: str) -> int:
    import websockets

    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws?t={token}x") as ws:
            await ws.recv()
            return 0
    except websockets.exceptions.ConnectionClosed as e:
        return int(e.code or 0)
    except websockets.exceptions.InvalidStatus as e:
        return int(e.response.status_code)


async def record(args) -> int:
    import websockets

    pairs = []
    for item in args.pairs:
        a, sep, b = item.partition(":")
        if not sep:
            raise SystemExit(f"--pairs entry {item!r} needs deck0:deck1")
        pairs.append((resolve_deck(a), resolve_deck(b)))
    if not pairs and not args.puzzles:
        raise SystemExit("--pairs is required")
    decks = [(p.read_text(encoding="utf-8"), q.read_text(encoding="utf-8")) for p, q in pairs]

    run_dir = (
        Path(args.run_dir) if args.run_dir else ROOT / "data/cache/app/transcript" / f"run-{int(time.time())}"
    )
    if args.port:
        if not args.token:
            raise SystemExit("--port needs --token (an already running server)")
        proc, port, token = None, args.port, args.token
    else:
        proc, port, token = start_server(args, run_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # learn mode surrenders every game at --max-prompts instead of one game at its first prompt
    surrender_g = args.games // 2 if args.games >= 2 and not args.learn else -1
    t0 = time.time()
    n_frames = 0
    try:
        with gzip.open(out, "wt", encoding="utf-8") as fh:

            def write(g, frame):
                nonlocal n_frames
                fh.write(json.dumps({"g": g, "frame": frame}, separators=(",", ":")) + "\n")
                n_frames += 1

            # A server we attached to (--port) may sit behind adb/Tailscale: no keepalive pings there (the web
            # client sends none either); a long search must not look like a dead link.
            ping = None if args.port else 20
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws?t={token}", max_size=None, ping_interval=ping) as ws:
                write(None, json.loads(await ws.recv()))
                write(None, {"close": await bad_token(port, token)})
                await ws.send(json.dumps({"t": "bogus"}))
                write(None, json.loads(await ws.recv()))

                if args.puzzles:
                    await puzzle_round(ws, write)
                    await ws.close()
                    print(json.dumps({"frames": n_frames, "puzzles": True, "wall_s": round(time.time() - t0, 1),
                                      "out": str(out), "server": args.server, "run_dir": str(run_dir)}),
                          file=sys.stderr)
                    return 0

                for g in range(args.games):
                    human, bot = decks[g % len(decks)]
                    msg = {
                        "t": "new_game",
                        "deck": {"ydk": human},
                        "bot_deck": {"ydk": bot},
                        "first": "me" if g % 2 == 0 else "bot",
                        "chain": "auto" if g % 2 == 0 else "all",
                        "mode": args.mode,
                    }
                    if args.learn:
                        msg["practice"] = True
                    await ws.send(json.dumps(msg))
                    rng = SplitMix(args.seed ^ 0xA5A5 ^ g)
                    did_resync = g != 0
                    did_surrender = False
                    did_winchance = False
                    winchance_at = 1
                    n_prompts = 0  # human prompts seen in this game (learn mode)
                    chosen = 0
                    asked: set[int] = set()
                    while True:
                        raw = await ws.recv()
                        frame = json.loads(raw)
                        if frame.get("thinking") is not None:
                            continue
                        write(g, frame)
                        if frame.get("t") == "error":
                            raise RuntimeError(f"game {g}: server error {frame}")
                        if frame.get("result") is not None:
                            break
                        if frame.get("prompt") is not None:
                            if not did_resync:
                                did_resync = True
                                await ws.send(json.dumps({"t": "resync"}))
                                continue
                            if g == surrender_g and not did_surrender:
                                did_surrender = True
                                await ws.send(json.dumps({"t": "surrender"}))
                                continue
                            if args.mode == "product" and g >= winchance_at and not did_winchance:
                                did_winchance = True
                                await ws.send(json.dumps({"t": "settings", "winchance": True}))
                                continue
                            if args.learn:
                                pid = frame["prompt"]["id"]
                                key = n_prompts
                                if key not in asked:
                                    asked.add(key)
                                    if key in (1, 4):
                                        await ws.send(json.dumps({"t": "hint"}))
                                        hint = json.loads(await ws.recv())
                                        write(g, hint)
                                        if hint.get("t") != "hint" or hint.get("prompt") != pid:
                                            raise RuntimeError(f"game {g}: hint answer {hint}")
                                    if key == 3 and chosen > 0:
                                        await ws.send(json.dumps({"t": "undo"}))
                                        n_prompts += 1
                                        continue
                                if n_prompts >= args.max_prompts and not did_surrender:
                                    did_surrender = True
                                    await ws.send(json.dumps({"t": "surrender"}))
                                    continue
                                n_prompts += 1
                                chosen += 1
                            n = len(frame["prompt"]["options"])
                            await ws.send(
                                json.dumps(
                                    {"t": "choose", "prompt": frame["prompt"]["id"], "index": rng.next() % n}
                                )
                            )
                    if args.review:
                        await review_round(ws, write, g, rng, whatif=g == 0)
                    print(f"game {g}: {args.seed + g}", file=sys.stderr, flush=True)
                await ws.close()
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            if hasattr(proc, "_log_fh"):
                proc._log_fh.close()
            if hasattr(proc, "_out_fh"):
                proc._out_fh.close()
    print(
        json.dumps(
            {
                "frames": n_frames,
                "games": args.games,
                "wall_s": round(time.time() - t0, 1),
                "out": str(out),
                "server": args.server,
                "run_dir": str(run_dir),
            }
        ),
        file=sys.stderr,
    )
    return 0


async def recv_frame(ws, write, g):
    while True:
        frame = json.loads(await ws.recv())
        if frame.get("thinking") is not None:
            continue
        write(g, frame)
        return frame


async def review_round(ws, write, g, rng, *, whatif: bool) -> None:
    """Review the finished game; with `whatif`, also play a what-if game from its middle decision and review it."""
    await ws.send(json.dumps({"t": "review"}))
    while True:
        frame = await recv_frame(ws, write, g)
        if frame.get("t") == "review":
            break
        if frame.get("t") != "review_progress":
            raise RuntimeError(f"game {g}: review answer {frame}")
    decs = frame["decisions"]
    if not decs:
        raise RuntimeError(f"game {g}: review with no decisions")
    for k in sorted({0, len(decs) - 1}):
        await ws.send(json.dumps({"t": "review_frame", "k": k}))
        f = await recv_frame(ws, write, g)
        if f.get("t") != "review_frame" or f.get("k") != k:
            raise RuntimeError(f"game {g}: review_frame answer {f}")
    await ws.send(json.dumps({"t": "review_frame", "k": len(decs)}))
    if (await recv_frame(ws, write, g)).get("t") != "error":
        raise RuntimeError(f"game {g}: review_frame past the end was not an error")
    await ws.send(json.dumps({"t": "surrender"}))
    if (await recv_frame(ws, write, g)).get("t") != "error":
        raise RuntimeError(f"game {g}: surrender after the end was not an error")
    if not whatif:
        return
    k = len(decs) // 2
    await ws.send(json.dumps({"t": "review_frame", "k": k}))
    f = await recv_frame(ws, write, g)
    option = len(f["options"]) - 1
    await ws.send(json.dumps({"t": "whatif", "k": k, "option": option}))
    prompts = 0
    while True:
        frame = await recv_frame(ws, write, g)
        if frame.get("t") == "error":
            raise RuntimeError(f"game {g}: what-if error {frame}")
        if frame.get("result") is not None:
            break
        if frame.get("prompt") is not None:
            prompts += 1
            if prompts >= 6:
                await ws.send(json.dumps({"t": "surrender"}))
                continue
            n = len(frame["prompt"]["options"])
            await ws.send(json.dumps({"t": "choose", "prompt": frame["prompt"]["id"], "index": rng.next() % n}))
    await review_round(ws, write, g, rng, whatif=False)


async def puzzle_round(ws, write) -> None:
    """A4b: list puzzles, play one to a solve/fail, ask the other's solution, and call primer + study once."""
    await ws.send(json.dumps({"t": "puzzles"}))
    listed = await recv_frame(ws, write, 0)
    if listed.get("t") != "puzzles" or len(listed.get("items") or []) < 2:
        raise RuntimeError(f"puzzles answer {listed}")
    items = listed["items"]

    async def until_prompt():
        while True:
            f = await recv_frame(ws, write, 0)
            if f.get("t") == "error":
                raise RuntimeError(f"puzzle: server error {f}")
            puzzle = f.get("puzzle") or {}
            if f.get("prompt") is not None or f.get("result") is not None or puzzle.get("status") not in (None, "playing"):
                return f

    await ws.send(json.dumps({"t": "puzzle", "id": items[0]["id"]}))
    first = await until_prompt()
    # primer and study once each, on the live game's decks
    await ws.send(json.dumps({"t": "primer"}))
    primer = await recv_frame(ws, write, 0)
    if primer.get("t") != "primer":
        raise RuntimeError(f"primer answer {primer}")
    await ws.send(json.dumps({"t": "study"}))
    study = await recv_frame(ws, write, 0)
    if study.get("t") not in ("error", "study_status"):
        raise RuntimeError(f"study answer {study}")

    done = first.get("result") is not None or (first.get("puzzle") or {}).get("status") not in (None, "playing")
    while not done and first.get("prompt") is not None:
        await ws.send(json.dumps({"t": "choose", "prompt": first["prompt"]["id"], "index": 0}))
        while True:
            f = await recv_frame(ws, write, 0)
            if f.get("t") == "error":
                raise RuntimeError(f"puzzle: server error {f}")
            puzzle = f.get("puzzle") or {}
            if f.get("result") is not None or puzzle.get("status") in ("solved", "failed"):
                done = True
                break
            if f.get("prompt") is not None:
                first = f
                break
    if not done:
        raise RuntimeError("puzzle 0 neither solved nor failed")

    await ws.send(json.dumps({"t": "puzzle", "id": items[1]["id"]}))
    await until_prompt()
    await ws.send(json.dumps({"t": "puzzle_solution", "id": items[1]["id"]}))
    while True:
        f = await recv_frame(ws, write, 0)
        if f.get("t") == "puzzle_solution":
            break
        if f.get("t") == "error":
            raise RuntimeError(f"puzzle_solution answer {f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="rust", help="python | rust | a shell command")
    ap.add_argument("--rust-bin", type=Path, default=ROOT / "rust/target/debug/decklight-serve")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--mode", choices=("fast", "product"), default="fast")
    ap.add_argument("--replay-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=ROOT / "data/cache/app/ws_transcript.jsonl.gz")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--no-test-hooks", action="store_true", help="do not pass --fixed-seed/--test-opponent")
    ap.add_argument("--learn", action="store_true", help="practice games with hint and undo (A4)")
    ap.add_argument("--review", action="store_true", help="learn mode: review, review_frame and what-if (A4)")
    ap.add_argument("--puzzles", action="store_true", help="A4b: list/play puzzles, solution, primer and study")
    ap.add_argument("--max-prompts", type=int, default=12, help="learn mode: surrender at this human prompt")
    ap.add_argument("--port", type=int, default=None, help="connect to an already running server")
    ap.add_argument("--token", default=None, help="token of an already running server")
    a = ap.parse_args(argv)
    import asyncio

    return asyncio.run(record(a))


if __name__ == "__main__":
    raise SystemExit(main())
