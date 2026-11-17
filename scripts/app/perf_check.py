#!/usr/bin/env python3
"""App load-time and bot-move check (user 2026-11-11: "make sure the app loads very quickly from opening and also
playing against the bot too").

    uv run --no-sync python scripts/app/perf_check.py --label before
    uv run --no-sync python scripts/app/perf_check.py --label after --rust-bin rust/target/release/decklight-serve

Starts the release server against the product bot (`data/cache/onnx/y`) and measures:
  ready_s          process start -> "listening on" (3 runs, median)
  lobby_s          WS connect -> first lobby frame (3 runs, median)
  first_prompt_*   new_game -> first frame that waits for the human, `first: me` and `first: bot`, 3 games each
                   on mitsurugi vs the lobby's default bot deck (sky-striker-meta1) and blue-eyes vs it
  bot_move_s       2 short product games (fixed seeded choices, surrender at prompt 8 or when the 90 s phase
                   cap is reached -- each bot interval can hold several searches, so a longer game would put
                   the check over its 5-minute budget): our answer -> next frame that needs us, split by
                   whether a move in that window searched (server timing lines); a game that ends sooner is fine

The server runs with DECKLIGHT_TIMING=1 (startup / new-game phase timings, `TIMING <label> <seconds>`) and
DECKLIGHT_PLAY_TIMING=1 (one `product: search k=... t=...` line per searched move). Writes JSON to
data/cache/app_speed/<label>.json and exits non-zero when a measurement is missing or a game never prompts.

Smoke-sized: the product bot searches on its own turns, so the bot-first openings and the two short games dominate;
keep the whole run under ~5 min on the laptop.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import queue
import secrets
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_gate import SplitMix  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BOT_DECK = "sky-striker-meta1"
DECKS = ["mitsurugi", "blue-eyes"]


def deck_text(name: str) -> str:
    p = Path(name)
    if not p.is_file():
        p = ROOT / "decks/pool" / f"{name}.ydk"
    return p.read_text(encoding="utf-8")


def median(xs: list[float]) -> float:
    return round(statistics.median(xs), 4)


def percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return round(s[round((len(s) - 1) * p)], 4)


def set_ort_path() -> None:
    if os.environ.get("ORT_DYLIB_PATH"):
        return
    libs = sorted(glob.glob(str(ROOT / ".venv/lib/python3.*/site-packages/onnxruntime/capi/libonnxruntime.so.*")))
    if libs:
        os.environ["ORT_DYLIB_PATH"] = libs[-1]


def launch(run_dir: Path, tag: str, play_timing: bool) -> tuple:
    """Start the server, return (proc, port, token, ready_s, log_path). ready_s = spawn -> 'listening on'."""
    run_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    log_path = run_dir / f"server_{tag}.log"
    log = log_path.open("wb")
    cmd = [
        "taskset", "-c", "0-3", str(ARGS.rust_bin),
        "--port", "0", "--token", token,
        "--records", str(run_dir / f"records_{tag}"),
        "--pics-dir", str(run_dir / "pics"),
        "--root", str(ROOT),
    ]
    env = os.environ.copy()
    env["DECKLIGHT_TIMING"] = "1"
    if ARGS.no_warm:
        env["DECKLIGHT_NO_WARM"] = "1"
    if play_timing:
        env["DECKLIGHT_PLAY_TIMING"] = "1"
    t0 = time.perf_counter()
    proc = subprocess_popen(cmd, log, env)
    lines: queue.Queue = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line.decode(errors="replace"))

    threading.Thread(target=reader, daemon=True).start()
    while time.perf_counter() - t0 < 120:
        try:
            line = lines.get(timeout=0.1)
        except queue.Empty:
            if proc.poll() is not None:
                tail = log_path.read_text(errors="replace")[-2000:]
                raise RuntimeError(f"server {tag} exited before listening (rc={proc.returncode})\n{tail}")
            continue
        if "listening on" in line:
            ready = time.perf_counter() - t0
            return proc, int(line.rsplit(" ", 1)[1]), token, ready, log_path
    proc.kill()
    raise RuntimeError(f"server {tag} did not print 'listening on' in 120 s")


def subprocess_popen(cmd: list[str], log, env: dict):
    import subprocess

    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=log, env=env)


def stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def timing_lines(log_path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    try:
        text = log_path.read_text(errors="replace")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        if not line.startswith("TIMING "):
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            out[parts[1]].append(float(parts[2]))
        except ValueError:
            pass
    return out


def read_tail(log_path: Path, offset: int) -> tuple[str, int]:
    try:
        with log_path.open("rb") as f:
            f.seek(offset)
            data = f.read()
        return data.decode(errors="replace"), offset + len(data)
    except FileNotFoundError:
        return "", offset


def file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0


def wait_for_warm(log_path: Path, timeout: float = 90.0) -> float | None:
    """Wait until the background ONNX warm-up prints its TIMING line (the app after its first moment)."""
    end = time.time() + timeout
    while time.time() < end:
        lines = timing_lines(log_path)
        if "warm.onnx" in lines:
            return lines["warm.onnx"][-1]
        time.sleep(0.1)
    return None


async def ws_connect(port: int, token: str):
    import websockets

    return await websockets.connect(f"ws://127.0.0.1:{port}/ws?t={token}", max_size=None, ping_interval=None)


async def next_meaningful(ws, timeout: float = 180.0, drain: float = 0.0) -> dict:
    """Next frame that is neither a thinking ticker nor a lobby/room frame."""
    while True:
        f = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if f.get("thinking") is not None:
            continue
        if f.get("t") in ("lobby", "room"):
            continue
        return f


async def next_prompt_or_result(ws, timeout: float = 180.0) -> dict:
    """Next frame that needs the human (a prompt) or ends the duel."""
    while True:
        f = await next_meaningful(ws, timeout)
        if f.get("t") == "error" or f.get("prompt") is not None or f.get("result") is not None:
            return f


def fail(msg: str) -> None:
    print(f"ALERT perf_check: {msg}", flush=True)
    raise SystemExit(1)


async def measure_lobby(port: int, token: str) -> float:
    t0 = time.perf_counter()
    ws = await ws_connect(port, token)
    try:
        while True:
            f = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if f.get("t") == "lobby":
                return time.perf_counter() - t0
    finally:
        await ws.close()


async def measure_first_prompt(ws, deck: str, bot_deck: str, first: str) -> float:
    t0 = time.perf_counter()
    await ws.send(json.dumps({
        "t": "new_game", "deck": {"ydk": deck_text(deck)}, "bot_deck": {"ydk": deck_text(bot_deck)},
        "first": first, "chain": "auto", "mode": "product",
    }))
    while True:
        f = json.loads(await asyncio.wait_for(ws.recv(), 180))
        if f.get("thinking") is not None:
            continue
        if f.get("t") == "error":
            fail(f"new game ({deck} vs {bot_deck}, first {first}) failed: {f}")
        if f.get("result") is not None:
            fail(f"new game ({deck} vs {bot_deck}, first {first}) ended before a human prompt")
        if f.get("prompt") is not None:
            return time.perf_counter() - t0


async def play_short(ws, log_path: Path, deck: str, bot_deck: str, seed: int, max_prompts: int = 8,
                     deadline: float | None = None) -> list[dict]:
    """A product game answered with a fixed seeded choice; returns answer -> next-prompt intervals."""
    rng = SplitMix(seed)
    await ws.send(json.dumps({
        "t": "new_game", "deck": {"ydk": deck_text(deck)}, "bot_deck": {"ydk": deck_text(bot_deck)},
        "first": "me", "chain": "auto", "mode": "product",
    }))
    f = await next_prompt_or_result(ws)
    if f.get("t") == "error" or f.get("result") is not None or f.get("prompt") is None:
        fail(f"short game {seed}: no opening human prompt: {f}")
    intervals: list[dict] = []
    answered = 0
    while True:
        p = f["prompt"]
        n = len(p["options"])
        if answered >= max_prompts or (deadline is not None and time.time() >= deadline):
            await ws.send(json.dumps({"t": "surrender"}))
            while True:
                g = await next_meaningful(ws)
                if g.get("t") == "error":
                    fail(f"short game {seed}: surrender failed: {g}")
                if g.get("result") is not None:
                    return intervals
        idx = rng.next() % n
        offset = file_size(log_path)
        t0 = time.perf_counter()
        await ws.send(json.dumps({"t": "choose", "prompt": p.get("id"), "index": idx}))
        g = await next_prompt_or_result(ws)
        dt = time.perf_counter() - t0
        new, _ = read_tail(log_path, offset)
        searched = [ln for ln in new.splitlines() if ln.startswith("product: search ")]
        durations = []
        for ln in searched:
            try:
                durations.append(float(ln.rsplit("t=", 1)[1]))
            except (IndexError, ValueError):
                pass
        intervals.append({"s": round(dt, 4), "searched": bool(searched), "search_t": durations})
        answered += 1
        if g.get("t") == "error":
            fail(f"short game {seed}: server error: {g}")
        if g.get("result") is not None:
            print(f"short game {seed}: ended naturally after {answered} prompts", flush=True)
            return intervals
        if g.get("prompt") is None:
            fail(f"short game {seed}: frame with no prompt: {g}")
        f = g


def summarise_bot(intervals: list[dict]) -> dict:
    searches = [t for iv in intervals for t in iv["search_t"]]
    searched = [iv for iv in intervals if iv["searched"]]
    plain = [iv for iv in intervals if not iv["searched"]]

    def dist(xs: list[float]) -> dict:
        return {"n": len(xs), "p50": percentile(xs, 0.5) if xs else None,
                "p90": percentile(xs, 0.9) if xs else None, "max": round(max(xs), 4) if xs else None}

    return {
        "frames": len(intervals),
        "searched_frames": len(searched),
        "unsearched_frames": len(plain),
        "searches": dist(searches),
        "searched_wall_s": dist([iv["s"] for iv in searched]),
        "unsearched_wall_s": dist([iv["s"] for iv in plain]),
    }


async def run_games(port: int, token: str, log_path: Path) -> tuple[dict, list[dict]]:
    first: dict[str, list[float]] = {"me": [], "bot": []}
    per_deck: dict[str, dict[str, list[float]]] = {d: {"me": [], "bot": []} for d in DECKS}
    intervals: list[dict] = []
    cold_me: float | None = None
    ws = await ws_connect(port, token)
    try:
        # Drain the opening lobby.
        f = json.loads(await asyncio.wait_for(ws.recv(), 30))
        if f.get("t") != "lobby":
            fail(f"first frame is not the lobby: {f}")
        for deck in DECKS:
            for which in ("me", "bot"):
                for _ in range(3):
                    dt = await measure_first_prompt(ws, deck, BOT_DECK, which)
                    if cold_me is None and which == "me":
                        cold_me = dt
                    first[which].append(dt)
                    per_deck[deck][which].append(dt)
                    print(f"first_prompt {deck} vs {BOT_DECK} first={which}: {dt:.3f}s", flush=True)
        deadline = time.time() + 90
        for g in range(2):
            got = await play_short(ws, log_path, "mitsurugi", BOT_DECK, seed=1000 + g, deadline=deadline)
            intervals.extend(got)
            print(f"short game {g}: {len(got)} answered prompts", flush=True)
    finally:
        await ws.close()
    out = {
        "first_prompt_me_s": median(first["me"]),
        "first_prompt_bot_s": median(first["bot"]),
        "cold_first_prompt_me_s": round(cold_me, 4) if cold_me is not None else None,
        "first_prompt_runs": first,
        "first_prompt_per_deck": per_deck,
    }
    return out, intervals


def breakdown(log_paths: list[Path], only: str | None = None) -> dict[str, float]:
    agg: dict[str, list[float]] = defaultdict(list)
    for p in log_paths:
        for label, vals in timing_lines(p).items():
            if only and not label.startswith(only):
                continue
            agg[label].extend(vals)
    return {k: median(v) for k, v in agg.items()}


def print_table(title: str, rows: dict[str, float]) -> None:
    print(f"\n{title} (median s):", flush=True)
    for label, secs in sorted(rows.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {label:28s} {secs:8.4f}", flush=True)


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="name of the JSON under data/cache/app_speed/")
    ap.add_argument("--rust-bin", type=Path, default=ROOT / "rust/target/release/decklight-serve")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/cache/app_speed/runs")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/cache/app_speed")
    ap.add_argument("--no-warm", action="store_true", help="disable the startup ONNX warm-up (the before path)")
    ARGS = ap.parse_args()
    set_ort_path()
    if not ARGS.rust_bin.is_file():
        fail(f"{ARGS.rust_bin} does not exist (build it: cargo build --release -p decklight-server)")
    if not (ROOT / "data/cache/onnx/y/manifest.json").is_file():
        fail("data/cache/onnx/y is missing (the product export)")

    # 1. ready_s: three cold starts.
    ready_runs: list[float] = []
    ready_logs: list[Path] = []
    for i in range(3):
        proc, _port, _token, ready, log = launch(ARGS.run_dir, f"ready{i}", play_timing=False)
        ready_runs.append(ready)
        ready_logs.append(log)
        stop(proc)
        print(f"ready run {i}: {ready:.3f}s", flush=True)
    ready_s = median(ready_runs)

    # 2-4. One warm server for the lobby / new-game / bot-move measurements.
    proc, port, token, ready_warm, log_path = launch(ARGS.run_dir, "games", play_timing=True)
    try:
        warm = None
        if ARGS.no_warm:
            print("warm-up disabled (--no-warm): the before path", flush=True)
        else:
            warm = wait_for_warm(log_path, 90)
            if warm is None:
                print("ALERT perf_check: the ONNX warm-up line never appeared (first games may load it)", flush=True)
        lobby_runs = [asyncio.run(measure_lobby(port, token)) for _ in range(3)]
        lobby_s = median(lobby_runs)
        print(f"lobby runs: {[round(x, 3) for x in lobby_runs]}", flush=True)
        games, intervals = asyncio.run(run_games(port, token, log_path))
    finally:
        stop(proc)

    bot = summarise_bot(intervals)
    if not intervals:
        fail("no bot-move intervals were measured")
    if bot["searches"]["n"] == 0:
        fail("no bot move searched (DECKLIGHT_PLAY_TIMING line missing?)")
    if not games["first_prompt_runs"]["me"] or not games["first_prompt_runs"]["bot"]:
        fail("a first-prompt series is missing")

    ready_bd = breakdown(ready_logs, "ready.")
    game_bd = breakdown([log_path], "newgame.")
    print_table("startup breakdown", ready_bd)
    print_table("new-game breakdown", game_bd)

    result = {
        "label": ARGS.label,
        "rust_bin": str(ARGS.rust_bin),
        "onnx": "data/cache/onnx/y",
        "ready_s": ready_s,
        "ready_runs": [round(x, 4) for x in ready_runs],
        "lobby_s": lobby_s,
        "lobby_runs": [round(x, 4) for x in lobby_runs],
        **games,
        "bot_move": bot,
        "breakdown_ready": ready_bd,
        "breakdown_newgame": game_bd,
        "warm_onnx_s": warm,
        "pairs": [[d, BOT_DECK] for d in DECKS],
    }
    ARGS.out_dir.mkdir(parents=True, exist_ok=True)
    out = ARGS.out_dir / f"{ARGS.label}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\nsummary:", flush=True)
    print(json.dumps({
        "ready_s": ready_s,
        "lobby_s": lobby_s,
        "first_prompt_me_s": result["first_prompt_me_s"],
        "cold_first_prompt_me_s": result["cold_first_prompt_me_s"],
        "first_prompt_bot_s": result["first_prompt_bot_s"],
        "bot_search_p50": bot["searches"]["p50"],
        "bot_search_p90": bot["searches"]["p90"],
        "bot_search_max": bot["searches"]["max"],
        "bot_unsearched_p50": bot["unsearched_wall_s"]["p50"],
    }, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
