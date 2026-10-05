"""Play a duel with one opencode session per turn (context does not grow across turns).

    uv run python scripts/play_game.py <game> <brief> [--model M]

Loop: run `opencode run --auto -m M --dir . "<msg>" -f <brief> -f <prologue>` under `taskset -c 0-3`, where msg =
"Play ONLY turn <t> of duel <game> (both seats) following the attached brief. The current log and
state are attached; do not re-run log/show unless something changed. Deck sheets and strategy
summaries are in the attached prologue; re-fetch only full card text (`txt.py`) when you need it.
Stop as soon as the turn number
changes or the duel is over; do not write the report." <prologue> is a temp file built by
`build_prologue` (log tail + show + both decks' compressed `--brief` sheets and strategy TL;DRs,
cached per game).
After the duel is over, one final session: "The duel <game> is over: run
`save` once and write the report per the brief." Stops on duel over, on no progress twice in a row
(no new choices), or after 60 sessions. Transient API errors (overloaded / rate limit) are retried
like `delegate.sh`. Per-session rows go to `data/cache/briefs/runs.tsv` (tag `<game>.t<turn>`).
Never starts if `<game>` has a live pid file (`data/cache/briefs/play-<game>.pid`).

`--cmd-template` (or the YGO_PLAY_CMD env var) overrides the opencode command for tests: a shell
template with {msg} {brief} {model} {game} {turn} {state} {prologue} {prologue_args} {tag}
placeholders ({state} = the game's state.json path, {prologue} = the prologue file path,
{prologue_args} = " -f <prologue>" or "").
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import deck_sheet
import hand_play

from ygo_zero.demos.names import CardNames
from ygo_zero.engine.duel import Engine

BRIEFS = ROOT / "data" / "cache" / "briefs"
RUNS = BRIEFS / "runs.tsv"
DEFAULT_MODEL = "opencode-go/muse-spark-1.3-contributor"
MAX_SESSIONS = 60
PROLOGUE_TURNS = 3  # `log` tail per session: the last 3 turns (full log stays available on demand)
MAX_PROLOGUE_CHARS = 16_000  # truncate the log head first, then the strategy TL;DRs; never show/sheets
SHEET_TEXT_CHARS = 100  # per-card text cap in the cached prologue sheets (compressed `--brief`)
STRATEGY_CHARS = 600  # per-deck strategy TL;DR length in the prologue
_TLDR_MARK = "\n## Strategy summaries\n"
_TRANSIENT = re.compile(r"overloaded|rate limit|ECONNRESET|503|502", re.IGNORECASE)


def _now() -> str:
    return time.strftime("%FT%T", time.localtime())


def duel_status(game: str) -> tuple[bool, int, int]:
    """(over, current turn, number of choices) of the saved duel, via replay."""
    st = hand_play._load(game)
    engine = Engine()
    _, obs = hand_play._replay(engine, st)
    n = len(st["choices"])
    if obs is None:
        turn = st["turns"][-1] if st["turns"] else 0
        return True, turn, n
    return False, int(obs["global"][2]), n


def _result_status(log_path: Path) -> str:
    try:
        text = log_path.read_text()
    except OSError:
        return "none"
    m = re.findall(r"^status:\s*([a-z]+)", text, re.MULTILINE)
    return m[-1] if m else "none"


def _compress_sheet(text: str) -> str:
    """A `--brief` sheet with each card's text capped at SHEET_TEXT_CHARS (prologue cache)."""
    out = []
    for line in text.splitlines():
        head, sep, tail = line.partition(" — ")
        out.append(head + sep + tail[:SHEET_TEXT_CHARS] if sep else line)
    return "\n".join(out)


def game_sheets(game: str) -> str:
    """Both decks' compressed `--brief` sheets + both strategy TL;DRs, cached per game in the
    briefs dir (computed once). Sections: `## Deck sheets`, then `## Strategy summaries`."""
    path = BRIEFS / f"play-{game}.sheets.md"
    if path.exists():
        return path.read_text()
    st = hand_play._load(game)
    parts = []
    for deck in st["decks"]:
        parts.append(f"### {deck}\n" + _compress_sheet(deck_sheet.sheet(deck, brief=True)))
    tldrs = []
    for deck in st["decks"]:
        sp = ROOT / "data" / "strategy" / f"{deck}.md"
        tldr = sp.read_text()[:STRATEGY_CHARS] if sp.exists() else "(no strategy note found)"
        tldrs.append(f"### {deck}\n{tldr}")
    text = "## Deck sheets\n" + "\n\n".join(parts) + _TLDR_MARK + "\n".join(tldrs)
    path.write_text(text)
    return text


def build_prologue(game: str, turn: int) -> str:
    """The turn's opening context, so the session needs no tool call before its first decision: the
    `log` tail (last PROLOGUE_TURNS turns) + the current `show`, then both decks' compressed
    `--brief` sheets plus both strategy TL;DRs (cached per game by `game_sheets`: every session
    carries them). Kept under MAX_PROLOGUE_CHARS by truncating the log head first, then the
    strategy TL;DRs — never the state, never the deck sheets."""
    st = hand_play._load(game)
    engine = Engine()
    names = CardNames()
    env, obs = hand_play._replay(engine, st)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hand_play._log(st, names, engine, PROLOGUE_TURNS)
    log_text = buf.getvalue().strip()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        hand_play._show(env, obs, st, names, False)
    show_text = buf.getvalue().strip()
    cached = game_sheets(game)
    sheets, _, tldrs = cached.partition(_TLDR_MARK)
    head = (
        f"Turn {turn} of duel {game} ({st['decks'][0]} vs {st['decks'][1]}). "
        "The current log and state are attached; do not re-run log/show unless something "
        "changed."
    )
    log_block = "\n\n## Recent log (last 3 turns)\n"
    tail = "\n\n## Current state\n" + show_text + "\n\n" + sheets
    tldr_block = _TLDR_MARK + tldrs
    marker = "[... earlier log truncated; run `hand_play.py log` for the full log ...]\n"
    lines = log_text.splitlines()
    while lines and len(head) + len(log_block) + len(marker) + len("\n".join(lines)) + len(tail) + len(tldr_block) + 60 > MAX_PROLOGUE_CHARS:
        lines.pop(0)
    log_text = (marker + "\n".join(lines)) if lines else marker.strip()
    while tldrs and len(head) + len(log_block) + len(log_text) + len(tail) + len(tldr_block) + 60 > MAX_PROLOGUE_CHARS:
        tldrs = tldrs[: len(tldrs) // 2]
        tldr_block = _TLDR_MARK + tldrs + "\n[... strategy summaries truncated ...]"
    return head + log_block + log_text + tail + tldr_block


def run_session(
    *,
    game: str,
    turn: int | None,
    brief: str,
    model: str,
    tag: str,
    final: bool = False,
    cmd_template: str | None = None,
    tries: int = 4,
    prologue_path: str | None = None,
) -> tuple[int, Path]:
    """Run one opencode session; return (exit code, log path). Retries transient API errors."""
    if final:
        msg = f"The duel {game} is over: run `save` once and write the report per the brief."
    else:
        assert turn is not None
        msg = (
            f"Play ONLY turn {turn} of duel {game} (both seats) following the attached brief. "
            "The current log and state are attached; do not re-run log/show unless something "
            "changed. Deck sheets and strategy summaries are in the attached prologue; "
            "re-fetch only full card text (`txt.py`) when you need it. "
            "Stop as soon as the turn number changes or the duel is over; "
            "do not write the report."
        )
    template = cmd_template or (
        "taskset -c 0-3 opencode run --auto -m {model} --dir . {msg} -f {brief}{prologue_args}"
    )
    prologue_args = f" -f {shlex.quote(prologue_path)}" if prologue_path else ""
    cmd = template.format(
        msg=shlex.quote(msg), brief=shlex.quote(brief), model=shlex.quote(model),
        game=shlex.quote(game), turn=shlex.quote(str(turn)),
        state=shlex.quote(str(hand_play.WORK / game / "state.json")),
        prologue=shlex.quote(prologue_path) if prologue_path else "",
        prologue_args=prologue_args,
        tag=shlex.quote(tag),
    )
    log_path = BRIEFS / f"play-{tag}.log"
    rc = 1
    t0 = time.time()
    start = _now()
    for attempt in range(1, tries + 1):
        with log_path.open("a") as f:
            f.write(f"$ {cmd}\n")
            proc = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f,
                                    stderr=subprocess.STDOUT, check=False)
        rc = proc.returncode
        if rc == 0:
            break
        try:
            tail = log_path.read_text()[-400:]
        except OSError:
            tail = ""
        if not _TRANSIENT.search(tail) or attempt == tries:
            break
        wait = 60 * attempt
        with log_path.open("a") as f:
            f.write(f"[play_game] transient error, retry {attempt + 1}/{tries} after {wait}s\n")
        time.sleep(wait)
    if not RUNS.exists():
        RUNS.write_text("tag\tmodel\tbrief\tstart\tend\tseconds\texit\tstatus\n")
    secs = int(time.time() - t0)
    with RUNS.open("a") as f:
        f.write(f"{tag}\t{model}\t{brief}\t{start}\t{_now()}\t{secs}\t{rc}\t{_result_status(log_path)}\n")
    return rc, log_path


def _live_pid(path: Path) -> bool:
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def play(game: str, brief: str, model: str, max_sessions: int, cmd_template: str | None) -> str:
    """The per-turn loop. Returns the stop reason: over | no-progress | max-sessions."""
    pid_path = BRIEFS / f"play-{game}.pid"
    if _live_pid(pid_path):
        print(f"refusing: {game} is already playing (pid {pid_path.read_text().strip()})")
        return "already-running"
    BRIEFS.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    try:
        stagnant = 0
        sessions = 0
        while True:
            over, turn, n0 = duel_status(game)
            if over:
                run_session(game=game, turn=None, brief=brief, model=model,
                            tag=f"{game}.final", final=True, cmd_template=cmd_template)
                return "over"
            if sessions >= max_sessions:
                return "max-sessions"
            tag = f"{game}.t{turn}"
            prologue_file = BRIEFS / f"play-{tag}.prologue.md"
            prologue_file.write_text(build_prologue(game, turn))
            run_session(game=game, turn=turn, brief=brief, model=model,
                        tag=tag, cmd_template=cmd_template,
                        prologue_path=str(prologue_file))
            sessions += 1
            _, _, n1 = duel_status(game)
            stagnant = 0 if n1 != n0 else stagnant + 1
            if stagnant >= 2:
                return "no-progress"
    finally:
        try:
            pid_path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Play a duel with one opencode session per turn.")
    ap.add_argument("game")
    ap.add_argument("brief")
    ap.add_argument("--model", default=os.environ.get("YGO_AGENT_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max-sessions", type=int, default=MAX_SESSIONS)
    ap.add_argument("--cmd-template", default=os.environ.get("YGO_PLAY_CMD"),
                    help="shell template overriding the opencode command (tests)")
    a = ap.parse_args(argv)
    print(play(a.game, a.brief, a.model, a.max_sessions, a.cmd_template))
    return 0


if __name__ == "__main__":
    sys.exit(main())
