"""Per-deck strategy notes (game plan, key starters, what to play through, hand-trap timing) injected into the
planner, game-turn and interruption prompts. One Markdown file per pool deck: `data/strategy/<deck>.md`, written by
`scripts/build_strategy_notes.py`. Missing notes are fine (empty string)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTES_DIR = ROOT / "data" / "strategy"
MAX_CHARS = 2800  # prompt budget per deck (2000 note + 500 win condition, 2026-10-04)


def notes_path(deck: str) -> Path:
    return NOTES_DIR / f"{deck}.md"


def load_notes(deck: str, max_chars: int = MAX_CHARS) -> str:
    """The deck's strategy notes, trimmed to `max_chars`, or "" when none exist."""
    p = notes_path(deck)
    if not p.is_file():
        return ""
    text = p.read_text().strip()
    return text if len(text) <= max_chars else text[:max_chars].rsplit("\n", 1)[0]


PRINCIPLES_PATH = NOTES_DIR / "PRINCIPLES.md"  # deck-independent play principles (2026-10-04, from strategy guides)


def principles() -> str:
    return PRINCIPLES_PATH.read_text().strip() if PRINCIPLES_PATH.is_file() else ""


def notes_block(deck: str) -> str:
    """Prompt section with the general principles and the deck's notes (empty string when there are neither)."""
    n, g = load_notes(deck), principles()
    out = f"\n{g}\n" if g else ""
    return out + (f"\nSTRATEGY NOTES for {deck} (how strong players pilot it; follow when legal):\n{n}\n" if n else "")
