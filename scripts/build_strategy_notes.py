"""Write per-deck strategy notes for the planner (`data/strategy/<deck>.md`).

Usage:
    uv run python scripts/build_strategy_notes.py [--decks snake-eye blue-eyes] [--workers 6] [--force]

For every pool deck (`decks/pool/*.ydk`) without notes (or all with `--force`),
ask muse at DEFAULT reasoning effort (no `variant`) with the card sheet, the
deck's archetype/era/plan row from `decks/pool/GENERATED.md` (or the YGOPRODeck
page description for meta decks, matched by the `pretty_url` in the ydk header
comment), and up to 3 of the deck's strongest engine-verified lines
(`plan_lines.load_examples`). Writes ≤ 2000 chars of Markdown with the six
required headings.

    uv run python scripts/build_strategy_notes.py --win-conditions [--workers 3]

prepends a `## Win condition` section (≤ 450 chars: how the deck closes games, what it needs, what stops it) to every
note that lacks one (2026-10-04: only 2 of 175 notes said how the deck wins).
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from decklight.decks import load_ydk
from decklight.demos import llm as _llm
from decklight.demos.names import CardNames
from decklight.demos.plan import card_sheet, compact_line_json
from decklight.demos.strategy import NOTES_DIR, notes_path
from decklight.engine.duel import Engine

MODEL = "opencode/muse-spark-1.3-contributor-free"

SYSTEM = (
    "You are a Yu-Gi-Oh! TCG expert. Write a short, concrete strategy note for "
    "the given deck from its decklist and verified combo lines. "
    "Reply with Markdown only, no prose outside the six sections."
)

HEADINGS = [
    "## Game plan",
    "## Starters & extenders",
    "## Priorities going first",
    "## Going second / breaking boards",
    "## Hand traps & interruption timing",
    "## Common mistakes",
]


def _load_plan_lines():
    spec = importlib.util.spec_from_file_location("plan_lines", ROOT / "scripts" / "plan_lines.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plan_lines"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_generated() -> dict[str, dict[str, str]]:
    """GENERATED.md table: deck -> {archetype, era, plan}."""
    out: dict[str, dict[str, str]] = {}
    path = ROOT / "decks" / "pool" / "GENERATED.md"
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("deck", "---", ""):
            continue
        if set(cells[0]) <= set("- "):
            continue
        deck, archetype, era = cells[0], cells[1], cells[2]
        plan = cells[6] if len(cells) > 6 else ""
        if deck and deck != "deck":
            out[deck] = {"archetype": archetype, "era": era, "plan": plan}
    return out


def load_meta_descriptions() -> dict[str, str]:
    """pretty_url -> raw HTML deck_description from cached YGOPRODeck pages."""
    import json

    out: dict[str, str] = {}
    for page in sorted((ROOT / "data" / "cache" / "meta_decks").glob("page-*.json")):
        try:
            items = json.loads(page.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(items, list):
            continue
        for d in items:
            if isinstance(d, dict) and d.get("pretty_url") and d.get("deck_description"):
                out.setdefault(d["pretty_url"], d["deck_description"])
    return out


def strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def ydk_pretty_url(deck_name: str) -> str | None:
    """The ygoprodeck pretty_url from the ydk header `#source .../deck/<pretty>`."""
    path = ROOT / "decks" / "pool" / f"{deck_name}.ydk"
    try:
        head = path.read_text().splitlines()[:8]
    except OSError:
        return None
    for line in head:
        m = re.search(r"ygoprodeck\.com/deck/(\S+)", line)
        if m:
            return m.group(1).strip().rstrip("/")
    return None


def build_prompt(
    deck_name: str,
    sheet: str,
    context: str,
    examples_block: str,
    max_chars: int = 2000,
) -> str:
    headings = "\n".join(f"- `{h}`" for h in HEADINGS)
    return f"""Deck: {deck_name}

CONTEXT (archetype / tournament info):
{context}

DECKLIST (card sheet — every card you may name):
{sheet}

STRONGEST ENGINE-VERIFIED LINES (compact JSON, names only):
{examples_block}

Write at most {max_chars} characters of Markdown with EXACTLY these six headings, in this order
(all six must appear, even if a section is one sentence):
{headings}

Rules:
- Name ONLY cards that appear in the decklist above (plus generic opponent cards like "Ash Blossom" when discussing hand traps).
- Be concrete: which card to search first, which starter to normal summon, what to hold for follow-up, what to play through.
- going first = the deck's best turn-1 end board; going second = how it breaks boards and OTKs.
- Keep each section to 1-2 sentences. Total at most {max_chars} characters.
Markdown only, no prose outside the six sections."""


WIN_HEADING = "## Win condition"


def win_prompt(deck_name: str, sheet: str, context: str, note: str) -> str:
    return f"""Deck: {deck_name}

CONTEXT (archetype / tournament info):
{context}

DECKLIST (card sheet — every card you may name):
{sheet}

CURRENT STRATEGY NOTE:
{note}

Write ONE Markdown section, at most 450 characters, starting with the line `{WIN_HEADING}`:
- how this deck actually wins games (OTK with which monsters, burn, lock / floodgates, grind with recursion,
  alternate win condition, deck-out), naming the key cards from the decklist;
- what it needs on board / in GY / in hand to close the game, and on which turn it usually wins (going first vs second);
- what stops it (the choke points an opponent should hit).
Markdown only, nothing outside the section."""


def add_win_conditions(args, pool: list[str]) -> None:
    targets = [d for d in pool if notes_path(d).is_file() and WIN_HEADING not in notes_path(d).read_text()]
    if not targets:
        print("win: every note already has a win condition")
        return
    gen, meta = parse_generated(), load_meta_descriptions()
    engine, names = Engine(), CardNames()
    prompts = []
    for d in targets:
        sheet = card_sheet(engine, load_ydk(ROOT / "decks" / "pool" / f"{d}.ydk"), names)
        row = gen.get(d, {})
        bits = [f"{k.capitalize()}: {row[k]}" for k in ("archetype", "era", "plan") if row.get(k)]
        pretty = ydk_pretty_url(d)
        if pretty and pretty in meta:
            bits.append(f"YGOPRODeck page ({pretty}): {strip_html(meta[pretty])[:1500]}")
        prompts.append(win_prompt(d, sheet, "\n".join(bits) or "(no published context)", notes_path(d).read_text()))
    responses = _llm.complete_many(prompts, model=args.model, system=SYSTEM_WIN, max_workers=args.workers,
                                   return_exceptions=True, timeout=args.llm_timeout, variant=None)  # fmt: skip
    wrote, failed = 0, 0
    for d, resp in zip(targets, responses, strict=True):
        text = "" if isinstance(resp, BaseException) else str(resp).strip()
        if WIN_HEADING not in text:
            print(f"win: {d}: bad answer ({str(resp)[:120]!r}), skipping")
            failed += 1
            continue
        text = text[text.index(WIN_HEADING):][:500].rstrip()
        notes_path(d).write_text(text + "\n" + notes_path(d).read_text())  # first, so prompt trimming keeps it
        wrote += 1
    print(f"win: wrote={wrote} failed={failed} decks={len(targets)}")


SYSTEM_WIN = "You are a Yu-Gi-Oh! TCG expert. Explain concisely how the given deck wins games. Markdown only."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", nargs="*", default=None, help="pool decks (default: all without notes)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="rebuild notes even when present")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-chars", type=int, default=2000, help="note budget (writes truncate here)")
    ap.add_argument("--llm-timeout", type=int, default=900)
    ap.add_argument("--win-conditions", action="store_true", help="prepend a win-condition section to every note")
    args = ap.parse_args()

    pool = sorted(p.stem for p in (ROOT / "decks" / "pool").glob("*.ydk"))
    wanted = set(args.decks) if args.decks else None
    if wanted:
        unknown = sorted(w for w in wanted if w not in set(pool))
        if unknown:
            ap.error(f"unknown decks: {', '.join(unknown)}")
        pool = [d for d in pool if d in wanted]
    if args.win_conditions:
        add_win_conditions(args, pool)
        return
    targets = [d for d in pool if args.force or not notes_path(d).is_file()]
    skipped = len(pool) - len(targets)
    if not targets:
        print(f"notes: wrote=0 skipped={skipped} (everything up to date)")
        return

    gen = parse_generated()
    meta = load_meta_descriptions()
    plan_lines = _load_plan_lines()

    engine = Engine()
    names = CardNames()

    prompts: list[str] = []
    decks: list[str] = []
    for deck_name in targets:
        try:
            deck = load_ydk(ROOT / "decks" / "pool" / f"{deck_name}.ydk")
        except OSError as e:
            print(f"notes: {deck_name}: cannot load ydk ({e}), skipping")
            continue
        sheet = card_sheet(engine, deck, names)
        row = gen.get(deck_name, {})
        bits = []
        if row.get("archetype"):
            bits.append(f"Archetype: {row['archetype']}")
        if row.get("era"):
            bits.append(f"Era: {row['era']}")
        if row.get("plan"):
            bits.append(f"Published plan: {row['plan']}")
        pretty = ydk_pretty_url(deck_name)
        if pretty and pretty in meta:
            bits.append(f"YGOPRODeck page ({pretty}): {strip_html(meta[pretty])[:1500]}")
        context = "\n".join(bits) if bits else "(no published context)"
        try:
            examples = plan_lines.load_examples(deck_name, 3)
        except Exception:  # noqa: BLE001 - no verified lines available -> prompt without examples
            examples = []
        examples_block = (
            "\n\n".join(compact_line_json(e) for e in examples[:3])
            if examples
            else "(no engine-verified lines yet)"
        )
        prompts.append(build_prompt(deck_name, sheet, context, examples_block, args.max_chars))
        decks.append(deck_name)

    responses = _llm.complete_many(
        prompts,
        model=args.model,
        system=SYSTEM,
        max_workers=args.workers,
        return_exceptions=True,
        timeout=args.llm_timeout,
        variant=None,  # DEFAULT reasoning effort
    )

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    wrote, failed, incomplete = 0, 0, []
    for deck_name, resp in zip(decks, responses):
        if isinstance(resp, BaseException):
            print(f"notes: {deck_name}: LLM error ({resp}), skipping")
            failed += 1
            continue
        text = (resp if isinstance(resp, str) else str(resp)).strip()
        if len(text) > args.max_chars:
            text = text[: args.max_chars].rsplit("\n", 1)[0].rstrip()
        notes_path(deck_name).write_text(text + "\n")
        wrote += 1
        if any(h not in text for h in HEADINGS):
            incomplete.append(deck_name)
    print(f"notes: wrote={wrote} skipped={skipped} failed={failed} decks={len(decks)}")
    if incomplete:
        print(f"notes: incomplete (missing headings): {' '.join(sorted(incomplete))}")


if __name__ == "__main__":
    main()
