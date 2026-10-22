"""Grow the deck pool (ADR 0004 §4): muse proposes a decklist per archetype, we ground and validate it.

    uv run python scripts/build_decks.py --candidates decks/pool/CANDIDATES.tsv [--only snake-eye yubel] [--workers 6]

CANDIDATES.tsv: `slug<TAB>archetype / build<TAB>era<TAB>split` (split: train | holdout). For each slug without a
`decks/pool/<slug>.ydk`:
1. muse writes the list as JSON (names + counts); the prompt includes the card-DB names that share the archetype's
   words, so names come out exact.
2. Names are resolved against the card DB (exact/normalised only). Unresolved names go back to muse once with the
   candidates.
3. Validation: every card has an official script, ≤3 copies (alt arts grouped), main 40–60, extra ≤15, extra-deck
   monsters re-split into the extra deck. Not checked: banlists (rogue and older-era lists are welcome).
4. Engine smoke: 2 random-play duels of the deck vs itself (300 decisions each) must raise no error.
Passing decks go to `decks/pool/<slug>.ydk` and `decks/pool/GENERATED.md`. Holdout slugs go to `decks/pool/HOLDOUT`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decklight.decks import Deck
from decklight.demos import llm
from decklight.demos.names import CardNames
from decklight.engine import constants as C
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "decks" / "pool"
SCRIPTS = ROOT / "third_party" / "CardScripts" / "official"
EXTRA_T = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK
MODEL = "opencode/muse-spark-1.3-contributor-free"

SYSTEM = "You are a Yu-Gi-Oh! TCG deckbuilding expert. You answer with JSON only."


def parse_obj(text: str) -> dict | None:
    """The first JSON object in a reply (code fences and prose tolerated)."""
    t = re.sub(r"^```\w*\n?|\n?```\s*$", "", text.strip())
    for cand in (t, (re.search(r"\{.*\}", t, re.DOTALL) or [None])[0]):
        if cand:
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def related_names(names: CardNames, archetype: str, limit: int = 120) -> list[str]:
    """Card-DB names sharing a distinctive word with the archetype description (grounding for the prompt)."""
    words = {w for w in re.findall(r"[a-z0-9]+", archetype.lower()) if len(w) >= 4} - {
        "deck", "build", "turbo", "control", "combo", "engine", "variant", "pure", "with", "from", "rogue",
    }  # fmt: skip
    out = sorted(
        {
            n
            for c, n in names.names.items()
            if names.alias.get(c, 0) == 0 and words & set(re.findall(r"[a-z0-9]+", n.lower()))
        }
    )
    return out[:limit]


def deck_prompt(slug: str, archetype: str, era: str, related: list[str]) -> str:
    rel = "\n".join(f"- {n}" for n in related) or "(none found)"
    return f"""Build a strong, tournament-style Yu-Gi-Oh! decklist.

Deck: {archetype}
Era / format: {era}

Card names that exist in our card database and share words with this archetype (use these exact spellings):
{rel}

Rules:
- Main deck 40-45 cards, Extra deck exactly 15 (or fewer only if the strategy truly needs fewer).
- At most 3 copies of any card. Use real, exact English TCG card names (official spelling).
- Include the archetype core plus the engine/staples (hand traps, generic extenders, boss monsters) a strong player
  of that era would run. The deck must be able to combo/play on its own.
- Only real cards that existed in that era.

Answer JSON only:
{{"slug": "{slug}", "main": [["Card Name", 3], ...], "extra": [["Card Name", 1], ...], "notes": "one line: the deck's game plan"}}"""


def repair_prompt(base: str, prev: str, problems: list[str], suggestions: dict[str, list[str]]) -> str:
    sug = "\n".join(f"- {k}: did you mean one of {v}?" for k, v in suggestions.items() if v)
    return f"""{base}

Your previous answer:
{prev[:6000]}

Problems:
{chr(10).join("- " + p for p in problems[:40])}
{("Spelling suggestions from our database:" + chr(10) + sug) if sug else ""}

Fix every problem (replace cards we cannot find with real cards that fill the same role) and answer the full JSON again."""


def suggest(names: CardNames, bad: str, k: int = 4) -> list[str]:
    toks = set(re.findall(r"[a-z0-9]+", bad.lower()))
    scored = []
    for c, n in names.names.items():
        if names.alias.get(c, 0):
            continue
        nt = set(re.findall(r"[a-z0-9]+", n.lower()))
        s = len(toks & nt)
        if s:
            scored.append((-s / max(len(toks | nt), 1), n))
    return [n for _, n in sorted(scored)[:k]]


def validate_codes(
    main: list[int], extra: list[int], engine: Engine, names: CardNames, min_extra: int = 5
) -> list[str]:
    """Validate resolved passcode lists: engine DB + official scripts, alias-grouped ≤3, sizes.

    Shared by build_decks (name resolution drops unknown cards first, so `main`/`extra`
    hold only resolved codes) and fetch_meta_decks (passcodes straight from YGOPRODeck).
    `min_extra` is 5 for muse-built lists; meta fetches pass 0 (real lists set the size).
    """
    problems: list[str] = []
    per_card: Counter = Counter()
    for code in (*main, *extra):
        canon = names.alias.get(code, 0) or code
        per_card[canon] += 1
        if engine.cards.get(code) is None:
            problems.append(f"{code}: missing from the engine card DB")
        elif not (SCRIPTS / f"c{code}.lua").exists() and not (SCRIPTS / f"c{canon}.lua").exists():
            problems.append(f"{names.names.get(code, code)}: no engine script for it (replace it)")
    for canon, n in per_card.items():
        if n > 3:
            problems.append(f"{names.names.get(canon, canon)}: {n} copies (max 3)")
    if not 40 <= len(main) <= 60:
        problems.append(f"main deck has {len(main)} cards after removing unknown cards (need 40-60)")
    if len(extra) > 15:
        problems.append(f"extra deck has {len(extra)} cards (max 15)")
    if len(extra) < min_extra:
        problems.append(f"extra deck has only {len(extra)} cards")
    return problems


def ground(engine: Engine, names: CardNames, d: dict) -> tuple[Deck | None, list[str], dict[str, list[str]]]:
    """JSON list -> Deck, plus problems and spelling suggestions."""
    problems: list[str] = []
    sugg: dict[str, list[str]] = {}
    main: list[int] = []
    extra: list[int] = []
    per_card: Counter = Counter()
    for section in ("main", "extra"):
        for item in d.get(section, []) or []:
            try:
                name, n = (
                    (item[0], int(item[1]))
                    if isinstance(item, list)
                    else (item["name"], int(item.get("count", 1)))
                )
            except Exception:  # noqa: BLE001
                problems.append(f"malformed entry {item!r}")
                continue
            codes = names.resolve(name)
            if not codes:
                problems.append(f"unknown card name: {name}")
                sugg[name] = suggest(names, name)
                continue
            code = codes[0]
            canon = names.alias.get(code, 0) or code
            if not (SCRIPTS / f"c{code}.lua").exists() and not (SCRIPTS / f"c{canon}.lua").exists():
                problems.append(f"no engine script for {name} (not supported by our engine; replace it)")
                continue
            cd = engine.cards.get(code)
            if cd is None:
                problems.append(f"{name}: missing from the engine card DB")
                continue
            per_card[canon] += n
            (extra if cd.type & EXTRA_T else main).extend([code] * n)
    problems += validate_codes(main, extra, engine, names)
    counts_ok = 40 <= len(main) <= 60 and 5 <= len(extra) <= 15 and max(per_card.values(), default=0) <= 3
    deck = Deck(tuple(main), tuple(extra), (), "") if counts_ok else None
    return deck, problems, sugg


def smoke(engine: Engine, deck: Deck, games: int = 2, steps: int = 300) -> str | None:
    """Random-play the deck against itself; return an error string or None."""
    for g in range(games):
        rng = random.Random(g)
        try:
            env = CppEnv(engine, DuelConfig(deck, deck, seed=1000 + g))
            obs = env.reset()
            for _ in range(steps):
                if obs is None:
                    break
                obs = env.step(rng.randrange(max(int(obs["action_mask"].sum()), 1)))
        except Exception as e:  # noqa: BLE001
            return f"smoke game {g}: {e}"[:300]
    return None


def write_ydk(slug: str, deck: Deck, archetype: str, era: str) -> Path:
    p = POOL / f"{slug}.ydk"
    body = [
        f"#created by ygo-zero build_decks (muse, {time.strftime('%Y-%m-%d')}): {archetype} [{era}]",
        "#main",
    ]
    body += [str(c) for c in deck.main] + ["#extra"] + [str(c) for c in deck.extra] + ["!side", ""]
    p.write_text("\n".join(body))
    return p


def build_one(engine, names, slug, archetype, era, variant, repairs=2) -> tuple[str, str, Deck | None, dict]:
    base = deck_prompt(slug, archetype, era, related_names(names, archetype))
    prompt, resp = base, ""
    info: dict = {"attempts": 0}
    for _attempt in range(repairs + 1):
        info["attempts"] += 1
        try:
            resp = llm.complete(prompt, model=MODEL, system=SYSTEM, timeout=900, variant=variant)
        except Exception as e:  # noqa: BLE001
            info["error"] = str(e)[:200]
            return slug, "llm_error", None, info
        d = parse_obj(resp)
        if d is None:
            prompt = base + "\n\nYour previous answer was not valid JSON. Answer with the JSON object only."
            continue
        deck, problems, sugg = ground(engine, names, d)
        info["notes"] = d.get("notes", "")
        info["problems"] = problems[:20]
        # clean, or (last attempt) only unknown names were dropped and the counts still hold
        if deck is not None and (
            not problems
            or (
                _attempt == repairs
                and all(p.startswith(("unknown card name", "no engine script")) for p in problems)
            )
        ):
            return slug, "ok", deck, info
        prompt = repair_prompt(base, json.dumps(d), problems, sugg)
    return slug, "invalid", None, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=POOL / "CANDIDATES.tsv")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--variant", default=None, help="muse reasoning effort (default: provider's; lists need recall)"
    )
    args = ap.parse_args()
    engine, names = Engine(), CardNames()
    rows = []
    for line in args.candidates.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        slug, archetype, era, split = (line.split("\t") + ["", "", "train"])[:4]
        if args.only and slug not in args.only:
            continue
        if (POOL / f"{slug}.ydk").exists():
            continue
        rows.append((slug.strip(), archetype.strip(), era.strip(), split.strip() or "train"))
    print(f"{len(rows)} decks to build", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(build_one, engine, names, s, a, e, args.variant) for s, a, e, _ in rows]
        results = [f.result() for f in futs]
    gen = POOL / "GENERATED.md"
    if not gen.exists():
        gen.write_text(
            "# Generated pool decks (scripts/build_decks.py)\n\n| deck | archetype | era | split | main | extra | plan |\n|---|---|---|---|---|---|---|\n"
        )
    holdout = POOL / "HOLDOUT"
    for (slug, status, deck, info), (_, archetype, era, split) in zip(results, rows, strict=True):
        if status == "ok":
            err = smoke(engine, deck)
            if err:
                status = "smoke_fail"
                info["smoke"] = err
        if status != "ok":
            print(f"FAIL {slug}: {status} {json.dumps(info)[:600]}", flush=True)
            continue
        write_ydk(slug, deck, archetype, era)
        with gen.open("a") as f:
            notes = str(info.get("notes", "")).replace("|", "/")[:160]
            f.write(
                f"| {slug} | {archetype} | {era} | {split} | {len(deck.main)} | {len(deck.extra)} | {notes} |\n"
            )
        if split == "holdout":
            with holdout.open("a") as f:
                f.write(slug + "\n")
        print(
            f"OK {slug}: main {len(deck.main)} extra {len(deck.extra)} ({info['attempts']} calls) {split}",
            flush=True,
        )


if __name__ == "__main__":
    main()
