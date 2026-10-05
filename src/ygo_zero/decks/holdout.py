"""Card-level holdout: which specific cards appear in no training data at all.

Deck-level holdout (`decks/pool/HOLDOUT`) keeps whole decks out of the
gauntlet, but their cards can still leak into training through other decks
(gauntlet lists, teacher/DAgger games, scenarios). This module audits that
leakage so a few archetypes can be held out at the *card* level for the
report card's truly-unseen-cards measurement (ADR 0007).

Training sources (mirrors `scripts/build_bc_shards.py::SOURCES` intent):
  - every record under the given `data/demos/<dir>` roots (both decks of
    each record `config`),
  - `decks/gauntlet/*.ydk`,
  - non-HOLDOUT `decks/pool/*.ydk`,
  - `data/scenarios/*.json` (cards by name, resolved to codes; unknown
    names are reported, never guessed).
"""

from __future__ import annotations

import glob
import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from ygo_zero.decks.ydk import Deck, load_ydk

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = REPO_ROOT / "decks" / "pool"
GAUNTLET_DIR = REPO_ROOT / "decks" / "gauntlet"
SCENARIO_DIR = REPO_ROOT / "data" / "scenarios"
DEMOS_DIR = REPO_ROOT / "data" / "demos"
HOLDOUT_FILE = POOL_DIR / "HOLDOUT"
CARD_HOLDOUT_FILE = POOL_DIR / "CARD_HOLDOUT"

CONFIG_KEYS = ("deck0_main", "deck0_extra", "deck1_main", "deck1_extra")


def read_stems(path: str | Path) -> list[str]:
    """One deck stem per line; `#` comments and blanks skipped."""
    path = Path(path)
    if not path.exists():
        return []
    stems = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stems.append(line)
    return stems


def load_pool_decks(
    pool_dir: str | Path = POOL_DIR, skip: set[str] | None = None
) -> dict[str, Deck]:
    """All `*.ydk` decks under `pool_dir` except `skip` stems."""
    skip = skip or set()
    decks: dict[str, Deck] = {}
    for path in sorted(Path(pool_dir).glob("*.ydk")):
        if path.stem in skip:
            continue
        try:
            decks[path.stem] = load_ydk(path)
        except Exception as e:  # noqa: BLE001 - one bad deck must not kill the audit
            logger.warning("holdout: skipping unreadable deck %s: %s", path, e)
    return decks


def deck_codes(deck: Deck) -> set[int]:
    """Union of main/extra/side codes (presence, not multiplicity)."""
    return set(deck.main) | set(deck.extra) | set(deck.side)


def staples(
    pool_decks: Mapping[str, Deck] | Iterable[Deck], min_decks: int = 8
) -> set[int]:
    """Cards present in >= `min_decks` different pool decks (generic staples)."""
    decks = pool_decks.values() if isinstance(pool_decks, Mapping) else pool_decks
    counts: Counter[int] = Counter()
    for deck in decks:
        for code in deck_codes(deck):
            counts[code] += 1
    return {code for code, n in counts.items() if n >= min_decks}


def record_config_codes(config: Mapping) -> set[int]:
    """Union of the four deck lists of a record `config` (ints only)."""
    codes: set[int] = set()
    for key in CONFIG_KEYS:
        for code in config.get(key, []) or []:
            if isinstance(code, int) and code > 0:
                codes.add(code)
    return codes


def _expand_record_files(roots: Iterable[str | Path]) -> list[Path]:
    """Glob/dir/file entries -> sorted non-empty `*.jsonl` record files."""
    files: set[Path] = set()
    for root in roots:
        text = str(root)
        if any(ch in text for ch in "*?[]"):
            files.update(Path(p) for p in sorted(glob.glob(text, recursive=True)))
            continue
        path = Path(root)
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(path.rglob("*.jsonl"))
    return sorted(p for p in files if p.is_file() and p.stat().st_size)


def record_card_counts(roots: Iterable[str | Path]) -> tuple[Counter[int], int]:
    """Per-card record frequency: (counts, n_records scanned).

    `counts[code]` = number of *records* whose config decks contain it.
    Streams files line by line; lines without a dict `config` are skipped.
    """
    counts: Counter[int] = Counter()
    n_records = 0
    for path in _expand_record_files(roots):
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("holdout: cannot open %s: %s", path, e)
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                config = obj.get("config") if isinstance(obj, dict) else None
                if not isinstance(config, dict):
                    continue
                n_records += 1
                for code in record_config_codes(config):
                    counts[code] += 1
    return counts, n_records


def records_using(
    roots: Iterable[str | Path], codes: set[int], max_refs: int = 8
) -> dict[int, dict]:
    """For each target code: {n: #records using it, refs: [file:line(id)...]}."""
    hits: dict[int, dict] = {c: {"n": 0, "refs": []} for c in codes}
    for path in _expand_record_files(roots):
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                config = obj.get("config") if isinstance(obj, dict) else None
                if not isinstance(config, dict):
                    continue
                used = record_config_codes(config) & codes
                if not used:
                    continue
                rid = obj.get("line_id") or obj.get("game_id") or ""
                for code in used:
                    hits[code]["n"] += 1
                    if len(hits[code]["refs"]) < max_refs:
                        hits[code]["refs"].append(f"{path}:{lineno}({rid})")
    return hits


def scenario_card_names(path: str | Path) -> list[str]:
    """Card names on both sides of a scenario file.

    Zone lists hold either plain names or {"card": name, ...} dicts;
    anything else (positions like "set", counts, ...) is ignored.
    """
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("holdout: cannot read scenario %s: %s", path, e)
        return []
    names: list[str] = []
    if not isinstance(obj, dict):
        return names
    for side in (obj.get("p0"), obj.get("p1")):
        if not isinstance(side, dict):
            continue
        for zone in side.values():
            if not isinstance(zone, list):
                continue
            for item in zone:
                if isinstance(item, str) and item:
                    names.append(item)
                elif isinstance(item, dict) and isinstance(item.get("card"), str):
                    names.append(item["card"])
    return names


def scenario_codes(
    scenario_dir: str | Path = SCENARIO_DIR, names=None
) -> tuple[Counter[int], list[str]]:
    """Resolve every scenario's card names to codes.

    Returns (counts, unknowns). `counts[code]` = #scenarios using it.
    Unknown names are reported as "file:name" strings, never guessed
    (they simply do not count as training cards).
    """
    from ygo_zero.demos.names import CardNames

    names = names if names is not None else CardNames()
    counts: Counter[int] = Counter()
    unknowns: list[str] = []
    for path in sorted(Path(scenario_dir).glob("*.json")):
        seen: set[int] = set()
        for name in scenario_card_names(path):
            hit = names.resolve(name)
            if hit:
                seen.add(hit[0])
            else:
                unknowns.append(f"{path.name}:{name}")
        for code in seen:
            counts[code] += 1
    if unknowns:
        logger.warning("holdout: %d unresolvable scenario card names", len(unknowns))
    return counts, unknowns


def training_cards(
    roots: Iterable[str | Path] | None = None,
    *,
    pool_dir: str | Path = POOL_DIR,
    gauntlet_dir: str | Path = GAUNTLET_DIR,
    scenario_dir: str | Path = SCENARIO_DIR,
    holdout_file: str | Path = HOLDOUT_FILE,
    names=None,
) -> Counter[int]:
    """Every card code in any training source.

    `roots` are `data/demos/<dir>` files/dirs/globs (default: all of
    `data/demos`). Count = number of *distinct decks/sources* using the
    card: identical decklists across records count once, each ydk file
    counts once, each scenario side counts once. (Per-record frequencies
    live in `record_card_counts`.)
    """
    if roots is None:
        roots = [DEMOS_DIR]
    counts: Counter[int] = Counter()
    seen_decks: set[frozenset] = set()

    def add_deck(codes: set[int]) -> None:
        key = frozenset(codes)
        if codes and key not in seen_decks:
            seen_decks.add(key)
            for code in codes:
                counts[code] += 1

    for path in _expand_record_files(roots):
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                config = obj.get("config") if isinstance(obj, dict) else None
                if not isinstance(config, dict):
                    continue
                for side in (("deck0_main", "deck0_extra"), ("deck1_main", "deck1_extra")):
                    add_deck(record_config_codes({k: config.get(k, []) for k in side}))

    holdout_stems = set(read_stems(holdout_file))
    ydk_files = sorted(Path(gauntlet_dir).glob("*.ydk")) if Path(gauntlet_dir).is_dir() else []
    ydk_files += [
        p
        for p in sorted(Path(pool_dir).glob("*.ydk"))
        if p.stem not in holdout_stems and Path(pool_dir).is_dir()
    ]
    for path in ydk_files:
        try:
            add_deck(deck_codes(load_ydk(path)))
        except Exception as e:  # noqa: BLE001 - one bad deck must not kill the audit
            logger.warning("holdout: skipping unreadable deck %s: %s", path, e)

    if Path(scenario_dir).is_dir():
        scen_counts, _ = scenario_codes(scenario_dir, names=names)
        # scenario sides are distinct sources on top of deck identity
        for code, n in scen_counts.items():
            counts[code] += n
    return counts


def card_holdout_codes(
    pool_dir: str | Path = POOL_DIR,
    holdout_file: str | Path = CARD_HOLDOUT_FILE,
    min_decks: int = 8,
) -> set[int]:
    """Non-staple cards of the CARD_HOLDOUT decks ([] when the file is absent)."""
    stems = read_stems(holdout_file)
    if not stems:
        return set()
    pool = load_pool_decks(pool_dir)
    held = {s: pool[s] for s in stems if s in pool}
    missing = [s for s in stems if s not in pool]
    if missing:
        logger.warning("holdout: CARD_HOLDOUT stems without a pool deck: %s", missing)
    return {c for deck in held.values() for c in deck_codes(deck)} - staples(pool, min_decks)


def check_records(
    paths: Iterable[str | Path],
    codes: set[int] | None = None,
    **kwargs,
) -> list[str]:
    """Violations: records/decks whose cards intersect the card-holdout set.

    `paths` are record files/dirs/globs and/or `.ydk` files. Returns one
    human-readable string per violation (a record whose config decks, or a
    deck whose list, contains a card-holdout code).
    """
    if codes is None:
        codes = card_holdout_codes(**kwargs) if kwargs else card_holdout_codes()
    if not codes:
        return []
    violations: list[str] = []
    expanded: list[Path] = []
    for entry in paths:
        text = str(entry)
        if text.endswith(".ydk") and not any(ch in text for ch in "*?[]"):
            p = Path(text)
            if p.is_file():
                expanded.append(p)
            continue
        expanded.extend(_expand_record_files([entry]))
    for path in expanded:
        if path.suffix == ".ydk":
            try:
                used = deck_codes(load_ydk(path)) & codes
            except Exception as e:  # noqa: BLE001
                logger.warning("holdout: skipping unreadable deck %s: %s", path, e)
                continue
            for code in sorted(used):
                violations.append(f"{path}: deck uses card-holdout {code}")
            continue
        try:
            fh = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                config = obj.get("config") if isinstance(obj, dict) else None
                if not isinstance(config, dict):
                    continue
                used = record_config_codes(config) & codes
                if used:
                    rid = obj.get("line_id") or obj.get("game_id") or "?"
                    for code in sorted(used):
                        violations.append(f"{path}:{lineno} record {rid} uses card-holdout {code}")
    return violations


def deck_audit_rows(
    pool_dir: str | Path = POOL_DIR,
    holdout_file: str | Path = HOLDOUT_FILE,
    training: Counter[int] | None = None,
    record_counts: Counter[int] | None = None,
    min_decks: int = 8,
) -> list[dict]:
    """One row per pool deck for the card-holdout audit table.

    Row: stem, holdout (in HOLDOUT), n_nonstaple, n_seen (# non-staple
    cards with training count > 0), unseen_frac, record_hits (sum of
    per-record counts over non-staple cards). Sorted by unseen_frac desc,
    then n_nonstaple desc. `training`/`record_counts` may be precomputed
    (tests); when None they count as empty (all cards unseen).
    """
    training = training if training is not None else Counter()
    record_counts = record_counts if record_counts is not None else Counter()
    holdout_stems = set(read_stems(holdout_file))
    pool = load_pool_decks(pool_dir)
    generic = staples(pool, min_decks)
    rows = []
    for stem, deck in pool.items():
        nonstaple = deck_codes(deck) - generic
        n = len(nonstaple)
        seen = sum(1 for c in nonstaple if training.get(c, 0) > 0)
        rows.append(
            {
                "stem": stem,
                "holdout": stem in holdout_stems,
                "n_nonstaple": n,
                "n_seen": seen,
                "n_unseen": n - seen,
                "unseen_frac": (n - seen) / n if n else 1.0,
                "record_hits": sum(record_counts.get(c, 0) for c in nonstaple),
            }
        )
    rows.sort(key=lambda r: (-r["unseen_frac"], -r["n_nonstaple"], r["stem"]))
    return rows
