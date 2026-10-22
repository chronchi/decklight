"""Card-level holdout audit: which decks' cards are (nearly) unseen in training.

  audit [--min-decks 8]   table every pool deck's unseen fraction -> TSV
  propose --n 4           pick N card-holdout candidates -> proposal txt
  check <globs...>        fail on records/decks using CARD_HOLDOUT cards
  newcards [--dir decks/newcards] [--play 2]
                          audit the new-cards evaluation decks (+ engine smoke)

uv run python scripts/card_holdout.py audit
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from decklight.decks.holdout import (
    CARD_HOLDOUT_FILE,
    deck_audit_rows,
    deck_codes,
    load_pool_decks,
    record_card_counts,
    records_using,
    staples,
    training_cards,
)

AUDIT_TSV = REPO_ROOT / "data" / "cache" / "card_holdout_audit.tsv"
PROPOSAL_TXT = REPO_ROOT / "data" / "cache" / "card_holdout_proposal.txt"

# Heuristic style tags from deck stems (proposal diversity only, not gospel).
COMBO_HINTS = ("drytron", "spyral", "adamancipator", "synchron", "dragon-link", "dracotail",
               "mitsurugi", "fiendsmith", "maliss", "ignister", "wind-up", "nekroz")
CONTROL_HINTS = ("labrynth", "traptrix", "eldlich", "altergeist", "striker", "paleozoic",
                 "goat", "doomz", "magistus", "dinosaur")


def style_of(stem: str) -> str:
    s = stem.lower()
    combo = any(h in s for h in COMBO_HINTS)
    control = any(h in s for h in CONTROL_HINTS)
    if combo and not control:
        return "combo"
    if control and not combo:
        return "control"
    return "mixed"


def archetype_of(stem: str) -> str:
    """First stem token ('doomz-meta1' -> 'doomz'); meta suffixes stripped."""
    return stem.split("-")[0]


def training_roots() -> list[str]:
    """Record files feeding training, mirroring build_bc_shards.SOURCES.

    Read via AST (never executed: that module parses argv at import).
    """
    globs: list[str] = []
    try:
        tree = ast.parse((REPO_ROOT / "scripts" / "build_bc_shards.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SOURCES" for t in node.targets
            ):
                for patterns in ast.literal_eval(node.value).values():
                    globs.extend(patterns)
    except Exception as e:  # noqa: BLE001 - fallback covers a moved SOURCES
        print(f"(SOURCES unreadable: {e})", file=sys.stderr)
    if not globs:  # fallback: everything under data/demos
        globs = ["data/demos/**/*.jsonl"]
    files = sorted(
        {str(p) for g in globs for p in REPO_ROOT.glob(g) if p.is_file() and p.stat().st_size}
    )
    return files


def _card_names():
    try:
        from decklight.demos.names import CardNames

        return CardNames()
    except Exception as e:  # noqa: BLE001 - names are best-effort only
        print(f"(card names unavailable: {e})", file=sys.stderr)
        return None


def cmd_audit(min_decks: int) -> list[dict]:
    t0 = time.time()
    roots = training_roots()
    print(f"{len(roots)} record files")
    training = training_cards(roots)
    print(f"training cards: {len(training)} distinct ({time.time() - t0:.0f}s)")
    t1 = time.time()
    record_counts, n_records = record_card_counts(roots)
    print(f"records scanned: {n_records} ({time.time() - t1:.0f}s)")
    rows = deck_audit_rows(training=training, record_counts=record_counts, min_decks=min_decks)
    names = _card_names()
    pool = load_pool_decks()
    generic = staples(pool, min_decks)
    lines = ["stem\tholdout\tn_nonstaple\tn_seen\tn_unseen\tunseen_frac\trecord_hits\texamples"]
    for r in rows:
        examples = ""
        if names is not None and (stem_deck := pool.get(r["stem"])) is not None:
            unseen = sorted(c for c in deck_codes(stem_deck) - generic if training.get(c, 0) == 0)
            examples = "|".join(names.names.get(c, str(c)) for c in unseen[:3])
        lines.append(
            f"{r['stem']}\t{r['holdout']}\t{r['n_nonstaple']}\t{r['n_seen']}\t"
            f"{r['n_unseen']}\t{r['unseen_frac']:.3f}\t{r['record_hits']}\t{examples}"
        )
    AUDIT_TSV.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {AUDIT_TSV} ({time.time() - t0:.0f}s total)")
    print(f"{'stem':45s} {'hold':5s} {'cards':>5s} {'seen':>4s} {'unseen%':>7s} {'rechits':>7s}")
    for r in rows:
        if r["holdout"] or r["unseen_frac"] >= 0.5:
            print(
                f"{r['stem']:45s} {r['holdout']!s:5s} {r['n_nonstaple']:>5d} "
                f"{r['n_seen']:>4d} {r['unseen_frac']:>7.3f} {r['record_hits']:>7d}"
            )
    return rows


def cmd_propose(n: int, min_decks: int) -> None:
    rows = cmd_audit(min_decks)
    cands = [r for r in rows if r["holdout"] and r["unseen_frac"] >= 0.8]
    # Prefer different archetypes (first token), then a combo/control mix.
    picked: list[dict] = []
    seen_arch: set[str] = set()
    for r in cands:
        if archetype_of(r["stem"]) not in seen_arch:
            picked.append(r)
            seen_arch.add(archetype_of(r["stem"]))
        if len(picked) == n:
            break
    if len(picked) < n:  # second pass: allow same-archetype sisters
        for r in cands:
            if r not in picked:
                picked.append(r)
            if len(picked) == n:
                break
    styles = {style_of(r["stem"]) for r in picked}
    if len(picked) > 1 and len(styles) < 2:
        for r in cands:
            if r not in picked and style_of(r["stem"]) not in styles:
                picked[-1] = r
                break
    pool = load_pool_decks()
    generic = staples(pool, min_decks)
    roots = training_roots()
    training = training_cards(roots)
    names = _card_names()
    out = [
        "# Card-holdout proposal (ADR 0007): decks whose non-staple cards are unseen in training.",
        "# Claude decides the final set; this file is input only. Do NOT write decks/pool/CARD_HOLDOUT.",
        f"# min_decks={min_decks} n={n}",
        "",
    ]
    for r in picked:
        deck = pool[r["stem"]]
        nonstaple = deck_codes(deck) - generic
        seen_cards = sorted(c for c in nonstaple if training.get(c, 0) > 0)
        out += [
            f"== {r['stem']} (style={style_of(r['stem'])}, archetype={archetype_of(r['stem'])})",
            (
                f"   non-staple={r['n_nonstaple']} seen={r['n_seen']} "
                f"unseen_frac={r['unseen_frac']:.3f} record_hits={r['record_hits']}"
            ),
        ]
        if not seen_cards:
            out.append("   fully unseen: no training records to drop.")
            continue
        using = records_using(roots, set(seen_cards))
        drop_files: Counter[str] = Counter()
        for code in seen_cards:
            label = names.names.get(code, str(code)) if names else str(code)
            info = using[code]
            for ref in info["refs"]:
                drop_files[ref.split(":")[0]] += 1
            out.append(
                f"   leak {code} ({label}): {info['n']} records e.g. {', '.join(info['refs'][:3])}"
            )
        out.append(f"   records to drop: {sum(v['n'] for v in using.values())} record-uses in "
                   f"{len(drop_files)} files: {', '.join(sorted(drop_files))}")
    if len(picked) < n:
        out.append(f"\nNOTE: only {len(picked)} HOLDOUT decks reach unseen_frac >= 0.8 (wanted {n}).")
    PROPOSAL_TXT.parent.mkdir(parents=True, exist_ok=True)
    PROPOSAL_TXT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nwrote {PROPOSAL_TXT}")


def cmd_check(globs: list[str]) -> int:
    from decklight.decks.holdout import check_records

    if not CARD_HOLDOUT_FILE.exists():
        print("no decks/pool/CARD_HOLDOUT yet — nothing to check.")
        return 0
    violations = check_records(globs)
    for v in violations:
        print(v)
    if violations:
        print(f"{len(violations)} card-holdout violations.")
        return 1
    print("no card-holdout violations.")
    return 0


def cmd_newcards(deck_dir: Path, play: int) -> int:
    """Audit `decks/newcards` evaluation decks (ADR 0007 report-card input).

    Per deck: main/extra counts (40-44 / <=15), <=3 copies, unknown codes,
    non-staple count, seen-in-training count (must be 0), and a
    check_decks-style missing-namesake audit. Exit 1 on any violation.
    With `--play N`: N random-policy games vs a pool deck (seeds 0..N-1,
    600 decisions max) through CppEnv with no engine errors.
    """
    from collections import Counter

    from decklight.decks.ydk import load_ydk

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_decks import audit as namesake_audit

    deck_dir = deck_dir if deck_dir.is_absolute() else REPO_ROOT / deck_dir
    paths = sorted(deck_dir.glob("*.ydk"))
    if not paths:
        print(f"no decks in {deck_dir}")
        return 1
    names = _card_names()
    pool = load_pool_decks()
    generic = staples(pool)
    roots = training_roots()
    training = training_cards(roots)
    print(f"{len(paths)} newcards decks, training cards: {len(training)} distinct")
    bad = 0
    print(f"{'deck':14s} {'main':>4s} {'extra':>5s} {'nonst':>5s} {'seen':>4s}  status")
    for path in paths:
        deck = load_ydk(path)
        issues: list[str] = []
        n_main, n_extra = len(deck.main), len(deck.extra)
        if not 40 <= n_main <= 44:
            issues.append(f"main={n_main} (want 40-44)")
        if n_extra > 15:
            issues.append(f"extra={n_extra} (want <=15)")
        over = {c: n for c, n in Counter([*deck.main, *deck.extra]).items() if n > 3}
        if over:
            label = names.names.get(next(iter(over)), next(iter(over))) if names else next(iter(over))
            issues.append(f"{len(over)} cards with >3 copies e.g. {label}")
        codes = deck_codes(deck)
        if names is not None:
            unknown = sorted(c for c in codes if c not in names.names)
            if unknown:
                issues.append(f"{len(unknown)} unknown codes e.g. {unknown[0]}")
        nonstaple = codes - generic
        seen = sorted(c for c in nonstaple if training.get(c, 0) > 0)
        if seen:
            label = names.names.get(seen[0], seen[0]) if names else seen[0]
            issues.append(f"{len(seen)} non-staple cards seen in training e.g. {label}")
        if names is not None:
            for q, n, _who in namesake_audit(path, names):
                issues.append(f"missing namesake {q!r} (mentioned by {n})")
        print(
            f"{path.stem:14s} {n_main:>4d} {n_extra:>5d} {len(nonstaple):>5d} "
            f"{len(seen):>4d}  {'OK' if not issues else 'VIOLATION: ' + '; '.join(issues)}"
        )
        bad += bool(issues)
    if play > 0:
        bad += _smoke_newcards(paths, pool, play)
    if bad:
        print(f"{bad} newcards problem(s).")
        return 1
    print("all newcards decks OK.")
    return 0


def _smoke_newcards(paths: list[Path], pool: dict, games: int, steps: int = 600) -> int:
    """Random-policy games of each newcards deck vs a pool deck (CppEnv, no errors)."""
    import random

    from decklight.decks.ydk import load_ydk
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import DuelConfig, Engine

    foe = pool.get("salamangreat") or pool[min(pool)]
    foe_name = next(s for s, d in pool.items() if d is foe)
    engine = Engine()
    bad = 0
    for path in paths:
        deck = load_ydk(path)
        for g in range(games):
            rng = random.Random(g)
            try:
                env = CppEnv(engine, DuelConfig(deck, foe, seed=g))
                obs = env.reset()
                n = 0
                while obs is not None and n < steps:
                    obs = env.step(rng.randrange(max(int(obs["action_mask"].sum()), 1)))
                    n += 1
                errs = env.errors
            except Exception as e:  # noqa: BLE001 - one bad deck must not kill the audit
                print(f"{path.stem} vs {foe_name} seed {g}: EXCEPTION {e}"[:200])
                bad += 1
                continue
            if errs:
                print(f"{path.stem} vs {foe_name} seed {g}: ENGINE ERRORS {list(errs)[:2]}")
                bad += 1
            else:
                print(f"{path.stem} vs {foe_name} seed {g}: OK ({n} decisions)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit every pool deck's unseen fraction")
    a.add_argument("--min-decks", type=int, default=8)
    p = sub.add_parser("propose", help="propose N card-holdout decks")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--min-decks", type=int, default=8)
    c = sub.add_parser("check", help="fail on records/decks using CARD_HOLDOUT cards")
    c.add_argument("globs", nargs="+")
    n = sub.add_parser("newcards", help="audit decks/newcards evaluation decks")
    n.add_argument("--dir", default="decks/newcards")
    n.add_argument("--play", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "audit":
        cmd_audit(args.min_decks)
    elif args.cmd == "propose":
        cmd_propose(args.n, args.min_decks)
    elif args.cmd == "check":
        sys.exit(cmd_check(args.globs))
    elif args.cmd == "newcards":
        sys.exit(cmd_newcards(Path(args.dir), args.play))


if __name__ == "__main__":
    main()
