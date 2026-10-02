"""Extract structured combo lines from fetched guides (ADR 0002 stage 2).

Usage:
    uv run python scripts/extract_lines.py --guides "data/guides/<deck>/*.txt" \\
        [--gold <gold guide>] --deck <deck> --ydk <deck.ydk> --out <out.jsonl>
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from ygo_zero.demos import llm as _llm
from ygo_zero.demos.extract import (
    extract_guide_decklist,
    extract_lines,
    sequence_overlap,
    step_card_sequence,
)
from ygo_zero.demos.names import CardNames


def load_ydk(path: Path) -> set[int]:
    """Main + extra deck codes (side deck excluded)."""
    codes: set[int] = set()
    side = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!side"):
            side = True
            continue
        if not side and line.isdigit():
            codes.add(int(line))
    return codes


def summarize(deck: str, per_source: dict[str, list]) -> None:
    total = sum(len(v) for v in per_source.values())
    print(f"== {deck}: {total} lines from {len(per_source)} sources ==")
    grand_names = resolved = 0
    unresolved = Counter()
    for src, lines in per_source.items():
        steps = [len(line.steps) for line in lines]
        avg = sum(steps) / len(steps) if steps else 0.0
        print(f"  {src}: {len(lines)} lines, steps/line {steps} (avg {avg:.1f})")
        for line in lines:
            codes = (
                line.hand_codes
                + line.requires_codes
                + line.end_board_codes
                + [s.card_code for s in line.steps]
            )
            names_n = (
                len(line.hand)
                + len(line.requires)
                + len(line.end_board)
                + sum(1 for s in line.steps if s.card)
            )
            grand_names += names_n
            resolved += sum(1 for c in codes if c is not None)
            unresolved.update(line.unresolved_names)
    pct = 100 * resolved / grand_names if grand_names else 0.0
    print(f"  names: {resolved}/{grand_names} resolved ({pct:.1f}%)")
    print("  top unresolved:")
    for name, n in unresolved.most_common(20):
        print(f"    {n}x {name}")
    print(f"  llm backend: {_llm.last_backend}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guides", nargs="+", required=True, help="glob(s) for guide .txt files")
    ap.add_argument("--gold", default=None, help="gold guide for recall comparison")
    ap.add_argument("--deck", required=True)
    ap.add_argument("--ydk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-calls", type=int, default=60)
    ap.add_argument(
        "--write-decklists",
        default=None,
        help="write per-guide decklists to <dir>/<deck>/<source>.json when a guide has one",
    )
    args = ap.parse_args()

    files: list[Path] = []
    for g in args.guides:
        files.extend(Path(p) for p in sorted(glob.glob(g)))
    files = [f for f in files if f.is_file()]
    if not files:
        raise SystemExit("no guide files matched")

    names = CardNames()
    decklist = load_ydk(Path(args.ydk))
    print(f"decklist {args.ydk}: {len(decklist)} codes")

    calls = [0]

    def counting_complete(prompt: str, **kw) -> str:
        calls[0] += 1
        if calls[0] > args.max_calls:
            raise RuntimeError(f"LLM call budget ({args.max_calls}) exceeded")
        return _llm.complete(prompt, **kw)

    per_source: dict[str, list] = {}
    all_lines = []
    dupes = 0
    for f in files:
        lines = extract_lines(f, args.deck, decklist, names=names, complete_fn=counting_complete)
        kept = []
        seen = set()
        for line in lines:
            sig = tuple(step_card_sequence(line))
            if sig and sig in seen:
                dupes += 1
                continue
            seen.add(sig)
            kept.append(line)
        per_source[f.name] = kept
        all_lines.extend(kept)
        if args.write_decklists:
            dl = extract_guide_decklist(f.read_text(encoding="utf-8", errors="replace"), names)
            if dl:
                d = Path(args.write_decklists) / args.deck
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{f.stem}.json").write_text(json.dumps(dl, indent=1))
                print(f"  decklist {f.name}: main={len(dl['main'])} extra={len(dl['extra'])}")
    summarize(args.deck, per_source)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for line in all_lines:
            fh.write(json.dumps(line.to_json()) + "\n")
    print(f"wrote {len(all_lines)} lines -> {out} ({calls[0]} LLM calls, {dupes} dupes dropped)")

    if args.gold:
        gold = extract_lines(args.gold, args.deck, decklist, names=names, complete_fn=counting_complete)
        print(f"== gold {Path(args.gold).name}: {len(gold)} lines ==")
        fetched_seq = [(line.id, step_card_sequence(line)) for line in all_lines]
        for g in gold:
            gs = step_card_sequence(g)
            best, score = None, 0.0
            for lid, fs in fetched_seq:
                s = sequence_overlap(gs, fs)
                if s > score:
                    best, score = lid, s
            print(f"  gold '{g.title[:70]}' ({len(gs)} step cards) -> best {best} overlap {score:.2f}")


if __name__ == "__main__":
    main()
