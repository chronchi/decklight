"""Frequency vs value-head-impact primers: which ranking makes more sense? (user, 2026-10-18)

Reads the scored per-game summaries (``build_primers.py --score`` cache) and builds both primer kinds from the same
games: *frequency* (the summaries with their win-chance curve stripped) and *impact* (as scored). Reports:

1. **Staple share**: the share of top-8 key cards that are generic staples (in the main deck of >= 10% of all decks).
   A primer about *this* deck should name its engine, not Ash Blossom.
2. **Split-half stability**: games split in two halves by a hash of (seed, decks, game_idx); per deck the overlap
   (Jaccard) of the top-5 key cards and top-3 chokepoints built from each half. Noise check with 20 games per deck.
3. **Held-out validity**: rankings built from half A, judged on half B with one neutral event definition for both
   methods (step answer pairs: the opponent's next play in the same turn after a play of the deck):
   - chokepoints: the deck's loss rate in B games where a card in A's top-3 chokepoints was answered, minus the loss
     rate in B games where only other cards were answered (bigger = the ranking found the answers that matter);
   - key cards: the deck's win rate in B games where it played >= 2 of A's top-3 key cards minus where it played
     none (bigger = the ranking found the cards that matter; confounded by draw luck, same for both methods).
4. Side-by-side text for a few decks (``--show``).

Usage: uv run python scripts/audit/compare_primers.py <summaries.json.gz> [--show N] [--json out.json]
Exit 2 when the cache is not scored or a half has no decks.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

STAPLE_SHARE = 0.10
MIN_SIDE_GAMES = 8  # a deck needs this many games in each half to count in the split-half / held-out numbers


def _half(s: dict) -> int:
    key = f"{s['seed']}|{'|'.join(s['decks'])}|{s.get('game_idx')}".encode()
    return hashlib.sha1(key).digest()[0] & 1


def _strip(summaries: list[dict]) -> list[dict]:
    out = copy.deepcopy(summaries)
    for s in out:
        s.pop("w_end", None)
        for st in s["steps"]:
            st.pop("w", None)
            st.pop("credit", None)
    return out


def _build(P, summaries, cards, names):
    # aggregate() canon-folds summaries in place; give it a copy so the caller's list stays reusable.
    return P.aggregate(copy.deepcopy(summaries), cards, names, {})["decks"]


def _top(entry: dict | None, field: str, n: int) -> list[int]:
    return [c["code"] for c in (entry or {}).get(field, [])[:n]]


def _jacc(a: list[int], b: list[int]) -> float | None:
    if not a and not b:
        return None
    return len(set(a) & set(b)) / len(set(a) | set(b))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv=None) -> int:
    import decklight.analysis.primer as P
    from decklight.analysis.events import deck_path
    from decklight.decks import load_ydk
    from decklight.engine.resources import load_cards
    from decklight.web.view import DB

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summaries", type=Path)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    summ = json.loads(gzip.decompress(args.summaries.read_bytes()))
    for s in summ:
        P._canon_summary(s)
    if not all(P._is_scored(s) for s in summ):
        print(
            f"ALERT compare_primers: {sum(not P._is_scored(s) for s in summ)} unscored summaries", flush=True
        )
        return 2
    cards, names = load_cards(), DB.names()
    P.MIN_GAMES = 1

    # staples: base on every deck list in the games
    by_name = {d: load_ydk(deck_path(d)) for s in summ for d in s["decks"]}
    hash_of_name = {d: P.hash_of(x) for d, x in by_name.items()}
    mains: dict[str, set[int]] = {hash_of_name[d]: {P.canon(c) for c in x.main} for d, x in by_name.items()}
    df = Counter(c for m in mains.values() for c in m)
    staples = {c for c, n in df.items() if n >= STAPLE_SHARE * len(mains)}

    full = {"frequency": _build(P, _strip(summ), cards, names), "impact": _build(P, summ, cards, names)}
    halves = {h: [s for s in summ if _half(s) == h] for h in (0, 1)}
    half = {
        m: {h: _build(P, _strip(halves[h]) if m == "frequency" else halves[h], cards, names) for h in (0, 1)}
        for m in ("frequency", "impact")
    }
    if not half["impact"][0] or not half["impact"][1]:
        print("ALERT compare_primers: a half has no decks", flush=True)
        return 2

    # neutral held-out events per (deck hash, game): answered codes, played codes, outcome
    def events_of(s):
        hs = [hash_of_name[d] for d in s["decks"]]
        steps = s["steps"]
        out = {}
        for seat in (0, 1):
            played = {
                st["code"]
                for st in steps
                if st["player"] == seat and st["kind"] in P.PLAY_KINDS and st["code"]
            }
            answered = {steps[i]["code"] for i, j in P._answer_pairs(steps) if steps[i]["player"] == seat}
            out[hs[seat]] = (played, answered, s["winner"] == seat)
        return out

    ev = {h: [events_of(s) for s in halves[h]] for h in (0, 1)}
    games_in = {h: Counter(k for e in ev[h] for k in e) for h in (0, 1)}
    both = [
        k for k in full["impact"] if games_in[0][k] >= MIN_SIDE_GAMES and games_in[1][k] >= MIN_SIDE_GAMES
    ]

    report: dict = {
        "games": len(summ),
        "decks": len(full["impact"]),
        "decks_both_halves": len(both),
        "staples": len(staples),
    }
    for m in ("frequency", "impact"):
        tops = [c["code"] for e in full[m].values() for c in e.get("key_cards", [])[:8]]
        r = {"staple_share_top8": round(sum(c in staples for c in tops) / max(1, len(tops)), 3)}
        r["jaccard_key5"] = round(
            _mean(
                _jacc(_top(half[m][0].get(k), "key_cards", 5), _top(half[m][1].get(k), "key_cards", 5))
                for k in both
            ),
            3,
        )
        r["jaccard_choke3"] = round(
            _mean(
                _jacc(_top(half[m][0].get(k), "chokepoints", 3), _top(half[m][1].get(k), "chokepoints", 3))
                for k in both
            ),
            3,
        )
        # held-out validity, both directions (A->B and B->A), pooled
        ch_top, ch_other, k_hi, k_none = [], [], [], []
        for a, b in ((0, 1), (1, 0)):
            for e in ev[b]:
                for k, (played, answered, won) in e.items():
                    if k not in both:
                        continue
                    tc = set(_top(half[m][a].get(k), "chokepoints", 3))
                    if answered & tc:
                        ch_top.append(not won)
                    elif answered:
                        ch_other.append(not won)
                    tk = set(_top(half[m][a].get(k), "key_cards", 3))
                    n = len(played & tk)
                    if n >= 2:
                        k_hi.append(won)
                    elif n == 0:
                        k_none.append(won)
        r["choke_loss_top"] = round(_mean(ch_top), 3)
        r["choke_loss_other"] = round(_mean(ch_other), 3)
        r["choke_gap_pp"] = round(100 * (_mean(ch_top) - _mean(ch_other)), 1)
        r["choke_n"] = [len(ch_top), len(ch_other)]
        r["key_win_hi"] = round(_mean(k_hi), 3)
        r["key_win_none"] = round(_mean(k_none), 3)
        r["key_gap_pp"] = round(100 * (_mean(k_hi) - _mean(k_none)), 1)
        r["key_n"] = [len(k_hi), len(k_none)]
        report[m] = r
    print(json.dumps(report, indent=1), flush=True)

    # side by side
    shown = 0
    for k in sorted(full["impact"], key=lambda h: -full["impact"][h]["games"]):
        if shown >= args.show:
            break
        fi, ff = full["impact"][k], full["frequency"][k]
        if not fi.get("key_cards"):
            continue
        shown += 1
        print(f"\n== {fi['archetype']} ({fi['games']} games; {', '.join(fi.get('files', [])[:2])})")
        print("  key, frequency: " + "; ".join(c["name"] for c in ff["key_cards"][:6]))
        print(
            "  key, impact:    " + "; ".join(f"{c['name']} {c['impact']:+.1f}" for c in fi["key_cards"][:6])
        )
        print("  stop, frequency: " + "; ".join(c["name"] for c in ff["chokepoints"][:4]))
        print(
            "  stop, impact:    "
            + "; ".join(f"{c['name']} {c.get('cost', 0):+.1f}" for c in fi["chokepoints"][:4])
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
