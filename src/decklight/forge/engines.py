"""Forge F2 archetype selection by engine novelty (docs/plans/2026-10-29-deck-forge.md).

Pure functions on Counters of card codes. A list's cards = main + extra codes.
"""

from __future__ import annotations

from collections import Counter


def staples(arch_lists: dict[str, list[Counter]], frac: float = 0.15,
            never: set[int] | None = None) -> set[int]:
    """Cards appearing in at least `frac` of archetypes, except those in `never`.

    An archetype "has" a card when any of its lists has it. Hand traps and
    generic Extra Deck cards land here. Pass `never` = the cards that belong to a
    named archetype (non-empty setcodes): a popular engine (Mitsurugi, LaD Ritual,
    Fiendsmith) sits in many mixes and would otherwise be classed as a staple and
    vanish from every signature (NOTEBOOK 2026-10-30 forge-select).
    """
    n = len(arch_lists)
    if n == 0:
        return set()
    have: Counter = Counter()
    for lists in arch_lists.values():
        union: set[int] = set()
        for ms in lists:
            union.update(ms.keys())
        for code in union:
            have[code] += 1
    never = never or set()
    return {code for code, c in have.items() if c / n >= frac and code not in never}


def signature(lists: list[Counter], staples: set[int], min_share: float = 0.5) -> set[int]:
    """Non-staple cards present in >= `min_share` of the archetype's lists."""
    n = len(lists)
    if n == 0:
        return set()
    have: Counter = Counter()
    for ms in lists:
        for code in ms:
            if code not in staples:
                have[code] += 1
    return {code for code, c in have.items() if c / n >= min_share}


def novelty(sig: set[int], covered: set[int]) -> float:
    """Share of the signature not yet covered; 0.0 for an empty signature."""
    if not sig:
        return 0.0
    return len(set(sig) - set(covered)) / len(sig)


def overlap(sig_a: set[int], sig_b: set[int]) -> float:
    """Overlap coefficient: |a & b| / min(|a|, |b|); 0.0 when either is empty."""
    if not sig_a or not sig_b:
        return 0.0
    return len(set(sig_a) & set(sig_b)) / min(len(sig_a), len(sig_b))


def greedy_select(
    cands: dict[str, tuple[int, set[int]]],
    covered: set[int],
    n: int,
    min_novelty: float = 0.5,
) -> list[dict]:
    """Pick `n` archetypes greedily by popularity * novelty.

    Each round picks the archetype with the highest ``popularity * novelty(sig,
    covered)`` among those with novelty >= ``min_novelty``, appends
    ``{arch, popularity, novelty, score, shares}`` and adds its signature to
    ``covered``. Stops early when nothing qualifies. Ties break by arch name.

    ``shares`` maps each previously picked archetype to its overlap with the
    pick (the original covered archetypes' per-arch signatures are not passed
    in, only their union, so overlaps with them cannot be computed here; the
    forge_select table reports those separately).
    """
    covered = set(covered)
    remaining = dict(cands)
    picked_sigs: dict[str, set[int]] = {}
    out: list[dict] = []
    for _ in range(n):
        best: str | None = None
        best_nov = 0.0
        best_score = 0.0
        for arch in sorted(remaining):
            pop, sig = remaining[arch]
            nov = novelty(set(sig), covered)
            if nov < min_novelty:
                continue
            score = pop * nov
            if best is None or score > best_score:
                best, best_nov, best_score = arch, nov, score
        if best is None:
            break
        pop, sig = remaining.pop(best)
        sig = set(sig)
        shares = {prev: overlap(sig, psig) for prev, psig in picked_sigs.items()}
        out.append({"arch": best, "popularity": pop, "novelty": best_nov,
                    "score": pop * best_nov, "shares": shares})
        picked_sigs[best] = sig
        covered |= sig
    return out
