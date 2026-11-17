"""Option-head cross-attention mass by token group for the top activation, by chain length and target."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from decklight.diag.behaviour import ScoredFrame
from decklight.diag.common import spearman
from decklight.diag.windows import bucket_chain


def _mass(f: ScoredFrame) -> dict[str, float] | None:
    j = f.top_activation()
    if j is None or f.out.attn is None or f.out.token_groups is None:
        return None
    row = np.asarray(f.out.attn[j], float)
    mass: dict[str, float] = defaultdict(float)
    for name, w in zip(f.out.token_groups, row, strict=False):
        mass[name] += float(w)
    return dict(mass)


def attention_table(frames: list[ScoredFrame], targets: dict | None = None, *, progress=None) -> dict:
    by_chain: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_target: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    chain_frames: dict[str, list[tuple[dict[str, float], float]]] = defaultdict(list)
    mass_by_frame: list[tuple[dict[str, float], float]] = []
    n = 0
    for i, f in enumerate(frames):
        if progress is not None and (i + 1) % 100 == 0:
            progress(f"attention {i + 1}/{len(frames)} windows")
        mass = _mass(f)
        if mass is None:
            continue
        n += 1
        mass_by_frame.append((mass, f.p_act))
        chain = bucket_chain(f.window.chain)
        chain_frames[chain].append((mass, f.p_act))
        for g, v in mass.items():
            by_chain[chain][g].append(v)
        complete = None if targets is None else targets.get(f.window.group, {}).get("complete")
        label = "unknown" if complete is None else "completes" if int(complete) else "broken"
        for g, v in mass.items():
            by_target[label][g].append(v)
    all_groups = sorted({g for m, _ in mass_by_frame for g in m})

    def _rho(pairs: list[tuple[dict[str, float], float]], g: str, min_n: int) -> float | None:
        if len(pairs) < min_n:
            return None
        r = spearman([m.get(g, 0.0) for m, _ in pairs], [p for _, p in pairs])
        return round(r, 4) if np.isfinite(r) else None

    rho = {g: _rho(mass_by_frame, g, 3) for g in all_groups}
    rho_by_chain = {
        chain: {g: _rho(pairs, g, 10) for g in all_groups} for chain, pairs in chain_frames.items()
    }
    n_by_chain = {chain: len(pairs) for chain, pairs in chain_frames.items()}

    def table(d: dict) -> dict:
        return {
            key: {g: round(float(np.mean(v)), 4) for g, v in sorted(groups.items(), key=lambda t: -np.mean(t[1]))}
            for key, groups in d.items()
        }

    means = {g: float(np.mean([m.get(g, 0.0) for m, _ in mass_by_frame])) for g in all_groups} if n else {}
    return {
        "windows": n,
        "available": n > 0,
        "by_chain": table(by_chain),
        "by_target": table(by_target),
        "n_by_chain": n_by_chain,
        "spearman_by_chain": rho_by_chain,
        "spearman_with_p_act": dict(sorted(rho.items(), key=lambda t: -(abs(t[1]) if t[1] is not None else -1))),
        "mean_mass": dict(sorted(means.items(), key=lambda t: -t[1])),
        "top_groups": [g for g, _ in sorted(means.items(), key=lambda t: -t[1])[:5]],
    }
