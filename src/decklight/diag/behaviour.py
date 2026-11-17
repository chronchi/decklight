"""Behaviour tables: p(activate) / pass share / card-level probabilities, optionally for two checkpoints."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from decklight.diag.adapter import AdapterOutput
from decklight.diag.common import phase_name
from decklight.diag.windows import Window, bucket_chain, window_obs
from decklight.engine.actions import Kind


@dataclass
class ScoredFrame:
    window: Window
    obs: dict
    out: AdapterOutput
    p_act: float
    p_pass: float
    argmax: int
    argmax_kind: int

    @property
    def argmax_activation(self) -> bool:
        return self.argmax_kind == int(Kind.ACTIVATE)

    def top_activation(self) -> int | None:
        """Legal activation option with the highest probability (None when no activation is legal)."""
        acts = np.asarray(self.obs["actions"])
        mask = np.asarray(self.obs["action_mask"]).astype(bool)
        js = [int(j) for j in np.flatnonzero(mask) if int(acts[j][0]) == int(Kind.ACTIVATE)]
        return max(js, key=lambda j: self.out.probs[j]) if js else None

    def card_probs(self) -> dict[int, float]:
        """Total probability of each activation card's options at this window."""
        acts = np.asarray(self.obs["actions"])
        legal = np.asarray(self.obs["action_mask"]).astype(bool)
        out: dict[int, float] = defaultdict(float)
        for i in np.flatnonzero(legal):
            if int(acts[i][0]) == int(Kind.ACTIVATE) and acts[i][2]:
                out[int(acts[i][2])] += float(self.out.probs[i])
        return dict(out)


def score_windows(engine, windows: list[Window], adapter, *, progress=None) -> tuple[list[ScoredFrame], int]:
    """Replay + run one adapter on every window; returns (frames, number of windows that failed to replay)."""
    frames, diverged = [], 0
    for i, w in enumerate(windows):
        try:
            obs = window_obs(engine, w)
        except RuntimeError:
            diverged += 1
            continue
        out = adapter.run(obs)
        acts = np.asarray(obs["actions"])
        mask = np.asarray(obs["action_mask"]).astype(bool)
        p_act = float(sum(out.probs[j] for j in np.flatnonzero(mask) if int(acts[j][0]) == int(Kind.ACTIVATE)))
        p_pass = float(sum(out.probs[j] for j in np.flatnonzero(mask) if int(acts[j][0]) == int(Kind.PASS)))
        argmax = int(np.argmax(out.probs))
        frames.append(ScoredFrame(w, obs, out, p_act, p_pass, argmax, int(acts[argmax][0])))
        if progress is not None and (i + 1) % 100 == 0:
            progress(f"{adapter.name}: {i + 1}/{len(windows)} windows")
    return frames, diverged


def _agg(vals: list[float]) -> dict:
    return {
        "n": len(vals),
        "p_act_mean": round(float(np.mean(vals)), 4) if vals else None,
        "p_act_median": round(float(np.median(vals)), 4) if vals else None,
    }


def behaviour_table(frames: list[ScoredFrame], *, top_cards: int = 20, names: dict[int, str] | None = None) -> dict:
    """Per checkpoint: by chain x turn owner x phase, and by activation card (top `top_cards` by n)."""
    by_group: dict[tuple[str, str, str], list[ScoredFrame]] = defaultdict(list)
    for f in frames:
        who = "own" if f.window.turn_owner == 1 else "opp" if f.window.turn_owner == 0 else "?"
        by_group[(bucket_chain(f.window.chain), who, phase_name(f.window.phase))].append(f)
    rows = []
    for (chain, who, phase), fs in sorted(by_group.items()):
        vals = [f.p_act for f in fs]
        rows.append({
            "chain": chain, "turn": who, "phase": phase, **_agg(vals),
            "pass_share": round(sum(f.argmax_kind == int(Kind.PASS) for f in fs) / len(fs), 4),
            "activation_share": round(sum(f.argmax_activation for f in fs) / len(fs), 4),
        })  # fmt: skip
    cards: dict[str, dict] = defaultdict(lambda: {"codes": set(), "vals": []})
    names = names or {}
    for f in frames:
        per_name: dict[str, float] = defaultdict(float)
        for code, p in f.card_probs().items():
            name = names.get(code) or f"#{code}"
            per_name[name] += float(p)
            cards[name]["codes"].add(code)
        for name, p in per_name.items():
            cards[name]["vals"].append(p)
    by_card = [
        {"name": name, "codes": sorted(d["codes"]), **_agg(d["vals"]), "windows": len(d["vals"])}
        for name, d in sorted(cards.items(), key=lambda t: (-len(t[1]["vals"]), t[0]))[:top_cards]
    ]
    return {
        "windows": len(frames),
        "starts": len({f.window.group for f in frames}),
        "p_act_mean": round(float(np.mean([f.p_act for f in frames])), 4) if frames else None,
        "p_act_median": round(float(np.median([f.p_act for f in frames])), 4) if frames else None,
        "argmax_activation_share": round(float(np.mean([f.argmax_activation for f in frames])), 4) if frames else None,
        "by_chain_turn_phase": rows,
        "by_card": by_card,
    }


def render_label(obs: dict, action: int, names: dict[int, str]) -> str:
    """Human-readable option label (the same shape as `window_read.label`)."""
    from decklight.demos.game import hidden_card, option_desc

    acts = np.asarray(obs["actions"])
    kind, code = int(acts[action][0]), int(acts[action][2])
    name = names.get(code, "") if code else hidden_card(obs, action)
    try:
        verb = Kind(kind).name.lower()
    except ValueError:
        verb = str(kind)
    try:
        desc = option_desc(obs, action)
    except Exception:  # noqa: BLE001 - labels are cosmetic: never fail a diagnosis on one
        desc = ""
    return f"{verb} {name} {desc!r}".strip()


def pair_behaviour(frames_a: list[ScoredFrame], frames_b: list[ScoredFrame], names: dict[int, str]) -> dict:
    """Side-by-side tables and the windows where the two argmax choices differ."""
    table_a = behaviour_table(frames_a, names=names)
    table_b = behaviour_table(frames_b, names=names)
    idx_b = {(f.window.group, f.window.k): f for f in frames_b}
    disagreements = []
    for fa in frames_a:
        fb = idx_b.get((fa.window.group, fa.window.k))
        if fb is None or fa.argmax == fb.argmax:
            continue
        disagreements.append({
            "group": fa.window.group, "k": fa.window.k, "line_id": fa.window.line_id,
            "chain": fa.window.chain, "phase": phase_name(fa.window.phase),
            "turn": "own" if fa.window.turn_owner == 1 else "opp",
            "a_label": render_label(fa.obs, fa.argmax, names), "b_label": render_label(fb.obs, fb.argmax, names),
            "a_p": round(float(fa.out.probs[fa.argmax]), 4),
            "b_p": round(float(fb.out.probs[fb.argmax]), 4),
            "a_activation": fa.argmax_activation, "b_activation": fb.argmax_activation,
        })  # fmt: skip
    return {
        "A": table_a,
        "B": table_b,
        "disagreements": disagreements,
        "disagreement_rate": round(len(disagreements) / len(frames_a), 4) if frames_a else None,
    }


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")
