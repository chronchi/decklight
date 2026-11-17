"""Weight-editing experiment (NOTEBOOK 2026-11-09 16:30): play games between two checkpoints, collect decision
windows, and score a checkpoint's behaviour on fixed windows.

Window kinds, read from the observation of the player to act (docs/design/obs_spec.md `global`):
- `attack`: my turn, MSG_SELECT_BATTLECMD, at least one ATTACK option legal;
- `interrupt`: the opponent's turn, MSG_SELECT_CHAIN, at least one ACTIVATE option legal;
- `battle`: my main phase (MSG_SELECT_IDLECMD) with both TO_BATTLE and TO_END legal and a monster on my field: the
  "go to battle or end the turn" choice. Its rate is read only where the policy leaves the main phase (argmax TO_BATTLE
  or TO_END), so combo actions taken first do not count;
- `other`: every other decision with 2+ legal options (sampled when collecting).

`score` also reports `interrupt_t1`: interrupt windows on turn 1 (the opponent's opening combo:
NOTEBOOK 2026-11-09 17:30). Games also report `opp_field_removed`: the opponent's field cards (mzone + szone) at the
first decision of my turn minus at the first decision of the next turn, over my turns where the opponent started
with 2+ field cards (board breaking); `opp_board_left`: the opponent's field cards at the first decision after its
own turn (how much it developed); `first_use_window` / `first_use_frac`: on opponent turns where I activated, the
index of my first activation among my interrupt windows that turn (1 = the first chance), and that index over the
turn's window count (how early I interrupt).

A checkpoint is any state dict with X's tensor names and shapes (`load_policy` builds the net from X's run args, so
an edited file needs no metrics.jsonl).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from decklight.decks import load_ydk
from decklight.engine import constants as C
from decklight.engine.actions import Kind
from decklight.engine.cpp_env import CppEnv
from decklight.engine.duel import DuelConfig, Engine
from decklight.eval.bc_policy import BCPolicy
from decklight.model.bc_data import OBS_KEYS
from decklight.model.ckpt import load_state

ROOT = Path(__file__).resolve().parents[3]
X_CKPT = ROOT / "runs/window-ppo-20261108-0940/train/export-X/policy/best.pt"
EMB = ROOT / "data/cards/emb/qwen3-0.6b-d512.npz"
KINDS = ("attack", "interrupt", "battle", "other")


def load_policy(ckpt: str | Path, *, threads: int = 1) -> BCPolicy:
    """BCPolicy with X's architecture and `ckpt`'s weights (strict: same names and shapes as X)."""
    pol = BCPolicy(X_CKPT, EMB, threads=threads)
    if Path(ckpt).resolve() != X_CKPT.resolve():
        state = load_state(Path(ckpt))
        pol.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        pol.name = f"bc:{Path(ckpt).stem}"
    pol.model.eval()
    return pol


def window_kind(obs: dict) -> str | None:
    """`attack` / `interrupt` / `other`, or None for a forced decision (fewer than 2 legal options)."""
    mask = np.asarray(obs["action_mask"], bool)
    if mask.sum() < 2:
        return None
    g = obs["global"]
    kinds = set(np.asarray(obs["actions"])[mask, 0].tolist())
    if g[5] == 1 and g[21] == C.MSG_SELECT_BATTLECMD and int(Kind.ATTACK) in kinds:
        return "attack"
    if g[5] == 0 and g[21] == C.MSG_SELECT_CHAIN and int(Kind.ACTIVATE) in kinds:
        return "interrupt"
    if g[5] == 1 and g[21] == C.MSG_SELECT_IDLECMD and g[8] > 0 and {int(Kind.TO_BATTLE), int(Kind.TO_END)} <= kinds:
        return "battle"
    return "other"


def target_kind(kind: str) -> int | None:
    return {"attack": int(Kind.ATTACK), "interrupt": int(Kind.ACTIVATE), "battle": int(Kind.TO_BATTLE)}.get(kind)


@dataclass
class SeatStats:
    """Behaviour of one policy over its games (argmax play)."""

    games: int = 0
    wins: float = 0.0
    windows: dict = field(default_factory=lambda: {k: 0 for k in KINDS})
    took: dict = field(default_factory=lambda: {"attack": 0, "interrupt": 0, "battle": 0})  # attacked / activated / to battle
    left: int = 0  # battle windows where it left the main phase (TO_BATTLE or TO_END)
    attack_turns: int = 0  # own turns with an attack window
    attacked_turns: int = 0  # ... where it attacked at least once
    opp_turns_live: int = 0  # opponent turns with an interrupt window
    opp_turns_used: int = 0  # ... where it activated at least once
    damage: list = field(default_factory=list)  # opponent LP lost per own turn
    removed: list = field(default_factory=list)  # opponent field cards removed per own turn (opponent started with 2+)
    opp_board: list = field(default_factory=list)  # opponent field cards right after each opponent turn
    first_use: list = field(default_factory=list)  # (index of my first activation, interrupt windows) per opponent turn

    def summary(self) -> dict:
        r = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
        return {"games": self.games, "win_rate": r(self.wins, self.games), "windows": dict(self.windows),
                "attack_rate": r(self.took["attack"], self.windows["attack"]),
                "interrupt_rate": r(self.took["interrupt"], self.windows["interrupt"]),
                "battle_rate": r(self.took["battle"], self.left), "battle_left": self.left,
                "attack_turn_rate": r(self.attacked_turns, self.attack_turns),
                "interrupt_turn_rate": r(self.opp_turns_used, self.opp_turns_live),
                "damage_per_turn": round(float(np.mean(self.damage)), 1) if self.damage else None,
                "opp_field_removed": round(float(np.mean(self.removed)), 3) if self.removed else None,
                "break_turns": len(self.removed),
                "opp_board_left": round(float(np.mean(self.opp_board)), 3) if self.opp_board else None,
                "first_use_window": round(float(np.mean([i for i, _ in self.first_use])), 2) if self.first_use else None,
                "first_use_frac": round(float(np.mean([i / n for i, n in self.first_use])), 3) if self.first_use else None}  # fmt: skip


def play_game(engine: Engine, pols: tuple, decks: tuple, seed: int, *, max_steps: int = 3000,
              stats: tuple | None = None, keep=None) -> dict:  # fmt: skip
    """One argmax game, pols[0] in seat 0. `stats` (one SeatStats per seat) is updated; `keep(obs, kind, actor,
    action)` is called at every window. Returns winner (0/1/2, None if cut) and the turn count."""
    env = CppEnv(engine, DuelConfig(decks[0], decks[1], seed=seed))
    for p in pols:
        if p.guard is not None:
            p.guard.reset()
    obs, steps = env.reset(), 0
    turn_seen: dict[tuple[int, int], set] = {}
    iwin: dict[tuple[int, int], list] = {}  # (seat, opponent turn) -> [interrupt windows, index of first activation]
    first: dict[int, tuple[int, tuple[int, int]]] = {}  # turn -> (turn player, LP by seat) at its first decision
    fieldc: dict[int, tuple[int, int]] = {}  # turn -> field cards (mzone + szone) by seat at its first decision
    while obs is not None and steps < max_steps:
        p = env.player
        a = int(pols[p]({k: v[None] for k, v in obs.items()}, np.array([0]))[0])
        kind = window_kind(obs)
        g = obs["global"]
        turn = int(g[2])
        if turn not in first:
            lp = (int(g[0]), int(g[1])) if p == 0 else (int(g[1]), int(g[0]))
            first[turn] = (p if g[5] == 1 else 1 - p, lp)
            mine, theirs = int(g[8]) + int(g[9]), int(g[15]) + int(g[16])
            fieldc[turn] = (mine, theirs) if p == 0 else (theirs, mine)
        if kind is not None:
            if keep is not None:
                keep(obs, kind, p, a)
            if stats is not None:
                s = stats[p]
                s.windows[kind] += 1
                tk = target_kind(kind)
                chosen = int(obs["actions"][a][0])
                if kind == "battle":
                    if chosen in (int(Kind.TO_BATTLE), int(Kind.TO_END)):
                        s.left += 1
                        s.took["battle"] += chosen == int(Kind.TO_BATTLE)
                elif tk is not None:
                    took = chosen == tk
                    s.took[kind] += took
                    seen = turn_seen.setdefault((p, turn), set())
                    seen.add(kind)
                    if kind == "interrupt":
                        live = iwin.setdefault((p, turn), [0, 0])
                        live[0] += 1
                        if took and not live[1]:
                            live[1] = live[0]
                    if took:
                        seen.add(kind + "_took")
        obs = env.step(a)
        steps += 1
    if stats is not None:
        for (p, _turn), seen in turn_seen.items():
            s = stats[p]
            if "attack" in seen:
                s.attack_turns += 1
                s.attacked_turns += "attack_took" in seen
            if "interrupt" in seen:
                s.opp_turns_live += 1
                s.opp_turns_used += "interrupt_took" in seen
        for (p, _turn), (n, i) in iwin.items():
            if i:
                stats[p].first_use.append((i, n))
    winner = env.winner if obs is None else None
    if stats is not None:
        turns = sorted(first)
        for t, nxt in zip(turns, turns[1:]):  # LP the turn player's opponent lost by the next turn's first decision
            tp, lp = first[t]
            stats[tp].damage.append(lp[1 - tp] - first[nxt][1][1 - tp])
            stats[1 - tp].opp_board.append(fieldc[nxt][tp])
            if fieldc[t][1 - tp] >= 2:
                stats[tp].removed.append(fieldc[t][1 - tp] - fieldc[nxt][1 - tp])
        if turns and winner in (0, 1) and first[turns[-1]][0] == winner:  # a kill on its own turn: all LP left
            tp, lp = first[turns[-1]]
            stats[tp].damage.append(lp[1 - tp])
        for p in (0, 1):
            stats[p].games += 1
            stats[p].wins += 1.0 if winner == p else 0.5 if winner == 2 else 0.0
    return {"winner": winner, "turns": int(obs["global"][2]) if obs is not None else None, "steps": steps}


def deck_pairs(holdout: bool, n_pairs: int, seed: int) -> list[tuple[str, str]]:
    pool = sorted(p.stem for p in (ROOT / "decks/pool").glob("*.ydk"))
    hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
    names = [d for d in pool if (d in hold) == holdout]
    rng = random.Random(seed)
    return [tuple(rng.sample(names, 2)) for _ in range(n_pairs)]


def load_deck(name: str):
    return load_ydk(ROOT / "decks/pool" / f"{name}.ydk")


class WindowStore:
    """Collected windows as stacked OBS_KEYS arrays plus metadata, saved as one compressed npz."""

    def __init__(self, other_every: int = 8, seed: int = 0):
        self.cols: dict[str, list] = {k: [] for k in OBS_KEYS}
        self.meta: dict[str, list] = {"kind": [], "actor": [], "action": [], "game": []}
        self.other_every, self.rng, self.game = other_every, random.Random(seed), 0

    def __call__(self, obs, kind, actor, action) -> None:
        if kind == "other" and self.rng.randrange(self.other_every):
            return
        for k in OBS_KEYS:
            self.cols[k].append(np.asarray(obs[k]).copy())
        for k, v in (("kind", kind), ("actor", actor), ("action", action), ("game", self.game)):
            self.meta[k].append(v)

    def save(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.stack(v) for k, v in self.cols.items()}
        arrays.update({f"meta_{k}": np.asarray(v) for k, v in self.meta.items()})
        np.savez_compressed(path, **arrays)
        return len(self.meta["kind"])


def load_windows(paths: list[Path], *, x_probs: bool = False) -> dict[str, np.ndarray]:
    """Stack window files. `x_probs`: also X's legal softmax per window (`xprobs`), cached next to each file
    (`<name>.xprobs.npy`, computed once: X never changes)."""
    parts = []
    for p in paths:
        d = dict(np.load(p))
        if x_probs:
            cache = Path(p).with_suffix(".xprobs.npy")
            if not cache.exists():
                np.save(cache, window_probs(load_policy(X_CKPT), d).astype(np.float32))
            d["xprobs"] = np.load(cache)
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


@torch.no_grad()
def window_probs(pol: BCPolicy, w: dict[str, np.ndarray], batch: int = 256) -> np.ndarray:
    """Legal-option softmax [n, 64] (0 on illegal options)."""
    out = []
    for i in range(0, len(w["meta_kind"]), batch):
        lg = pol.logits({k: w[k][i : i + batch] for k in OBS_KEYS}).astype(np.float64)
        m = w["action_mask"][i : i + batch].astype(bool)
        lg[~m] = -np.inf
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg)
        out.append(p / p.sum(1, keepdims=True))
    return np.concatenate(out)


def score(pol: BCPolicy, ref: BCPolicy | None, w: dict[str, np.ndarray]) -> dict:
    """Behaviour on fixed windows vs the reference (X; None = the cached `xprobs` column): target mass and argmax
    rates per kind, KL and argmax agreement on `other` windows."""
    p = window_probs(pol, w)
    q = w["xprobs"].astype(np.float64) if ref is None else window_probs(ref, w)
    kinds, kcol = w["meta_kind"], w["actions"][:, :, 0]
    out = {}
    for name in ("attack", "interrupt", "interrupt_t1", "battle"):
        kind = name.removesuffix("_t1")
        sel = (kinds == kind) & ((w["global"][:, 2] == 1) if name.endswith("_t1") else True)
        if not sel.any():
            out[name] = {"n": 0}
            continue
        tk = target_kind(kind)
        tgt = (kcol[sel] == tk) & w["action_mask"][sel]
        if kind == "battle":  # TO_BATTLE vs TO_END only, where X leaves the main phase
            leave = np.isin(kcol[sel], (int(Kind.TO_BATTLE), int(Kind.TO_END))) & w["action_mask"][sel]
            am_q = q[sel].argmax(1)
            keep = leave[np.arange(len(am_q)), am_q]
            sel = np.flatnonzero(sel)[keep]
            tgt, leave = tgt[keep], leave[keep]
            pp, qq = p[sel] * leave, q[sel] * leave
            p_sel, q_sel = pp / pp.sum(1, keepdims=True), qq / qq.sum(1, keepdims=True)
        else:
            p_sel, q_sel = p[sel], q[sel]
        am_p, am_q = p_sel.argmax(1), q_sel.argmax(1)
        out[name] = {"n": int(len(am_p)),
                     "mass": round(float((p_sel * tgt).sum(1).mean()), 4),
                     "mass_x": round(float((q_sel * tgt).sum(1).mean()), 4),
                     "argmax_rate": round(float(tgt[np.arange(len(am_p)), am_p].mean()), 4),
                     "argmax_rate_x": round(float(tgt[np.arange(len(am_q)), am_q].mean()), 4)}  # fmt: skip
    sel = kinds == "other"
    ps, qs = np.clip(p[sel], 1e-12, 1), np.clip(q[sel], 1e-12, 1)
    m = w["action_mask"][sel].astype(bool)
    kl = np.where(m, q[sel] * (np.log(qs) - np.log(ps)), 0).sum(1)
    out["other"] = {"n": int(sel.sum()), "kl_x_to_edit": round(float(kl.mean()), 4),
                    "argmax_agree": round(float((p[sel].argmax(1) == q[sel].argmax(1)).mean()), 4)}  # fmt: skip
    return out


def changed_params(ckpt: str | Path) -> dict:
    a, b = load_state(X_CKPT), load_state(Path(ckpt))
    if a.keys() != b.keys():
        raise ValueError("different tensor names from X")
    total = changed = 0
    tensors = []
    for k in a:
        d = (a[k].float() != b[k].float()).sum().item()
        total += a[k].numel()
        changed += d
        if d:
            tensors.append(k)
    return {"changed": int(changed), "total": int(total), "fraction": round(changed / total, 5), "tensors": tensors}
