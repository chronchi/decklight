"""Auxiliary evaluator heads on the frozen value trunk (engine plan N3).

Heads that predict, from the value trunk's state embedding, what happens by the
actor's next own-turn start. Analysis use now (engine reports/viewer); a training
signal later (N6, not here). The value head is untouched.

Targets (per kept decision, actor's view; obs `global` = `engine/obs.py:global_array`):
1. `d_lp` (2): LP change me/opp from now to the actor's next own-turn start, / 8000.
2. `d_cards` (14): count change per location me/opp over the same span.
3. `ends` (1, binary): the game ends before the actor's next own-turn start.
4. `turns_left` (1): log1p(number of duel turns from now to the game's end).
`win` is NOT an aux target (the value head already is).

"Next own-turn start" = the actor's first decision in a later duel turn where the
actor is turn player. Targets 1-2 are masked (not trained/evaluated) when that
never happens (the game ended first; winner-3 truncated games are dropped before
this module ever sees them). The target spec lives in `AUX_SPEC` so targets can
change later without touching the trainer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# obs `global` columns (engine/obs.py:global_array): [lp_me, lp_opp, turn, phase, first,
# my_turn, counts me x 7 locs, counts opp x 7 locs, chain, msg, trunc, ver]. Locations
# are query.QUERY_LOCATIONS = deck, hand, mzone, szone, grave, removed, extra.
LP_ME, LP_OPP, TURN, MY_TURN = 0, 1, 2, 5
COUNTS_ME = (6, 13)  # slice [6:13]
COUNTS_OPP = (13, 20)  # slice [13:20]
N_LOCS = 7
LP_NORM = 8000.0

AUX_SPEC: dict[str, dict] = {
    # dim: head width; loss: training loss; mask: which rows train/evaluate
    # ("has_next" = the actor gets another own turn, else targets 1-2 are void);
    # baseline: the no-learning reference R2/Brier is measured against.
    "d_lp": {"dim": 2, "loss": "mse", "mask": "has_next", "baseline": "zero",
             "desc": "LP change me/opp to the actor's next own-turn start, / 8000"},
    "d_cards": {"dim": 14, "loss": "mse", "mask": "has_next", "baseline": "zero",
                "desc": "count change per location me/opp over the same span"},
    "ends": {"dim": 1, "loss": "bce", "mask": None, "baseline": "base_rate",
             "desc": "the game ends before the actor's next own-turn start"},
    "turns_left": {"dim": 1, "loss": "mse", "mask": None, "baseline": "train_mean",
                   "desc": "log1p(duel turns from now to the game's end)"},
}


def _absolute_views(g: np.ndarray, actors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Actor-relative globals -> absolute seat views: (lp0, lp1, cnt0, cnt1)."""
    a0 = (actors == 0)
    lp0 = np.where(a0, g[:, LP_ME], g[:, LP_OPP]).astype(np.float64)
    lp1 = np.where(a0, g[:, LP_OPP], g[:, LP_ME]).astype(np.float64)
    cnt0 = np.where(a0[:, None], g[:, COUNTS_ME[0]:COUNTS_ME[1]], g[:, COUNTS_OPP[0]:COUNTS_OPP[1]])
    cnt1 = np.where(a0[:, None], g[:, COUNTS_OPP[0]:COUNTS_OPP[1]], g[:, COUNTS_ME[0]:COUNTS_ME[1]])
    return lp0, lp1, cnt0.astype(np.float64), cnt1.astype(np.float64)


def _own_turn_starts(turns: np.ndarray, actors: np.ndarray, turn_player: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Per seat: [(duel turn, first decision index)] of that seat's own-turn starts in scan order."""
    starts: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    seen: dict[int, set[int]] = {0: set(), 1: set()}
    for j in range(len(actors)):
        s = int(actors[j])
        if turn_player[j] == s and actors[j] == s and int(turns[j]) not in seen[s]:
            seen[s].add(int(turns[j]))
            starts[s].append((int(turns[j]), j))
    return starts


def extract_targets(globals: np.ndarray, actors: np.ndarray, winner: int, end_turn: int) -> dict[str, np.ndarray]:
    """Targets + masks for one replayed game (pure numpy).

    `globals` [N, 24]: per-decision `global` vectors, each in its own actor's view.
    `actors` [N]: acting seat per decision. `winner` in (0, 1) (winner-3 games are
    dropped by the caller). `end_turn`: the duel turn the game ended on.
    Returns `d_lp` [N, 2], `d_cards` [N, 14], `ends` [N], `turns_left` [N],
    `mask` [N] bool (has_next: True = the actor gets another own turn; targets 1-2
    train/evaluate only there).
    """
    g = np.asarray(globals, dtype=np.int64)
    a = np.asarray(actors, dtype=np.int64)
    assert winner in (0, 1), winner
    n = len(a)
    assert g.shape == (n, 24), g.shape
    turns = g[:, TURN]
    turn_player = np.where(g[:, MY_TURN] == 1, a, 1 - a)
    lp0, lp1, cnt0, cnt1 = _absolute_views(g, a)
    starts = _own_turn_starts(turns, a, turn_player)
    start_turns = {s: np.array([t for t, _ in starts[s]], dtype=np.int64) for s in (0, 1)}
    start_idx = {s: np.array([j for _, j in starts[s]], dtype=np.int64) for s in (0, 1)}

    d_lp = np.zeros((n, 2), np.float32)
    d_cards = np.zeros((n, 14), np.float32)
    mask = np.zeros(n, bool)
    for k in range(n):
        s = int(a[k])
        pos = int(np.searchsorted(start_turns[s], int(turns[k]) + 1, side="left"))
        if pos >= len(start_idx[s]):
            continue
        j = int(start_idx[s][pos])
        mask[k] = True
        me_now, op_now = (lp0[k], lp1[k]) if s == 0 else (lp1[k], lp0[k])
        me_fut, op_fut = (lp0[j], lp1[j]) if s == 0 else (lp1[j], lp0[j])
        d_lp[k] = [(me_fut - me_now) / LP_NORM, (op_fut - op_now) / LP_NORM]
        me_now_c, op_now_c = (cnt0[k], cnt1[k]) if s == 0 else (cnt1[k], cnt0[k])
        me_fut_c, op_fut_c = (cnt0[j], cnt1[j]) if s == 0 else (cnt1[j], cnt0[j])
        d_cards[k] = np.concatenate([me_fut_c - me_now_c, op_fut_c - op_now_c]).astype(np.float32)
    ends = (~mask).astype(np.float32)
    turns_left = np.log1p(np.clip(np.int64(end_turn) - turns, 0, None)).astype(np.float32)
    return {"d_lp": d_lp, "d_cards": d_cards, "ends": ends, "turns_left": turns_left, "mask": mask}


class AuxHeads(nn.Module):
    """Shared neck (Linear -> GELU, width 256) + one linear head per target group."""

    NECK = 256

    def __init__(self, emb_dim: int, neck: int = NECK) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.neck = nn.Sequential(nn.Linear(self.emb_dim, neck), nn.GELU())
        self.lp = nn.Linear(neck, 2)
        self.cards = nn.Linear(neck, 14)
        self.ends = nn.Linear(neck, 1)
        self.turns = nn.Linear(neck, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.neck(x)
        return {"d_lp": self.lp(h), "d_cards": self.cards(h),
                "ends": self.ends(h).squeeze(-1), "turns_left": self.turns(h).squeeze(-1)}


def aux_losses(pred: dict[str, torch.Tensor], tgt: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-target losses: MSE for d_lp / d_cards / turns_left (first two masked to
    has_next rows), BCE-with-logits for ends. Returns {name: scalar} plus "total"."""
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    lp = (((pred["d_lp"] - tgt["d_lp"]) ** 2).mean(-1) * m).sum() / denom
    cards = (((pred["d_cards"] - tgt["d_cards"]) ** 2).mean(-1) * m).sum() / denom
    turns = ((pred["turns_left"] - tgt["turns_left"]) ** 2).mean()
    ends = nn.functional.binary_cross_entropy_with_logits(pred["ends"], tgt["ends"])
    return {"d_lp": lp, "d_cards": cards, "ends": ends, "turns_left": turns,
            "total": lp + cards + ends + turns}
