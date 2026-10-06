"""A trained BC PolicyNet as a gauntlet policy (argmax over legal options, or sampling with a temperature).

pol = BCPolicy("runs/<run>/bc/best.pt", "data/cards/emb/qwen3-0.6b-d512.npz")   # args read from metrics.jsonl
actions = pol(buf, rows)                                                         # VecEnv buffers, like policies.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ygo_zero.eval.loopguard import LoopGuard
from ygo_zero.model.bc_data import KNOWN_KEYS, OBS_KEYS
from ygo_zero.model.policy import PolicyNet


def _run_args(ckpt: Path) -> dict:
    m = ckpt.parent / "metrics.jsonl"
    if m.is_file():
        for raw in m.read_text().splitlines():
            rec = json.loads(raw)
            if "args" in rec:
                return rec["args"]
    return {}


class BCPolicy:
    def __init__(self, ckpt: str | Path, emb: str | Path, temperature: float = 0.0, seed: int = 0,
                 threads: int = 2, batch: int = 256, device: str = "cpu", loop_guard: bool = True) -> None:  # fmt: skip
        ckpt = Path(ckpt)
        a = _run_args(ckpt)
        torch.set_num_threads(threads)
        self.hist = int(a.get("hist", 0) or 0)
        self.known = bool(a.get("known", False))
        self.model = PolicyNet(dict(np.load(emb)), d=a.get("d", 256), layers=a.get("layers", 4),
                               heads=a.get("heads", 4), hist=self.hist, known=self.known)  # fmt: skip
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model", state)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.temperature, self.batch = temperature, batch
        self.rng = np.random.default_rng(seed)
        self.name = f"bc:{ckpt.parent.parent.name}"
        # exact-repetition loops (SELECT/UNSELECT, SPSUMMON/cancel) are broken per env slot: eval/loopguard.py
        self.guard = LoopGuard() if loop_guard else None

    @torch.no_grad()
    def logits(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if self.known and not all(k in obs for k in KNOWN_KEYS):
            raise KeyError("this checkpoint was trained with --known: the obs needs card_known/known_code")
        if self.hist > 0:
            if "hist_long" not in obs:
                raise KeyError(
                    "this checkpoint was trained with --hist: the obs needs hist_long/hist_long_mask"
                )
            t = {
                k: torch.from_numpy(np.ascontiguousarray(obs[k])).to(self.device)
                for k in OBS_KEYS
                if k not in ("history", "history_mask")
            }
            t["history"] = torch.from_numpy(np.ascontiguousarray(obs["hist_long"][:, : self.hist])).to(
                self.device
            )
            t["history_mask"] = torch.from_numpy(
                np.ascontiguousarray(obs["hist_long_mask"][:, : self.hist])
            ).to(self.device)
            if self.known:
                for k in KNOWN_KEYS:
                    t[k] = torch.from_numpy(np.ascontiguousarray(obs[k])).to(self.device)
        else:
            keys = tuple(OBS_KEYS) + (tuple(KNOWN_KEYS) if self.known else ())
            t = {k: torch.from_numpy(np.ascontiguousarray(obs[k])).to(self.device) for k in keys}
        return self.model(t).float().cpu().numpy()

    def __call__(
        self, buf: dict[str, np.ndarray], rows: np.ndarray, slots: np.ndarray | None = None
    ) -> np.ndarray:
        """Actions for buf rows `rows`. `slots` names the env behind each row for the loop guard (default: the row
        index, right for VecEnv buffers whose rows are fixed env slots)."""
        keys = tuple(OBS_KEYS) + (("hist_long", "hist_long_mask") if self.hist > 0 else ())
        keys += tuple(KNOWN_KEYS) if self.known else ()
        slots = rows if slots is None else slots
        out = np.zeros(len(rows), np.int32)
        for i in range(0, len(rows), self.batch):
            r = rows[i : i + self.batch]
            lg = self.logits({k: buf[k][r] for k in keys})
            if self.guard is None:
                if self.temperature <= 0:
                    out[i : i + len(r)] = lg.argmax(1)
                else:
                    z = lg / self.temperature
                    z = np.exp(z - z.max(1, keepdims=True))
                    z /= z.sum(1, keepdims=True)
                    out[i : i + len(r)] = [self.rng.choice(len(p), p=p) for p in z]
                continue
            for j, row in enumerate(r):
                obs = {k: buf[k][row] for k in ("cards", "card_mask", "global", "chain", "chain_mask", "actions",
                                                 "action_mask")}  # fmt: skip
                z = lg[j].astype(np.float64)[: obs["action_mask"].shape[0]]
                z[~obs["action_mask"].astype(bool)] = -np.inf
                slot = int(slots[i + j])
                z, key = self.guard.mask(slot, obs, z)
                if self.temperature <= 0:
                    a = int(np.argmax(z))
                else:
                    p = np.exp((z - z.max()) / self.temperature)
                    a = int(self.rng.choice(len(p), p=p / p.sum()))
                self.guard.note(slot, key, a)
                out[i + j] = a
        return out
