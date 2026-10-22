"""The product as `scripts/search_match.py` plays it, shared with the EDOPro host (EDOPro play D3).

`RouteWindowSearch`, `tag_game` and `seat_of` moved verbatim from `scripts/search_match.py`
(D3: pure move, search_match imports them back so its behaviour stays bit-identical).
`product_search_kwargs` + `build_search` + `load_policy`/`load_value` build the product
(P + value_p2, honest k4 s4 h1, det-tries 32, det-fallback skip, conf-skip 0.9, honest override `ho`)
from the product flags for both search_match-style jobs and the EDOPro host driver.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from decklight.search.window import WindowSearch

ROOT = Path(__file__).resolve().parents[3]

# The product (scripts/product.sh): honest k4 s4 h1, det-tries 32, skip, conf-skip 0.9, ho (search/honest.py
# HonestRule defaults = search_match --ho 1; adopted 2026-10-21, NOTEBOOK "Full-P + ho result").
PRODUCT_K = 4
PRODUCT_SAMPLES = 4
PRODUCT_HORIZON = 1
PRODUCT_DET_TRIES = 32
PRODUCT_DET_FALLBACK = "skip"
PRODUCT_CONF_SKIP = 0.9
PRODUCT_ROLLOUT_TEMP = 1.0
PRODUCT_EMB = ROOT / "data" / "cards" / "emb" / "qwen3-0.6b-d512.npz"  # same file search_match uses
# Custom cards: $YGO_EMB points the product at a merged card-embedding npz (default: stock).
EMB_ENV = "YGO_EMB"
PRODUCT_CKPT = ROOT / "data" / "cache" / "P" / "best.pt"
PRODUCT_VALUE = ROOT / "data" / "cache" / "value_p2" / "best.pt"


class RouteWindowSearch(WindowSearch):
    """WindowSearch that tags every rollout obs id into `route` as None (inside the search).

    search_game_gen yields outer decisions ([obs]) and rollout batches through one
    channel; the shared forward tells them apart by these tags (see tag_game): a
    rollout obs goes to the base policy, an outer obs to S's or R's policy by seat.
    """

    def __init__(self, *a, route: dict, **k) -> None:
        super().__init__(*a, **k)
        self._route = route

    def rollouts_gen(self, *a, **k):
        gen = super().rollouts_gen(*a, **k)
        try:
            req = next(gen)
            while True:
                for o in req:
                    self._route[id(o)] = None
                fb = yield req
                req = gen.send(fb)
        except StopIteration as stop:
            return stop.value


def tag_game(idx: int, s_seat: int, gen, route: dict):
    """Tag every obs id a game yields with (idx, s_seat); rollout obs are pre-tagged
    None by RouteWindowSearch (setdefault keeps that tag: marking runs first, as the
    inner generator drives rollouts_gen before this wrapper sees the request)."""
    try:
        req = next(gen)
        while True:
            for o in req:
                route.setdefault(id(o), (idx, s_seat))
            fb = yield req
            req = gen.send(fb)
    except StopIteration as stop:
        return stop.value


def seat_of(obs: dict) -> int:
    """Absolute seat of the acting player: global[4] ("I went first") is 1 iff seat 0 acts."""
    return 0 if int(obs["global"][4]) == 1 else 1


def product_search_kwargs(fast: bool = False) -> dict:
    """The product's search_match args (scripts/product.sh); fast = raw P (conf-skip 0: never search)."""
    return {
        "k": PRODUCT_K, "horizon": PRODUCT_HORIZON, "samples": PRODUCT_SAMPLES,
        "rollout_temp": PRODUCT_ROLLOUT_TEMP, "conf_skip": 0.0 if fast else PRODUCT_CONF_SKIP,
        "own_frac": 1.0, "own_turn1": False, "honest": True, "det_tries": PRODUCT_DET_TRIES,
        "det_fallback": PRODUCT_DET_FALLBACK, "margin": 0.0, "max_search_turn": 0, "max_searches": 0,
        "ho": not fast,  # True -> HonestRule() (bool here: the dict is written into game metadata)
    }


def load_policy(ckpt: str | Path, emb: str | Path, device: str = "cpu",
                batch: int = 512, threads: int = 1):
    """The product's P (a BCPolicy, like search_match builds it)."""
    from decklight.eval.bc_policy import BCPolicy

    return BCPolicy(ckpt, emb, threads=threads, device=device, batch=batch)


def load_value(value_ckpt: str | Path | None, emb: str | Path, device: str = "cpu",
               batch: int = 512, threads: int = 1):
    """The product's learned value (ValueNet.values), or None without --value."""
    if value_ckpt is None:
        return None
    from decklight.eval.value import ValueNet

    return ValueNet(value_ckpt, emb, device=device, batch=batch).values


def build_search(engine, *, seed: int, value=None, route: dict | None = None,
                 pool=None, fast: bool = False, **kw) -> WindowSearch:
    """A WindowSearch with the product's args (or `fast`: conf-skip 0 downstream).

    `route` given -> RouteWindowSearch tagging into it (the routed path); otherwise plain.
    Extra `kw` overrides product defaults (k/samples/horizon/...); unknown keys raise.
    """
    args = {"k": PRODUCT_K, "horizon": PRODUCT_HORIZON, "samples": PRODUCT_SAMPLES,
            "rollout_temp": PRODUCT_ROLLOUT_TEMP}
    for k, v in kw.items():
        if k not in args:
            raise ValueError(f"build_search: unknown search arg {k!r}")
        args[k] = v
    never = _never_logits
    if route is None:
        return WindowSearch(engine, never, seed=seed, value=value, pool=pool, **args)
    return RouteWindowSearch(engine, never, seed=seed, value=value, route=route, pool=pool, **args)


def _never_logits(buf) -> np.ndarray:
    raise AssertionError("WindowSearch.logits is unused: play.py drives rollouts via drive_many")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_product_paths(ckpt=None, value=None, emb=None) -> tuple[Path, Path | None, Path]:
    """Product ckpt/value/emb paths (defaults: scripts/product.sh), all required to exist.

    `emb` falls back to `$YGO_EMB` (a merged npz covering custom cards) before the stock file.
    """
    ckpt_p = Path(ckpt) if ckpt is not None else PRODUCT_CKPT
    value_p = Path(value) if value is not None else PRODUCT_VALUE
    emb_p = Path(emb or os.environ.get(EMB_ENV) or PRODUCT_EMB)
    missing = [str(p) for p in (ckpt_p, emb_p) if not p.is_file()]
    if value_p is not None and not value_p.is_file():
        missing.append(str(value_p))
    if missing:
        raise FileNotFoundError(f"product files missing: {missing}")
    return ckpt_p, value_p, emb_p


def make_pool(replay_threads: int):
    """Speed plan E6 pool for branch replays (bit-identical), or None for the sequential path."""
    if replay_threads < 1:
        raise ValueError(f"--replay-threads must be >= 1, got {replay_threads}")
    return None if replay_threads == 1 else ThreadPoolExecutor(max_workers=replay_threads)
