"""Window sets: replayable references (start config + choice prefix), never raw observations.

Sources:
- `labels`: the label-job records (`data/cache/oppfilt_in/*.jsonl`, as `opp_probe.py`);
- `starts` / `starts-eval`: line starts scanned with a reference policy (training / held-out files);
- `streams`: recorded EDOPro client streams rebuilt as engine puzzle starts (`play.snapshot`).

A `Window` identifies one decision. `materialize` replays its prefix to attach the public metadata used by the
filters (chain length, phase, turn owner, legal option kinds, activation card codes).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from decklight.demos.dataset import record_config
from decklight.engine.actions import Kind
from decklight.engine.env import PyEnv

ROOT = Path(__file__).resolve().parents[3]
SOURCE_KINDS = ("labels", "starts", "starts-eval", "streams")


def _audit(name: str):
    """Import a `scripts/audit` module without making `scripts` a package."""
    import importlib

    for p in (ROOT / "scripts", ROOT / "scripts" / "audit"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module(name)


@dataclass
class Window:
    """One decision of our seat. `start` is the replayable start dict (`config` + `prefix` + `script`)."""

    wid: str
    group: str
    source: str
    line_id: str
    k: int
    start: dict
    prefix: list[int]
    our_seat: int
    auto_single: bool = True
    chain: int = -1
    phase: int = -1
    turn_owner: int = -1  # 1 = our turn, 0 = opponent's
    legal_kinds: list[int] = field(default_factory=list)
    activation_cards: list[int] = field(default_factory=list)
    activation_names: list[str] = field(default_factory=list)
    supplied: dict = field(default_factory=dict)

    @property
    def has_activation(self) -> bool:
        return int(Kind.ACTIVATE) in self.legal_kinds

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Window:
        return cls(**d)


@dataclass
class WindowFrame:
    window: Window
    obs: dict


def config_to_dict(c) -> dict:
    """DuelConfig -> the `config` dict stored in demo/line-start records (record_config is its inverse)."""
    return {
        "deck0_main": list(c.deck0.main),
        "deck0_extra": list(c.deck0.extra),
        "deck1_main": list(c.deck1.main),
        "deck1_extra": list(c.deck1.extra),
        "seed": int(c.seed),
        "lp": int(c.lp),
        "hand": int(c.hand),
        "draw": int(c.draw),
        "setup": [list(p) for p in c.setup],
        "shuffle": bool(c.shuffle),
    }


def _act(pol, obs) -> int:
    return int(pol({k: np.asarray(v)[None] for k, v in obs.items()}, np.arange(1), slots=np.zeros(1, np.int64))[0])


def _line_turn(engine, st: dict) -> int:
    env = PyEnv(engine, record_config(st), st.get("auto_single", True))
    env.reset()
    for c in st["prefix"]:
        env.step(int(c))
    return int(env.duel.turn)


def window_obs(engine, window: Window) -> dict:
    """Replay a window's prefix from its start config; returns the observation at the decision."""
    env = PyEnv(engine, record_config(window.start), window.auto_single)
    obs = env.reset()
    for c in window.prefix:
        if obs is None or not obs["action_mask"][int(c)]:
            raise RuntimeError(f"{window.wid}: recorded prefix does not replay")
        obs = env.step(int(c))
    if obs is None:
        raise RuntimeError(f"{window.wid}: game ended before the window")
    return obs


def _metadata(obs: dict) -> dict:
    legal = np.flatnonzero(np.asarray(obs["action_mask"]).astype(bool))
    actions = np.asarray(obs["actions"])
    kinds = sorted({int(actions[int(i)][0]) for i in legal})
    cards = sorted(
        {int(actions[int(i)][2]) for i in legal if int(actions[int(i)][0]) == int(Kind.ACTIVATE) and actions[int(i)][2]}
    )
    return {
        "chain": int(np.asarray(obs["chain_mask"]).sum()),
        "phase": int(np.asarray(obs["global"])[3]),
        "turn_owner": int(np.asarray(obs["global"])[5]),
        "legal_kinds": kinds,
        "activation_cards": cards,
    }


_NAMES: dict[int, str] | None = None


def _card_names() -> dict[int, str]:
    global _NAMES
    if _NAMES is None:
        try:
            from decklight.demos.names import CardNames

            _NAMES = CardNames().names
        except Exception:  # noqa: BLE001 - names are cosmetic; codes remain
            _NAMES = {}
    return _NAMES


def materialize(engine, windows: list[Window], *, progress=None) -> list[Window]:
    """Replay each window (when metadata is missing) and fill chain/phase/turn/kinds/cards in place."""
    out = []
    names = None
    for i, w in enumerate(windows):
        if w.chain < 0 or w.phase < 0 or not w.legal_kinds:
            obs = window_obs(engine, w)
            for k, v in _metadata(obs).items():
                setattr(w, k, v)
        if not w.activation_names and w.activation_cards:
            names = names if names is not None else _card_names()
            w.activation_names = [names.get(c, str(c)) for c in w.activation_cards]
        out.append(w)
        if progress is not None and (i + 1) % 200 == 0:
            progress(f"materialized {i + 1}/{len(windows)} windows")
    return out


def bucket_chain(chain: int) -> str:
    return "0" if chain <= 0 else "1" if chain == 1 else "2+"


def filter_windows(
    windows: list[Window],
    *,
    turn_owner: int | None = None,
    chain: int | None = None,
    min_chain: int | None = None,
    max_chain: int | None = None,
    phase: int | list[int] | None = None,
    has_activation: bool | None = None,
) -> list[Window]:
    out = []
    phases = None if phase is None else {int(x) for x in (phase if isinstance(phase, list) else [phase])}
    for w in windows:
        if turn_owner is not None and w.turn_owner != turn_owner:
            continue
        if chain is not None and w.chain != chain:
            continue
        if min_chain is not None and w.chain < min_chain:
            continue
        if max_chain is not None and w.chain > max_chain:
            continue
        if phases is not None and w.phase not in phases:
            continue
        if has_activation is not None and w.has_activation != has_activation:
            continue
        out.append(w)
    return out


def group_by_start(windows: list[Window]) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {}
    for w in windows:
        out.setdefault(w.group, []).append(w)
    return out


def save_windows(path: Path, windows: list[Window]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(w.to_dict()) + "\n" for w in windows))


def load_windows(path: Path) -> list[Window]:
    return [Window.from_dict(json.loads(x)) for x in path.read_text().splitlines() if x.strip()]


def cache_path(cache_dir: Path, kind: str, **params) -> Path:
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return cache_dir / f"{kind}-{key}.jsonl"


def _dedup_starts(path: Path) -> list[dict]:
    """Training starts, indexed as `gen_opp_windows` indexed them (`oppwin-<idx>-...`)."""
    seen, out = set(), []
    for x in path.read_text().splitlines():
        if x.strip() and x not in seen:
            seen.add(x)
            out.append(json.loads(x))
    return out


def _supplied(inf: dict) -> dict:
    arms = inf.get("arms") or {}
    out = {}
    for arm, vals in arms.items():
        if not isinstance(vals, dict):
            continue
        for key, col in (("won", "won"), ("opp_field_end_turn", "field"), ("completed", "completed")):
            if vals.get(key) is not None:
                out[f"{arm}_{col}"] = float(vals[key])
    return out


def build_label_windows(win_dir: Path, starts_path: Path, *, max_starts: int = 0) -> list[Window]:
    """Every labelled window of the label job, replayed from the matching line start (targets need its script)."""
    starts = _dedup_starts(starts_path)
    recs = [
        json.loads(x)
        for f in sorted(win_dir.glob("*.jsonl"))
        for x in f.read_text().splitlines()
        if x.strip()
    ]
    out: list[Window] = []
    seen_starts = 0
    for r in recs:
        try:
            idx = int(str(r["line_id"]).split("/oppwin-")[1].split("-")[0])
        except (IndexError, ValueError):
            idx = -1
        st = starts[idx] if 0 <= idx < len(starts) else None
        if st is not None and (st.get("line_id") != r.get("start_line_id") or st.get("config") != r.get("config")):
            st = None
        if st is None:  # cannot recover the script: keep the window but targets will be skipped
            st = {
                "config": r["config"],
                "prefix": [],
                "script": [],
                "our_seat": int(r["our_seat"]),
                "auto_single": bool(r.get("auto_single", True)),
                "line_id": r.get("start_line_id", r["line_id"]),
            }
        seen_starts += 1
        group = f"label:{idx if idx >= 0 else r['line_id']}"
        choices = [int(c) for c in r["choices"]]
        for ks, inf in sorted(r["window_info"].items(), key=lambda t: int(t[0])):
            k = int(ks)
            if k > len(choices):
                continue
            out.append(
                Window(
                    wid=f"label:{r['line_id']}:{k}",
                    group=group,
                    source="labels",
                    line_id=r["line_id"],
                    k=k,
                    start=st,
                    prefix=choices[:k],
                    our_seat=int(r["our_seat"]),
                    auto_single=bool(r.get("auto_single", True)),
                    supplied=_supplied(inf),
                )
            )
        if max_starts and seen_starts >= max_starts:
            break
    return out


def _scan(engine, st: dict, pol) -> list[tuple[list[int], dict]]:
    """(prefix, obs) for each decision of our seat on the start's first turn with a legal activation.

    Mirrors `scripts/audit/window_gate.windows`: the opponent replays its script, our seat is played by `pol`.
    """
    from decklight.rl.lines import LineTracker

    env = PyEnv(engine, record_config(st), st.get("auto_single", True))
    obs = env.reset()
    for c in st["prefix"]:
        obs = env.step(int(c))
    me = int(st["our_seat"])
    t0 = int(env.duel.turn)
    tr = LineTracker(1)
    tr.begin(0, {"script": [tuple(x) for x in st.get("script", [])]})
    choices = [int(c) for c in st["prefix"]]
    out = []
    for _ in range(1500):
        if obs is None or int(env.duel.turn) > t0:
            break
        if env.player == me:
            legal = np.flatnonzero(np.asarray(obs["action_mask"]).astype(bool))
            if any(int(obs["actions"][int(i)][0]) == int(Kind.ACTIVATE) for i in legal):
                out.append((list(choices), {k: np.array(v) for k, v in obs.items()}))
            a = _act(pol, obs)
        else:
            a = tr.next(0, obs["actions"], obs["action_mask"])
            if a is None:
                a = _act(pol, obs)
        choices.append(int(a))
        obs = env.step(int(a))
    return out


def _scan_to_windows(engine, st: dict, pol, *, source: str, group: str, line_id: str) -> list[Window]:
    out = []
    for prefix, obs in _scan(engine, st, pol):
        k = len(prefix)
        meta = _metadata(obs)
        out.append(
            Window(
                wid=f"{group}:{k}",
                group=group,
                source=source,
                line_id=line_id,
                k=k,
                start=st,
                prefix=prefix,
                our_seat=int(st["our_seat"]),
                auto_single=bool(st.get("auto_single", True)),
                **meta,
            )
        )
    return out


def build_start_windows(engine, starts_path: Path, pol, *, source: str, max_starts: int = 0) -> list[Window]:
    """Scan line starts with `pol`; one window per activation-legal decision of our seat on the line's turn."""
    starts = [json.loads(x) for x in starts_path.read_text().splitlines() if x.strip()]
    if max_starts:
        starts = starts[:max_starts]
    out: list[Window] = []
    for si, st in enumerate(starts):
        out.extend(
            _scan_to_windows(
                engine, st, pol, source=source, group=f"{source}:{si}:{st['line_id']}", line_id=st["line_id"]
            )
        )
    return out


def build_stream_windows(engine, root: Path, pol, *, max_starts: int = 0) -> list[Window]:
    """Recorded client streams -> puzzle starts -> windows (hidden opponent cards sampled, `play.snapshot`)."""
    from decklight.play.snapshot import SkipPosition, build_config, extract_snapshots, read_stream

    out: list[Window] = []
    n = 0
    pattern = "g*/*.stream.jsonl.gz"
    for path in sorted(root.glob(pattern)):
        try:
            header, frames = read_stream(path)
            snaps = extract_snapshots(header, frames, engine.cards, stream=str(path))
        except Exception as e:  # noqa: BLE001 - a malformed stream is a counted skip, never kills the set
            print(f"diag windows: skip {path}: {type(e).__name__}: {e}", flush=True)
            continue
        for snap in snaps:
            if max_starts and n >= max_starts:
                return out
            try:
                built = build_config(snap, header, engine.cards, sample=0, seed=0)
            except SkipPosition as e:
                print(f"diag windows: skip {path} t{snap.turn}: {e}", flush=True)
                continue
            st = {
                "config": config_to_dict(built.config),
                "prefix": [],
                "script": [],
                "our_seat": 1,
                "auto_single": True,
                "line_id": f"{path}@t{snap.turn}",
                "kind": "stream",
            }
            out.extend(
                _scan_to_windows(
                    engine, st, pol, source="streams", group=f"stream:{path}:{snap.turn}", line_id=st["line_id"]
                )
            )
            n += 1
    return out
