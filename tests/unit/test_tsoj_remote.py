"""decklight.rl.remote: wire format, episode files, ExtFeed intake, RemoteAgent rewards (tsoj PPO step 2)."""

from __future__ import annotations

import socket
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from decklight.model.bc_data import OBS_KEYS
from decklight.rl.ppo import SlotStore
from decklight.rl.remote import STORE_KEYS, EpisodeLog, ExtFeed, read_episode, recv_msg, send_msg

SHAPES = {"cards": ((160, 21), np.int32), "card_mask": ((160,), bool), "global": ((24,), np.int32),
          "chain": ((16, 5), np.int32), "chain_mask": ((16,), bool), "actions": ((64, 10), np.int32),
          "action_mask": ((64,), bool), "history": ((32, 6), np.int32), "history_mask": ((32,), bool)}


def obs(seed: int = 0, n_legal: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    o = {k: rng.integers(0, 5, size=s).astype(t) for k, (s, t) in SHAPES.items()}
    o["action_mask"] = np.zeros(64, bool)
    o["action_mask"][:n_legal] = True
    o["hist_long"] = np.zeros((128, 6), np.int32)  # extra client keys are dropped on write
    return o


def write_ep(d: Path, n: int, version: int, reward: float) -> Path:
    log = EpisodeLog()
    for i in range(n):
        log.add(obs(i), i % 3, -1.0, 0.1 * i, version)
    return log.write(d, reward, {"deck": "x", "seat": 0, "n": n})


def ref(tob):
    z = torch.zeros(tob["action_mask"].shape, dtype=torch.float32)
    return torch.where(tob["action_mask"], z, torch.full_like(z, float("-inf")))


def store(cap: int = 1000) -> SlotStore:
    return SlotStore(cap, {k: np.zeros((1, *s), t) for k, (s, t) in SHAPES.items()})


def test_store_keys_match_bc_obs_keys():
    assert STORE_KEYS == tuple(OBS_KEYS)


def test_wire_roundtrip():
    a, b = socket.socketpair()
    msg = {"x": np.arange(5), "s": "é"}
    send_msg(a, msg)
    got = recv_msg(b)
    assert got["s"] == "é" and np.array_equal(got["x"], msg["x"])


def test_episode_roundtrip_keeps_store_keys_only(tmp_path):
    p = write_ep(tmp_path, 4, 7, 1.0)
    e = read_episode(p)
    assert set(e["obs"]) == set(STORE_KEYS)
    assert e["obs"]["cards"].shape == (4, 160, 21) and e["reward"] == 1.0
    assert e["version"].tolist() == [7] * 4 and e["action"].tolist() == [0, 1, 2, 0]
    with pytest.raises(ValueError):
        EpisodeLog().write(tmp_path, 0.0, {})


def test_extfeed_quota_stale_and_used(tmp_path):
    write_ep(tmp_path, 5, 1, 1.0)   # stale for min_version 2
    write_ep(tmp_path, 4, 3, -1.0)
    write_ep(tmp_path, 6, 3, 1.0)
    st_ = store()
    fin, st = ExtFeed(tmp_path).take(st_, quota=3, min_version=2, ref=ref, device=torch.device("cpu"))
    assert st["ext_stale"] == 1 and st["ext_files"] == 1 and st["ext_samples"] == 4  # quota reached after 1 file
    (slots, reward), = fin
    assert reward == -1.0 and len(slots) == 4
    assert np.allclose(st_.logp[slots], -1.0) and np.allclose(st_.value[slots], [0.0, 0.1, 0.2, 0.3])
    lp = st_.ref_lp[slots].astype(np.float32)
    assert np.allclose(lp[:, :3], np.log(1 / 3), atol=1e-2) and np.all(lp[:, 3:] == 0)
    assert len(list(tmp_path.glob("ep-*.npz"))) == 1 and len(list((tmp_path / "used").glob("ep-*.npz"))) == 2


def test_extfeed_refuses_a_shape_mismatch(tmp_path):
    write_ep(tmp_path, 2, 0, 1.0)
    bad = SlotStore(10, {**{k: np.zeros((1, *s), t) for k, (s, t) in SHAPES.items()},
                         "cards": np.zeros((1, 150, 21), np.int32)})
    with pytest.raises(RuntimeError, match="cards"):
        ExtFeed(tmp_path).take(bad, 100, 0, ref, torch.device("cpu"))


def _fake_server(path: str, action: int, stop: threading.Event, ready: threading.Event) -> None:
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    srv.settimeout(5)
    ready.set()  # the socket file exists from bind(); connecting before listen() is refused
    conn, _ = srv.accept()
    try:
        while not stop.is_set():
            recv_msg(conn)
            send_msg(conn, {"action": action, "logp": -0.5, "value": 0.25, "version": 9})
    except (ConnectionError, OSError):
        pass


@pytest.mark.parametrize("winner,seat,want", [((1, 1), 1, 1.0), ((0, 1), 1, -1.0), (None, 0, 0.0)])
def test_remote_agent_episode_reward(tmp_path, winner, seat, want):
    from decklight.play.agents import RemoteAgent

    d = tempfile.mkdtemp(dir="/tmp")  # AF_UNIX paths are limited to ~108 bytes
    path = f"{d}/s.sock"
    stop, ready = threading.Event(), threading.Event()
    threading.Thread(target=_fake_server, args=(path, 1, stop, ready), daemon=True).start()
    assert ready.wait(5)
    ag = RemoteAgent(sock=path, episode_dir=tmp_path)
    view = SimpleNamespace(obs=lambda: obs(0), opts=[0, 1, 2], seat=seat)
    assert ag.act(view) == 1 and ag.act(view) == 1
    ag.finalize(SimpleNamespace(winner=winner, turns=3), deck="sky")
    ag.finalize(SimpleNamespace(winner=winner, turns=3), deck="sky")  # idempotent
    stop.set()
    eps = list(tmp_path.glob("ep-*.npz"))
    assert len(eps) == 1
    e = read_episode(eps[0])
    assert e["reward"] == want and e["meta"]["seat"] == seat and e["meta"]["deck"] == "sky"
    assert e["action"].tolist() == [1, 1] and e["version"].tolist() == [9, 9]


def test_remote_agent_refuses_an_illegal_server_pick(tmp_path):
    from decklight.play.agents import RemoteAgent

    d = tempfile.mkdtemp(dir="/tmp")
    path = f"{d}/s.sock"
    stop, ready = threading.Event(), threading.Event()
    threading.Thread(target=_fake_server, args=(path, 5, stop, ready), daemon=True).start()
    assert ready.wait(5)
    ag = RemoteAgent(sock=path, episode_dir=tmp_path)
    with pytest.raises(RuntimeError, match="illegal"):
        ag.act(SimpleNamespace(obs=lambda: obs(0), opts=[0, 1, 2], seat=0))
    stop.set()


def test_remote_agent_refuses_relative_paths(tmp_path):
    from decklight.play.agents import RemoteAgent

    # the arbiter starts the client in scripts/tsoj/client: a relative episode dir silently lands there
    with pytest.raises(ValueError, match="absolute"):
        RemoteAgent(sock="/tmp/x.sock", episode_dir="runs/x")
