"""PPO against external opponents (NOTEBOOK 2026-11-07, tsoj PPO step 2): a policy server, its client, episode files.

Games against an opponent we cannot run inside the VecEnv (tsoj's `bot alphabeta` in its own arbiter) are played by
network clients (`edopro_join.py --agent remote`). Each client sends its observation to one `PolicyServer` per box
(unix socket, batched forward on the box's GPU or CPU), which samples an action from the current learner weights
exactly like `rollout.Collector` (temperature 1 over legal options) and returns (action, logp, value, version). The
client keeps the game's decisions and writes one episode file at the end. `train_rl.py --ext-dir` reads these files
(`ExtFeed`) into the PPO slot store as finished learner trajectories, and publishes its weights after every update
(`publish_weights`); the server reloads them.

Wire format: 8-byte little-endian length + pickle (both ends are our own processes on one box).
Episode file (`<dir>/ep-*.npz`, written atomically): obs arrays [T, ...] under `obs.<key>`, `action` [T] int64,
`logp` [T] float32, `value` [T] float32, `version` [T] int64, `reward` float, `meta` (JSON string).
The client side (PolicyClient, EpisodeLog) imports no torch.
"""

from __future__ import annotations

import json
import os
import pickle
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

_LEN = struct.Struct("<Q")
# = decklight.model.bc_data.OBS_KEYS (the store layout of a hist=0, known=False net), kept here so clients need no torch
STORE_KEYS = ("cards", "card_mask", "global", "chain", "chain_mask", "actions", "action_mask", "history",
              "history_mask")
OBS_HISTORY_STORE_KEYS = ("obs_history", "obs_history_mask")  # = bc_data.OBS_HISTORY_KEYS


def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_LEN.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("policy socket closed")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket):
    (n,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    return pickle.loads(_recv_exact(sock, n))


# --- client side (no torch) -------------------------------------------------------------------------------------

class PolicyClient:
    """One connection to the PolicyServer; `act(obs)` -> (action, logp, value, version)."""

    def __init__(self, path: str | Path, timeout_s: float = 120.0) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect(str(path))

    def act(self, obs: dict[str, np.ndarray]) -> tuple[int, float, float, int]:
        send_msg(self.sock, {k: np.ascontiguousarray(v) for k, v in obs.items()})
        r = recv_msg(self.sock)
        if "error" in r:
            raise RuntimeError(f"policy server: {r['error']}")
        return int(r["action"]), float(r["logp"]), float(r["value"]), int(r["version"])

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class EpisodeLog:
    """One seat's decisions in one game, written as an episode file at the end."""

    def __init__(self) -> None:
        self.obs: list[dict[str, np.ndarray]] = []
        self.action: list[int] = []
        self.logp: list[float] = []
        self.value: list[float] = []
        self.version: list[int] = []

    def add(self, obs: dict[str, np.ndarray], action: int, logp: float, value: float, version: int) -> None:
        self.obs.append({k: np.asarray(v) for k, v in obs.items()})
        self.action.append(int(action))
        self.logp.append(float(logp))
        self.value.append(float(value))
        self.version.append(int(version))

    def __len__(self) -> int:
        return len(self.action)

    def write(self, out_dir: str | Path, reward: float, meta: dict) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.action:
            raise ValueError("episode with 0 decisions")
        # only the PPO store's keys (hist=0, known=False nets; PolicyServer refuses others): ~200x smaller files;
        # plus the observable history when the client built it (an observable-history learner's store needs it)
        keys = STORE_KEYS + tuple(k for k in OBS_HISTORY_STORE_KEYS if k in self.obs[0])
        arrays = {f"obs.{k}": np.stack([o[k] for o in self.obs]) for k in keys}
        arrays.update(action=np.array(self.action, np.int64), logp=np.array(self.logp, np.float32),
                      value=np.array(self.value, np.float32), version=np.array(self.version, np.int64),
                      reward=np.array(float(reward), np.float32), meta=np.array(json.dumps(meta)))
        name = f"ep-{time.time_ns()}-{os.getpid()}"
        tmp = out_dir / f".{name}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        final = out_dir / f"{name}.npz"
        os.replace(tmp, final)
        return final


def read_episode(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        obs = {k[4:]: z[k] for k in z.files if k.startswith("obs.")}
        return {"obs": obs, "action": z["action"], "logp": z["logp"], "value": z["value"],
                "version": z["version"], "reward": float(z["reward"]), "meta": json.loads(str(z["meta"]))}


# --- server side (torch) ----------------------------------------------------------------------------------------

def publish_weights(model, path: str | Path, version: int) -> None:
    """Atomically write the learner's weights (incl. value head) for the PolicyServer to pick up."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    torch.save({"version": int(version), "model": state}, tmp)
    os.replace(tmp, path)


class PolicyServer:
    """Batched sampling server over the newest published weights (one per box)."""

    def __init__(self, weights: str | Path, arch_ckpt: str | Path, emb: str | Path, device: str = "cpu",
                 max_batch: int = 64, wait_ms: float = 2.0, seed: int = 0, threads: int = 4) -> None:
        import torch

        from decklight.eval.bc_policy import _run_args
        from decklight.model.policy_v3 import make_policy

        torch.set_num_threads(threads)
        self.torch = torch
        self.weights = Path(weights)
        a = _run_args(Path(arch_ckpt))
        if int(a.get("hist", 0) or 0) or bool(a.get("known", False)):
            raise ValueError("PolicyServer: hist>0 / known nets need hist_long / card_known in the stream obs")
        self.model = make_policy(dict(np.load(emb)), a, dropout=0.0, value=True).to(device).eval()
        self.device = torch.device(device)
        self.max_batch, self.wait_s = int(max_batch), float(wait_ms) / 1000.0
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.version, self._mtime, self._checked = -1, None, 0.0
        self._q: list[tuple[dict, threading.Event, dict]] = []
        self._cv = threading.Condition()
        self.served = 0
        self._reload(force=True)

    def _reload(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._checked < 1.0:
            return
        self._checked = now
        try:
            m = self.weights.stat().st_mtime_ns
        except FileNotFoundError:
            if force:
                raise
            return
        if m == self._mtime:
            return
        blob = self.torch.load(self.weights, map_location="cpu")
        state = {k: v.float() if v.is_floating_point() else v for k, v in blob["model"].items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("value_head.") for k in missing):
            raise RuntimeError(f"PolicyServer: weights do not fit the net ({missing[:3]}, {unexpected[:3]})")
        self.version, self._mtime = int(blob["version"]), m
        print(f"[policy-server] weights version {self.version}", flush=True)

    def infer(self, batch: list[dict[str, np.ndarray]]) -> list[dict]:
        """Sample one action per observation (Collector semantics: multinomial over legal, temperature 1)."""
        from decklight.rl.ppo import masked_log_softmax

        torch = self.torch
        keys = batch[0].keys()
        tob = {k: torch.from_numpy(np.stack([b[k] for b in batch])).to(self.device) for k in keys}
        with torch.no_grad():
            logits, value = self.model.forward_value(tob)
            legal = torch.isfinite(logits)
            lp = masked_log_softmax(logits)
            p = (lp.exp() * legal).float().cpu()
            a = torch.multinomial(p, 1, generator=self.gen).squeeze(1)
            lpa = lp.float().cpu().gather(1, a[:, None]).squeeze(1)
        v = value.float().cpu().numpy()
        return [{"action": int(a[i]), "logp": float(lpa[i]), "value": float(v[i]), "version": self.version}
                for i in range(len(batch))]

    def serve(self, path: str | Path, ready_file: str | Path | None = None) -> None:
        path = Path(path)
        if path.exists():
            path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(path))
        srv.listen(256)
        threading.Thread(target=self._loop, daemon=True).start()
        if ready_file:
            Path(ready_file).write_text(str(os.getpid()))
        print(f"[policy-server] listening on {path}", flush=True)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        try:
            while True:
                obs = recv_msg(conn)
                ev, out = threading.Event(), {}
                with self._cv:
                    self._q.append((obs, ev, out))
                    self._cv.notify()
                ev.wait()
                send_msg(conn, out)
        except (ConnectionError, EOFError, OSError):
            pass
        finally:
            conn.close()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
            time.sleep(self.wait_s)  # let concurrent clients join the batch
            with self._cv:
                items, self._q = self._q[: self.max_batch], self._q[self.max_batch:]
            try:
                self._reload()
                res = self.infer([o for o, _, _ in items])
            except Exception as e:  # noqa: BLE001 - report to every waiting client, keep serving
                res = [{"error": f"{type(e).__name__}: {e}"}] * len(items)
                print(f"ALERT policy-server: {type(e).__name__}: {e}", flush=True)
            for (_, ev, out), r in zip(items, res):
                out.update(r)
                ev.set()
            self.served += len(items)


# --- learner side -----------------------------------------------------------------------------------------------

class ExtFeed:
    """Finished external episodes -> PPO store trajectories (`train_rl.py --ext-dir`).

    `take` loads the oldest unread episode files until `quota` samples, drops episodes whose oldest decision used
    weights older than `min_version` (stale), checks every obs key/shape/dtype against the store, computes the frozen
    reference's log-probs (`ref`, for the KL term) and puts the samples into the store. Consumed files move to
    `<dir>/used/` (kept for audits) so a restart never reads one twice.
    """

    def __init__(self, ext_dir: str | Path) -> None:
        self.dir = Path(ext_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "used").mkdir(exist_ok=True)

    def take(self, store, quota: int, min_version: int, ref, device) -> tuple[list[tuple[np.ndarray, float]], dict]:
        import torch

        from decklight.rl.ppo import masked_log_softmax, to_torch

        files = sorted(self.dir.glob("ep-*.npz"))
        finished: list[tuple[np.ndarray, float]] = []
        st = {"ext_files": 0, "ext_samples": 0, "ext_stale": 0, "ext_wins": 0.0, "ext_backlog": len(files)}
        for f in files:
            if st["ext_samples"] >= quota:
                break
            ep = read_episode(f)
            os.replace(f, self.dir / "used" / f.name)
            if int(ep["version"].min()) < min_version:
                st["ext_stale"] += 1
                continue
            obs = ep["obs"]
            for k, arr in store.obs.items():
                if k not in obs:
                    raise RuntimeError(f"ext episode {f.name}: obs key {k!r} missing")
                if obs[k].shape[1:] != arr.shape[1:] or obs[k].dtype != arr.dtype:
                    raise RuntimeError(f"ext episode {f.name}: obs {k!r} {obs[k].dtype}{obs[k].shape[1:]} "
                                       f"!= store {arr.dtype}{arr.shape[1:]}")
            sub = {k: obs[k] for k in store.obs}
            with torch.no_grad():
                rl = masked_log_softmax(ref(to_torch(sub, device))).float().cpu().numpy()
            if rl.shape[1] < 64:
                rl = np.pad(rl, ((0, 0), (0, 64 - rl.shape[1])))
            slots = store.put(sub, ep["action"], ep["logp"], ep["value"], ref_lp=rl)
            finished.append((slots, float(ep["reward"])))
            st["ext_files"] += 1
            st["ext_samples"] += len(slots)
            st["ext_wins"] += float(ep["reward"] > 0)
        return finished, st
