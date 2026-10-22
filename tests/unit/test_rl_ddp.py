"""Multi-GPU PPO data-parallel (DDP) helpers + equivalence (CPU gloo, 2 processes, tiny d=16 nets).

- dist helpers: global advantage normalization == single-process norm on the
  concatenated data; MIN-minibatch; scalar all-reduces; broadcast; checksum.
- One `ppo_update` on two ranks with identical init but different data gives
  identical params across ranks, equal to the single-process update on the
  concatenated data (plain PPO and --priv-critic), within tolerance.
- The checksum check fires when one rank's params are perturbed.
- WORLD_SIZE unset: single-process no-ops (existing RL tests cover behaviour).
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from decklight.model.policy import PolicyNet
from decklight.rl import dist as distmod
from decklight.rl.ppo import SlotStore

_TRAIN_RL = None


def _train_rl_mod():
    global _TRAIN_RL
    if _TRAIN_RL is None:
        p = Path(__file__).resolve().parents[2] / "scripts" / "train_rl.py"
        spec = importlib.util.spec_from_file_location("train_rl_ddp_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _TRAIN_RL = mod
    return _TRAIN_RL


def _emb():
    rng = np.random.default_rng(0)
    n = 10
    return {
        "codes": np.arange(1, n + 1),
        "card": rng.normal(size=(n, 8)).astype(np.float32),
        "effects": rng.normal(size=(n, 2, 8)).astype(np.float32),
        "n_effects": np.ones(n, dtype=np.int64),
    }


def _obs_np(n, priv=False):
    mask = np.zeros((n, 64), bool)
    mask[:, :2] = True
    out = {
        "cards": np.zeros((n, 160, 21), np.float32),
        "card_mask": np.ones((n, 160), bool),
        "global": np.zeros((n, 24), np.float32),
        "chain": np.zeros((n, 16, 5), np.float32),
        "chain_mask": np.ones((n, 16), bool),
        "actions": np.zeros((n, 64, 10), np.float32),
        "action_mask": mask,
        "history": np.zeros((n, 32, 13), np.float32),
        "history_mask": np.ones((n, 32), bool),
    }
    if priv:
        out["priv_code"] = np.zeros((n, 160), np.int32)
    return out


def _spawn(fn, *args, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    mp.spawn(fn, args=args, nprocs=2, join=True)


# -- single-process no-ops --------------------------------------------------------

def test_dist_helpers_noop_single_process():
    assert not distmod.is_dist()
    assert distmod.world_size() == 1 and distmod.rank() == 0 and distmod.is_main()
    assert distmod.init_from_env("cpu") == (0, 1)
    assert not dist.is_initialized()
    dev = torch.device("cpu")
    assert distmod.sum_scalar(2.5, dev) == 2.5
    assert distmod.mean_scalar(2.5, dev) == 2.5
    assert distmod.min_scalar(2.5, dev) == 2.5
    assert distmod.max_scalar(2.5, dev) == 2.5
    assert distmod.min_int(7, dev) == 7
    assert distmod.bcast_flag(True) is True and distmod.bcast_flag(False) is False
    adv = np.array([1.0, 2.0, 3.0], np.float32)
    n = distmod.global_normalize(adv, dev)
    assert np.allclose(n, (adv - adv.mean()) / (adv.std() + 1e-8))
    mean, std, cnt = distmod.global_adv_stats(adv, dev)
    assert mean == float(adv.mean()) and cnt == 3
    assert abs(std - float(adv.std())) < 1e-6  # helper accumulates in float64
    net = PolicyNet(_emb(), d=16, layers=1, heads=2, dropout=0.0, value=True)
    assert distmod.checksum_ok(net, dev)
    x = torch.randn(3, requires_grad=True)
    (x * 2).sum().backward()
    before = x.grad.clone()
    distmod.average_grads_(net)  # no group: no-op, must not touch grads
    assert torch.equal(x.grad, before)
    distmod.barrier()
    distmod.teardown()  # no group: no-op


# -- two-rank helper checks in one spawn ------------------------------------------

def _helpers_worker(rank, world, tmpdir):
    dev = torch.device("cpu")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    assert distmod.is_dist() and distmod.world_size() == 2 and distmod.rank() == rank
    assert distmod.is_main() == (rank == 0)
    out = {}
    # global adv norm == single-process norm on the concatenation
    rng = np.random.default_rng(10 + rank)
    adv = rng.normal(size=(6,)).astype(np.float32)
    out["norm"] = distmod.global_normalize(adv, dev)
    np.save(f"{tmpdir}/adv{rank}.npy", adv)
    # MIN-minibatch + scalar reductions
    out["min"] = distmod.min_int(3 if rank == 0 else 5, dev)
    out["sum"] = distmod.sum_scalar(float(rank + 1), dev)
    out["mean"] = distmod.mean_scalar(float(rank + 1), dev)
    out["max"] = distmod.max_scalar(float(rank + 1), dev)
    out["bcast_t"] = distmod.bcast_flag(rank == 0)
    out["bcast_f"] = distmod.bcast_flag(False)
    # checksum: True identical, False after rank 1 perturbs
    torch.manual_seed(0)
    net = PolicyNet(_emb(), d=16, layers=1, heads=2, dropout=0.0, value=True)
    out["ck_same"] = distmod.checksum_ok(net, dev)
    if rank == 1:
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1.0)
                break
    out["ck_diff"] = distmod.checksum_ok(net, dev)
    torch.save(out, f"{tmpdir}/help{rank}.pt")
    dist.destroy_process_group()


def test_ddp_helpers_two_ranks(tmp_path):
    _spawn(_helpers_worker, 2, str(tmp_path), port=29541)
    h0 = torch.load(str(tmp_path / "help0.pt"), weights_only=False)
    h1 = torch.load(str(tmp_path / "help1.pt"), weights_only=False)
    a0 = np.load(str(tmp_path / "adv0.npy"))
    a1 = np.load(str(tmp_path / "adv1.npy"))
    cat = np.concatenate([a0, a1])
    want = ((cat - cat.mean()) / (cat.std() + 1e-8)).astype(np.float32)
    assert np.allclose(np.concatenate([h0["norm"], h1["norm"]]), want, atol=1e-6)
    assert h0["min"] == h1["min"] == 3
    assert h0["sum"] == h1["sum"] == 3.0
    assert h0["mean"] == h1["mean"] == 1.5
    assert h0["max"] == h1["max"] == 2.0
    assert h0["bcast_t"] and h1["bcast_t"]
    assert not h0["bcast_f"] and not h1["bcast_f"]
    assert h0["ck_same"] and h1["ck_same"]
    assert not h0["ck_diff"] and not h1["ck_diff"]


# -- ppo_update equivalence: 2 ranks x shard vs single x concat -------------------

N_SHARD = 8


def _shard_data(priv, rank):
    rng = np.random.default_rng(1000 + rank)
    obs = _obs_np(N_SHARD, priv=priv)
    act = rng.integers(0, 2, N_SHARD).astype(np.int64)
    logp = np.full(N_SHARD, -0.69, np.float32)
    val = rng.normal(size=N_SHARD).astype(np.float32)
    ref = np.zeros((N_SHARD, 64), np.float32)
    adv = rng.normal(size=N_SHARD).astype(np.float32)
    ret = rng.normal(size=N_SHARD).astype(np.float32)
    adv_n = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
    return obs, act, logp, val, ref, adv, ret, adv_n


def _run_update(priv, obs, act, logp, val, ref, adv_n, ret):
    """One full-batch ppo_update (epochs=1); returns (model, critic-or-None)."""
    mod = _train_rl_mod()
    dev = torch.device("cpu")
    emb = _emb()
    torch.manual_seed(0)
    model = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, value=True).to(dev)
    critic = None
    if priv:
        critic = PolicyNet(emb, d=16, layers=1, heads=2, dropout=0.0, known=True,
                           value=True).to(dev)
    store = SlotStore(64, obs, vrpo=False)
    slots = store.put(obs, act, logp, val, ref)
    trunk = [p for n, p in model.named_parameters() if not n.startswith("value_head.")]
    opt = torch.optim.AdamW([{"params": trunk},
                             {"params": list(model.value_head.parameters()), "lr": 1e-4}],
                            lr=2e-5)
    vopt = torch.optim.AdamW(model.value_head.parameters(), lr=3e-4)
    copt = torch.optim.AdamW(critic.parameters(), lr=3e-4) if priv else None
    # Full-batch single step each: rank grad = mean over its shard, averaged across
    # ranks == mean over the concatenation (equal shard sizes) == the single update.
    mod.ppo_update(model, model, opt, vopt, store, slots, adv_n, ret,
                   np.random.default_rng(7), dev, clip=0.2, vf_coef=0.5, kl_coef=0.0,
                   ent_coef=0.003, epochs=1, minibatch=len(slots), warm=False,
                   precision="fp32", max_upload_gb=6.0, compile=False, critic=critic,
                   copt=copt, priv_critic=priv, adv="gae")
    return model, critic


def _ppo_worker(rank, world, tmpdir, priv):
    dev = torch.device("cpu")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    obs, act, logp, val, ref, adv, ret, adv_n = _shard_data(priv, rank)
    model, critic = _run_update(priv, obs, act, logp, val, ref, adv_n, ret)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"{tmpdir}/m{rank}.pt")
    if priv:
        torch.save({k: v.cpu() for k, v in critic.state_dict().items()}, f"{tmpdir}/c{rank}.pt")
    dist.destroy_process_group()


def _check_equivalence(tmp_path, priv, port):
    _spawn(_ppo_worker, 2, str(tmp_path), priv, port=port)
    m0 = torch.load(str(tmp_path / "m0.pt"), weights_only=True)
    m1 = torch.load(str(tmp_path / "m1.pt"), weights_only=True)
    assert m0.keys() == m1.keys()
    for k in m0:
        assert torch.allclose(m0[k].float(), m1[k].float(), atol=1e-5, rtol=1e-4), k
    # single-process reference on the concatenated shards (no group here: helpers no-op)
    assert not distmod.is_dist()
    parts = [_shard_data(priv, r) for r in (0, 1)]
    obs = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    act = np.concatenate([p[1] for p in parts])
    logp = np.concatenate([p[2] for p in parts])
    val = np.concatenate([p[3] for p in parts])
    ref = np.concatenate([p[4] for p in parts])
    adv_n = np.concatenate([p[7] for p in parts])
    ret = np.concatenate([p[6] for p in parts])
    model, critic = _run_update(priv, obs, act, logp, val, ref, adv_n, ret)
    ref_state = {k: v.cpu() for k, v in model.state_dict().items()}
    for k in ref_state:
        assert torch.allclose(m0[k].float(), ref_state[k].float(), atol=1e-4, rtol=1e-4), k
    if priv:
        c0 = torch.load(str(tmp_path / "c0.pt"), weights_only=True)
        c1 = torch.load(str(tmp_path / "c1.pt"), weights_only=True)
        assert c0.keys() == c1.keys()
        for k in c0:
            assert torch.allclose(c0[k].float(), c1[k].float(), atol=1e-5, rtol=1e-4), k
        cref = {k: v.cpu() for k, v in critic.state_dict().items()}
        for k in cref:
            assert torch.allclose(c0[k].float(), cref[k].float(), atol=1e-4, rtol=1e-4), k


def test_ppo_update_ddp_matches_single(tmp_path):
    _check_equivalence(tmp_path, priv=False, port=29542)


def test_ppo_update_ddp_matches_single_priv(tmp_path):
    _check_equivalence(tmp_path, priv=True, port=29543)
