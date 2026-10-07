"""Turn-1 line generation + scoring (self-imitation step 1: headroom measurement, later the data generator).

For each deck and opening hand: 1 greedy + K sampled turn-1 lines (player 0 acting, opponent
passive), each scored by end field (cards controlled, same measure as scripts/playtest.py) and by the
privileged critic's value of the resulting turn-2 position. Verified going-first records get the same
treatment on their own configs plus an expert replay scored the same way.

Storage is replays only (config + choices); no observations are persisted.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from ygo_zero.decks import Deck
from ygo_zero.demos.game import Responder
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine, Placement
from ygo_zero.model.bc_data import KNOWN_KEYS, OBS_KEYS
from ygo_zero.model.policy import PolicyNet
from ygo_zero.rl.critic import critic_obs

ROOT = Path(__file__).resolve().parents[3]
FIELD_LOCS = (3, 4)  # obs card column 1: 3 mzone, 4 szone (same measure as playtest.my_field)
CAP = 400  # decision cap per turn-1 run (same as playtest: hitting it = "stuck")

TRAIN_OK = {"train": True, "holdout": False, "gauntlet": False, "custom": False}


def my_field(obs) -> int:
    """Cards the acting player's side controls on the field (mzone + szone)."""
    c, m = obs["cards"], obs["card_mask"]
    return int(sum(1 for i in np.flatnonzero(m) if int(c[i, 1]) in FIELD_LOCS and int(c[i, 3]) == 0))


def _turn(obs) -> int:
    return int(obs["global"][2])


def stable_seed(*parts) -> int:
    """A stable int seed from string parts (hash() is salted per process; random.Random(str) is not)."""
    h = hashlib.md5("/".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:15], 16)  # < 2**60: safe for DuelConfig, numpy SeedSequence and torch


def resolve_decks(spec: str, seed: int = 0, max_decks: int = 0) -> tuple[str, list[tuple[str, Path]], bool]:
    """Resolve `--decks` to (split label, [(name, ydk path)], train_ok).

    Split names: train (pool minus HOLDOUT minus kewl-tune), holdout (HOLDOUT names present in the
    pool, plus kewl-tune if present), gauntlet (ygo-agent's decks: EVAL ONLY, never training data).
    Anything else is a comma-separated list of ydk paths (split "custom", never training data).
    `--max_decks N` takes a deterministic sample (sorted, random.Random(seed)).
    """
    if spec in ("train", "holdout", "gauntlet"):
        if spec == "gauntlet":
            paths = sorted((ROOT / "decks" / "gauntlet").glob("*.ydk"))
        else:
            hold = set((ROOT / "decks/pool/HOLDOUT").read_text().split()) | {"kewl-tune"}
            pool = sorted((ROOT / "decks" / "pool").glob("*.ydk"))
            if spec == "train":
                paths = [p for p in pool if p.stem not in hold]
            else:
                have = {p.stem for p in pool}
                paths = [p for p in pool if p.stem in hold]
        decks = [(p.stem, p) for p in paths]
        split = spec
    else:
        decks = []
        for s in spec.split(","):
            s = s.strip()
            if s:
                p = Path(s)
                decks.append((p.stem, p))
        decks.sort()
        split = "custom"
    if max_decks and len(decks) > max_decks:
        decks = sorted(decks)
        decks = sorted(random.Random(seed).sample(decks, max_decks))
    return split, decks, TRAIN_OK[split]


def verified_records(deck: str, limit: int) -> list[dict]:
    """The deck's going-first verified line records, biggest declared end board first (playtest parity)."""
    recs = []
    for f in list(ROOT.glob(f"data/demos/*/verified/{deck}.jsonl")) + list(
        ROOT.glob(f"data/demos/planned*/improved/verified/{deck}.jsonl")
    ):
        for raw in f.read_text().splitlines():
            if raw.strip():
                r = json.loads(raw)
                if r.get("me", 0) == 0:
                    recs.append(r)
    recs.sort(key=lambda r: -len(r.get("end_board_codes") or []))
    return recs[:limit]


def config_to_dict(cfg: DuelConfig) -> dict:
    """A DuelConfig as a JSON-able dict (same keys as verified records' configs)."""
    return {
        "deck0_main": list(cfg.deck0.main), "deck0_extra": list(cfg.deck0.extra),
        "deck1_main": list(cfg.deck1.main), "deck1_extra": list(cfg.deck1.extra),
        "seed": int(cfg.seed), "lp": int(cfg.lp), "hand": int(cfg.hand), "draw": int(cfg.draw),
        "setup": [list(p) for p in cfg.setup], "shuffle": bool(cfg.shuffle),
    }  # fmt: skip


def config_from_dict(c: dict) -> DuelConfig:
    setup = tuple(Placement(*p) for p in c.get("setup", []))
    return DuelConfig(
        Deck(tuple(c["deck0_main"]), tuple(c["deck0_extra"])),
        Deck(tuple(c["deck1_main"]), tuple(c["deck1_extra"])),
        seed=int(c["seed"]), lp=int(c.get("lp", 8000)), hand=int(c.get("hand", 5)),
        draw=int(c.get("draw", 1)), setup=setup, shuffle=c.get("shuffle", True),
    )  # fmt: skip


def hand_config(deck: Deck, opp: Deck, seed: int) -> DuelConfig:
    """A normal opening (5 cards, player 0 first) from a hashed seed."""
    return DuelConfig(deck, opp, seed=seed)


def opp_for(decks: list[str], deck: str, seed: int, hand: int) -> str:
    """The fixed opponent deck for a (deck, hand): another deck of the split, deterministic in (seed, deck, hand)."""
    others = [d for d in decks if d != deck]
    if not others:
        return deck
    return random.Random(f"{seed}/{deck}/{hand}/opp").choice(sorted(others))


def run_turn1(envs: list[CppEnv], pols: list) -> list[dict]:
    """Player 0's turn 1 for several envs in lockstep (opponent passive), like playtest._turn1_fields but
    recording every choice (both players, all decisions incl. forced/passive) so the line replays exactly.

    Returns per env: {choices, p0_decisions, field, stuck, end_obs (copied, None if the game ended),
    winner, player} where player/winner describe the end state (winner None unless the game ended).
    """
    passive = Responder(0)
    for p in {id(p): p for p in pols}.values():
        if getattr(p, "guard", None) is not None:
            p.guard.reset()
    obs = [e.reset() for e in envs]
    last, n = [0] * len(envs), [0] * len(envs)
    choices: list[list[int]] = [[] for _ in envs]
    live = [True] * len(envs)
    while any(live):
        groups: dict[int, list[int]] = {}
        for i, e in enumerate(envs):
            if not live[i]:
                continue
            if obs[i] is None or _turn(obs[i]) > 1 or n[i] >= CAP:
                live[i] = False
                continue
            if e.player == 0:
                last[i] = my_field(obs[i])
                groups.setdefault(id(pols[i]), []).append(i)
            else:
                c = passive.passive(obs[i])
                choices[i].append(int(c))
                obs[i] = e.step(int(c))
        for idx in groups.values():
            buf = {k: np.stack([obs[i][k] for i in idx]) for k in obs[idx[0]]}
            acts = pols[idx[0]](buf, np.arange(len(idx)), slots=np.array(idx))
            for i, a in zip(idx, acts, strict=True):
                n[i] += 1
                choices[i].append(int(a))
                obs[i] = envs[i].step(int(a))
    out = []
    for i, e in enumerate(envs):
        # End field = p0's cards in the final obs (whoever-to-act's view; turn 2 can open with
        # player 0 to act). Only a game over (obs None) falls back to the last seen field.
        fld = last[i]
        if obs[i] is not None and (_turn(obs[i]) > 1 or n[i] >= CAP):
            c, m = obs[i]["cards"], obs[i]["card_mask"]
            ctrl = 0 if e.player == 0 else 1
            fld = int(sum(1 for j in np.flatnonzero(m) if int(c[j, 1]) in FIELD_LOCS and int(c[j, 3]) == ctrl))
        end = None if obs[i] is None else {k: np.array(v, copy=True) for k, v in obs[i].items()}
        out.append({
            "choices": choices[i], "p0_decisions": int(n[i]), "field": int(fld),
            "stuck": bool(n[i] >= CAP), "end_obs": end,
            "winner": None if obs[i] is not None else e.winner, "player": e.player,
        })
    return out


def replay_turn1(engine: Engine, cfg: DuelConfig, choices: list[int], auto_single: bool = True) -> dict:
    """Replay a recorded choice list through turn 1 (stops at turn 2 / game over / choices exhausted).

    Returns the same end-state dict as run_turn1 (minus p0_decisions accounting done by the caller).
    Raises on an illegal choice: the line must replay exactly.
    """
    from ygo_zero.demos.game import _opts

    env = CppEnv(engine, cfg, auto_single)
    obs = env.reset()
    last, actors0 = 0, 0
    for c in choices:
        if obs is None or _turn(obs) > 1:
            break
        if env.player == 0:
            last = my_field(obs)
            actors0 += 1
        n = len(_opts(obs))
        if not 0 <= int(c) < n:
            raise ValueError(f"choice {c} illegal with {n} options (player {env.player})")
        obs = env.step(int(c))
    fld = last
    if obs is not None:
        if _turn(obs) > 1:
            c, m = obs["cards"], obs["card_mask"]
            ctrl = 0 if env.player == 0 else 1
            fld = int(sum(1 for j in np.flatnonzero(m) if int(c[j, 1]) in FIELD_LOCS and int(c[j, 3]) == ctrl))
        elif env.player == 1:
            c, m = obs["cards"], obs["card_mask"]
            fld = int(sum(1 for j in np.flatnonzero(m) if int(c[j, 1]) in FIELD_LOCS and int(c[j, 3]) == 1))
        else:
            fld = my_field(obs)
    end = None if obs is None else {k: np.array(v, copy=True) for k, v in obs.items()}
    return {"field": int(fld), "end_obs": end,
            "winner": None if obs is not None else env.winner, "player": env.player,
            "p0_decisions": actors0, "stuck": False}


def run_args(ckpt: Path) -> dict:
    """The training args recorded next to a checkpoint (same convention as BCPolicy._run_args)."""
    m = Path(ckpt).parent / "metrics.jsonl"
    if m.is_file():
        for raw in m.read_text().splitlines():
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "args" in rec:
                return rec["args"]
    return {}


def load_critic_local(ckpt: Path, emb: dict, a: dict) -> PolicyNet:
    """Privileged critic PolicyNet(known=True, value=True) from a checkpoint (train_rl.load_critic parity).

    Non-strict load: the checkpoint may lack the known inputs and/or the value head (a BC checkpoint);
    only those may be missing. Local copy so this module never imports the training script.
    """
    net = PolicyNet(emb, d=a.get("d", 256), layers=a.get("layers", 4), heads=a.get("heads", 4), dropout=0.0,
                    hist=int(a.get("hist", 0) or 0), known=True, value=True)  # fmt: skip
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()},
                                              strict=False)  # fmt: skip
    assert not unexpected and all(k.startswith("value_head.") or k.startswith("known_emb.")
                                  for k in missing), (missing, unexpected)
    if any(k.startswith("known_emb.") for k in missing):
        torch.nn.init.zeros_(net.known_emb.weight)
    return net


@torch.no_grad()
def critic_vp0(critic: PolicyNet, ends: list[dict], hist: int, device: torch.device) -> list[float]:
    """v_p0 (player 0's estimated outcome in [-1, 1]) for turn-1 end states, one batched critic forward.

    Live states (turn 2 started): the privileged critic on the obs with priv_code, negated to player 0's
    view (the value head scores the acting player; at turn-2 start that is player 1). Ended games:
    +1/-1 by the winner (0.0 on a draw). The input list mixes both; order is preserved.
    """
    live_idx, live_obs, live_player = [], [], []
    for i, e in enumerate(ends):
        if e["end_obs"] is not None:
            live_idx.append(i)
            live_obs.append(e["end_obs"])
            live_player.append(e["player"])
    vals: dict[int, float] = {}
    if live_obs:
        batch: dict[str, np.ndarray] = {}
        for k in OBS_KEYS:  # hist_long already cut to `hist`, like rl.ppo.gather
            if k == "history" and hist > 0:
                batch[k] = np.stack([o["hist_long"][:, :hist] for o in live_obs])
            elif k == "history_mask" and hist > 0:
                batch[k] = np.stack([o["hist_long_mask"][:, :hist] for o in live_obs])
            else:
                batch[k] = np.stack([o[k] for o in live_obs])
        for k in KNOWN_KEYS:  # CppEnv obs carry these; zeros when absent
            batch[k] = _known_or_zeros(live_obs, k)
        priv = np.stack([o["priv_code"] for o in live_obs])
        cob = critic_obs(batch, priv)
        t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in cob.items()}
        v = critic.forward_value(t)[1].float().cpu().numpy()
        for i, vv, p in zip(live_idx, v, live_player, strict=True):
            vals[i] = float(vv) if p == 0 else float(-vv)
    out = []
    for i, e in enumerate(ends):
        if i in vals:
            out.append(vals[i])
        else:
            w = e["winner"]
            out.append(1.0 if w == 0 else (-1.0 if w == 1 else 0.0))
    return out


def _known_or_zeros(obs_list: list[dict], key: str) -> np.ndarray:
    if all(key in o for o in obs_list):
        return np.stack([o[key] for o in obs_list])
    n = obs_list[0]["cards"].shape[0]
    dt = np.uint8 if key == "card_known" else np.int32
    return np.zeros((len(obs_list), n), dt)


def best_of(perf: list[tuple[int, float]], k: int, by: int) -> tuple[int, float]:
    """(field, v) of the best of the first k samples, ranked by field (by=0) or v (by=1); first wins ties."""
    sub = perf[:k]
    j = max(range(len(sub)), key=lambda j: (sub[j][by], -j))
    return sub[j]


def summarize(hands: list[dict], klist: list[int]) -> dict:
    """Aggregate per-hand {greedy: (field, v), samples: [(field, v)], expert: (field, v) | None} rows.

    Best-of-k is per hand (the headroom question), then averaged over hands; best-by-field and best-by-v
    each report both measures of their argmax row.
    """
    s: dict = {"hands": len(hands)}
    if not hands:
        return s
    s["greedy_field"] = float(np.mean([h["greedy"][0] for h in hands]))
    s["greedy_v"] = float(np.mean([h["greedy"][1] for h in hands]))
    for k in klist:
        if not all(len(h["samples"]) >= k for h in hands):
            continue
        bf = [best_of(h["samples"], k, 0) for h in hands]
        bv = [best_of(h["samples"], k, 1) for h in hands]
        s[f"best{k}_by_field"] = [float(np.mean([p[0] for p in bf])), float(np.mean([p[1] for p in bf]))]
        s[f"best{k}_by_v"] = [float(np.mean([p[0] for p in bv])), float(np.mean([p[1] for p in bv]))]
    flat_f = [p[0] for h in hands for p in h["samples"]]
    flat_v = [p[1] for h in hands for p in h["samples"]]
    s["mean_sampled_field"] = float(np.mean(flat_f)) if flat_f else None
    s["mean_sampled_v"] = float(np.mean(flat_v)) if flat_v else None
    ev = [(h["expert"], h["greedy"]) for h in hands if h.get("expert") is not None]
    if ev:
        s["expert_field"] = float(np.mean([e[0][0] for e in ev]))
        s["expert_v"] = float(np.mean([e[0][1] for e in ev]))
        s["critic_ranks_expert_above_greedy"] = float(np.mean([e[0][1] > e[1][1] for e in ev]))
        s["expert_minus_greedy_v"] = float(np.mean([e[0][1] - e[1][1] for e in ev]))
    return s


def fail_loud(problems: list[str], out_dir: Path) -> "SystemExit":
    """Report invariant violations to stderr and <out>/ALERT (fail fast and loud)."""
    import sys

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ALERT", "a") as f:
        for p in problems:
            line = f"ALERT turn1_lines: {p}"
            print(line, file=sys.stderr, flush=True)
            f.write(line + "\n")
    return SystemExit(3)


# --- whole-turn lines (scripts/turn_lines.py): any start state, active opponent ---

def turn_player_of(obs, player: int) -> int:
    """The turn player at `obs` (absolute seat): the acting player when global[5] says so."""
    return int(player) if int(obs["global"][5]) == 1 else 1 - int(player)


def actor_field_of(obs, player: int, actor: int) -> int:
    """Field cards (mzone + szone) controlled by `actor` from an obs seen by `player`."""
    c, m = obs["cards"], obs["card_mask"]
    ctrl = 0 if int(player) == int(actor) else 1
    return int(sum(1 for i in np.flatnonzero(m) if int(c[i, 1]) in FIELD_LOCS and int(c[i, 3]) == ctrl))


def lp_diff_of(obs, player: int, actor: int) -> int:
    """LP(actor) - LP(opponent) from an obs seen by `player`."""
    g = obs["global"]
    if int(player) == int(actor):
        return int(g[0]) - int(g[1])
    return int(g[1]) - int(g[0])


def dags_config(row: dict) -> DuelConfig:
    """A DuelConfig for a DAGS starts.jsonl row (mains already in final shuffled order)."""
    from ygo_zero.demos.dataset import record_config as _rc

    if isinstance(row.get("config"), dict) and "deck0_main" in row["config"]:
        return _rc({"config": row["config"]})
    setup = tuple(Placement(*p) for p in row.get("setup", []))
    return DuelConfig(
        Deck(tuple(row["main0"]), tuple(row["extra0"])),
        Deck(tuple(row["main1"]), tuple(row["extra1"])),
        seed=int(row["seed"]), lp=int(row.get("lp", 8000)), hand=int(row.get("hand", 5)),
        draw=int(row.get("draw", 1)), setup=setup, shuffle=False,
    )  # fmt: skip


def replay_prefix(engine: Engine, cfg: DuelConfig, prefix: list[int], auto_single: bool = True):
    """Replay a prefix; returns (env, obs). Raises on game-over or an illegal choice."""
    from ygo_zero.demos.game import _opts

    env = CppEnv(engine, cfg, auto_single)
    obs = env.reset()
    if obs is None:
        raise ValueError("game ended before the first decision")
    for c in prefix:
        if obs is None:
            raise ValueError("game ended inside the prefix")
        if not 0 <= int(c) < len(_opts(obs)):
            raise ValueError(f"prefix choice {c} illegal with {len(_opts(obs))} options")
        obs = env.step(int(c))
    if obs is None:
        raise ValueError("game ended at the end of the prefix")
    return env, obs


def run_turn(envs: list[CppEnv], pols: list, actor: int, turn_start: int,
             opp: str = "policy", opp_pol=None, cap: int = CAP) -> list[dict]:
    """One full turn for `actor` (turn number `turn_start`) across envs in lockstep.

    Every env is already positioned at the start state (prefix replayed); its next turn number
    must be `turn_start`. The actor's decisions come from `pols[i]` (greedy or sampler); every
    other seat's decisions come from `opp_pol` greedily (opp="policy") or the passive responder
    (opp="passive"). Stops per env at the next turn, the duel end, or `cap` actor+opp decisions.
    Returns per env the same end-state dict as run_turn1 plus actor-view measures and the new
    (post-prefix) choices.
    """
    raise RuntimeError("run_turn needs current observations: use run_turn_from_obs")


def run_turn_from_obs(envs, obs_list, pols: list, actor: int, turn_start: int,
                      opp: str = "policy", opp_pol=None, passive=None, cap: int = CAP) -> list[dict]:
    """Lockstep turn play from already-observed states (see run_turn)."""
    import numpy as np

    if passive is None:
        passive = Responder(0)
    for p in {id(p): p for p in ([*pols] + ([opp_pol] if opp_pol is not None else []))}.values():
        if getattr(p, "guard", None) is not None:
            p.guard.reset()
    obs = list(obs_list)
    n_total = [0] * len(envs)
    n_actor = [0] * len(envs)
    n_opp = [0] * len(envs)
    n_opp_options = [0] * len(envs)  # non-actor decisions with a real choice (>1 option)
    last_field = [actor_field_of(o, e.player, actor) if o is not None else 0
                  for o, e in zip(obs, envs, strict=True)]
    last_lp = [lp_diff_of(o, e.player, actor) if o is not None else 0
               for o, e in zip(obs, envs, strict=True)]
    choices: list[list[int]] = [[] for _ in envs]
    live = [True] * len(envs)
    for i, (o, e) in enumerate(zip(obs, envs, strict=True)):
        if o is None or _turn(o) != int(turn_start):
            live[i] = False
    while any(live):
        actor_groups: dict[int, list[int]] = {}
        opp_idx: list[int] = []
        for i, e in enumerate(envs):
            if not live[i]:
                continue
            if obs[i] is None or _turn(obs[i]) != int(turn_start) or n_total[i] >= cap:
                live[i] = False
                continue
            if e.player == actor:
                last_field[i] = actor_field_of(obs[i], e.player, actor)
                last_lp[i] = lp_diff_of(obs[i], e.player, actor)
                actor_groups.setdefault(id(pols[i]), []).append(i)
            elif opp == "policy" and opp_pol is not None:
                n_opp_options[i] += int(int(obs[i]["action_mask"].sum()) > 1)
                opp_idx.append(i)
            else:
                n_opp_options[i] += int(int(obs[i]["action_mask"].sum()) > 1)
                c = passive.passive(obs[i])
                choices[i].append(int(c))
                n_total[i] += 1
                n_opp[i] += 1
                obs[i] = e.step(int(c))
        # one batched forward per distinct actor policy, plus one for the opponent policy
        for pid, idx in actor_groups.items():
            pol = next(p for p in pols if id(p) == pid)
            buf = {k: np.stack([obs[i][k] for i in idx]) for k in obs[idx[0]]}
            acts = pol(buf, np.arange(len(idx)), slots=np.array(idx))
            for i, a in zip(idx, acts, strict=True):
                assert envs[i].player == actor
                n_total[i] += 1
                n_actor[i] += 1
                choices[i].append(int(a))
                obs[i] = envs[i].step(int(a))
        if opp_idx:
            buf = {k: np.stack([obs[i][k] for i in opp_idx]) for k in obs[opp_idx[0]]}
            acts = opp_pol(buf, np.arange(len(opp_idx)), slots=np.array(opp_idx))
            for i, a in zip(opp_idx, acts, strict=True):
                assert envs[i].player != actor
                n_total[i] += 1
                n_opp[i] += 1
                choices[i].append(int(a))
                obs[i] = envs[i].step(int(a))
    out = []
    for i, e in enumerate(envs):
        fld, ofld, lp = last_field[i], 0, last_lp[i]
        if obs[i] is not None:
            fld = actor_field_of(obs[i], e.player, actor)
            ofld = actor_field_of(obs[i], e.player, 1 - actor)
            lp = lp_diff_of(obs[i], e.player, actor)
        end = None if obs[i] is None else {k: np.array(v, copy=True) for k, v in obs[i].items()}
        out.append({
            "choices": choices[i], "actor_decisions": int(n_actor[i]),
            "p0_decisions": int(n_actor[i]) if int(actor) == 0 else 0,
            "opp_decisions": int(n_opp[i]), "opp_options": int(n_opp_options[i]),
            "field": int(fld), "opp_field": int(ofld), "lp_diff": int(lp),
            "stuck": bool(n_total[i] >= cap), "end_obs": end,
            "winner": None if obs[i] is not None else e.winner, "player": e.player,
        })
    return out


def replay_turn(engine: Engine, cfg: DuelConfig, choices: list[int], actor: int, turn_start: int,
                auto_single: bool = True) -> dict:
    """Replay a full choice list (prefix + turn) until the turn after `turn_start` starts.

    Returns {actors, turns, field, opp_field, lp_diff, actor_decisions} with fields measured from
    the actor's view (like run_turn_from_obs). Raises on an illegal choice.
    """
    from ygo_zero.demos.game import _opts

    env = CppEnv(engine, cfg, auto_single)
    obs = env.reset()
    actors, turns, last_f, last_o, last_lp, n_actor = [], [], 0, 0, 0, 0
    for c in choices:
        if obs is None or _turn(obs) > int(turn_start):
            break
        actors.append(int(env.player))
        turns.append(_turn(obs))
        if env.player == actor:
            last_f = actor_field_of(obs, env.player, actor)
            last_lp = lp_diff_of(obs, env.player, actor)
            n_actor += 1
        n = len(_opts(obs))
        if not 0 <= int(c) < n:
            raise ValueError(f"choice {c} illegal with {n} options (player {env.player})")
        obs = env.step(int(c))
    # NOTE: choices from earlier turns (< turn_start, the prefix) are context: they are replayed
    # and recorded above exactly like the turn's own decisions; the caller labels them.
    fld, ofld, lp = last_f, last_o, last_lp
    if obs is not None:
        fld = actor_field_of(obs, env.player, actor)
        ofld = actor_field_of(obs, env.player, 1 - actor)
        lp = lp_diff_of(obs, env.player, actor)
    end = None if obs is None else {k: np.array(v, copy=True) for k, v in obs.items()}
    return {"actors": actors, "turns": turns, "field": int(fld), "opp_field": int(ofld),
            "lp_diff": int(lp), "actor_decisions": int(n_actor),
            "p0_decisions": int(n_actor) if int(actor) == 0 else 0,
            "end_obs": end, "winner": None if obs is not None else env.winner, "player": env.player}
