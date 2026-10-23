"""Value over-trust audit: does search pick positions the learned value net over-rates?

Designed to run on a vast GPU box (loads the policy + value nets); do not run it here
except --dry-run (engine only, replays 2 games, no network).

    uv run python scripts/audit/value_overtrust.py --games runs/<job>/card-search-<deck>/base/games-0.jsonl \\
        --ckpt runs/<run>/bc/best.pt --value runs/<run>/value/best.pt [--max-games N] [--device cuda]
    uv run python scripts/audit/value_overtrust.py --games <games.jsonl> --dry-run

Input: searched game records (search_match --search 1 output: decks, seed, choices, actors,
s_seat, winner, and the per-searched-decision log with k/cands/chosen/played). Each game is
replayed; at every one of our (S seat's) searched decisions the value net scores (a) the
position after the move search chose and (b) after the raw policy's argmax move (recomputed
with --ckpt behind a shared LoopGuard, the unrouted search_match discipline; mismatches vs
the logged `played` are reported). Value v -> p = (v + 1) / 2, converted from the acting
player's view to S's view (1 - p when the next actor is the opponent; terminal moves score
the game outcome). Draws/truncated games carry no outcome and are skipped.

Output: Brier mean((p - y)^2) and mean(p - y) (signed bias) for chosen-by-search positions
vs raw-argmax positions, separately for overridden decisions (search != raw) and agreeing
ones, with bootstrap 95% CIs over games. Over-trust = search-chosen overridden positions
have positive bias significantly larger than raw ones. Also prints the 20 most over-rated
overridden decisions (turn, options, p, outcome).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def fail(msg: str) -> None:
    print(f"ALERT value_overtrust: {msg}", flush=True)
    raise SystemExit(2)


def deck_path(name: str) -> Path:
    p = Path(name)
    if p.is_file():
        return p
    if p.suffix != ".ydk" and p.with_suffix(".ydk").is_file():
        return p.with_suffix(".ydk")
    for d in ("pool", "gauntlet", "newcards"):
        p = ROOT / "decks" / d / f"{name}.ydk"
        if p.is_file():
            return p
    fail(f"deck {name} not found in decks/pool|gauntlet|newcards")
    raise AssertionError("unreachable")


def load_games(path: Path) -> list[dict]:
    if path.is_dir():
        files = sorted(path.glob("games-*.jsonl"))
    else:
        files = [path]
    if not files:
        fail(f"no games files under {path}")
    recs = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    if not recs:
        fail(f"no game records in {files}")
    return recs


def to_p(v: float) -> float:
    """Value-head output ([-1, 1]) -> win probability, clipped like rollout.value_to_prob."""
    return float(min(1.0, max(0.0, (float(v) + 1.0) / 2.0)))


def group_metrics(rows: list[dict], key: str) -> dict:
    """Brier + signed bias over rows' `key` probabilities (rows carry p_* and y)."""
    ps = np.array([r[key] for r in rows], dtype=np.float64)
    ys = np.array([r["y"] for r in rows], dtype=np.float64)
    err = ps - ys
    return {"n": len(rows), "brier": float((err ** 2).mean()) if len(rows) else 0.0,
            "bias": float(err.mean()) if len(rows) else 0.0}


def summarize(rows: list[dict]) -> dict:
    """Brier/bias for the four groups: overridden chosen/raw, agreeing (chosen == raw)."""
    ov = [r for r in rows if r["overridden"]]
    ag = [r for r in rows if not r["overridden"]]
    return {
        "n_games": len({r["game"] for r in rows}),
        "n_overridden": len(ov),
        "n_agreeing": len(ag),
        "chosen_overridden": group_metrics(ov, "p_search"),
        "raw_overridden": group_metrics([r for r in ov if r["p_raw"] is not None], "p_raw"),
        "agreeing": group_metrics(ag, "p_search"),
    }


def bootstrap_cis(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Bootstrap 95% CIs over games (cluster resample): {group: {metric: (lo, hi)}}."""
    games: dict[int, list[dict]] = {}
    for r in rows:
        games.setdefault(r["game"], []).append(r)
    ids = sorted(games)
    if not ids:
        return {}
    rng = np.random.default_rng(seed)
    acc: dict[str, dict[str, list]] = {}
    for _ in range(n_boot):
        flat = [r for j in rng.integers(0, len(ids), len(ids)) for r in games[ids[int(j)]]]
        s = summarize(flat)
        for group in ("chosen_overridden", "raw_overridden", "agreeing"):
            for metric in ("brier", "bias"):
                acc.setdefault(group, {}).setdefault(metric, []).append(s[group][metric])
    return {g: {m: (round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))
                for m, v in ms.items()} for g, ms in acc.items()}


def bootstrap_gap_ci(rows: list[dict], n_boot: int, seed: int) -> tuple:
    """Bootstrap 95% CI of (chosen bias - raw bias) over overridden decisions, over games."""
    games: dict[int, list[dict]] = {}
    for r in [x for x in rows if x["overridden"] and x["p_raw"] is not None]:
        games.setdefault(r["game"], []).append(r)
    ids = sorted(games)
    if not ids:
        return (None, None)
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        flat = [r for j in rng.integers(0, len(ids), len(ids)) for r in games[ids[int(j)]]]
        bc = float(np.mean([r["p_search"] - r["y"] for r in flat])) if flat else 0.0
        br = float(np.mean([r["p_raw"] - r["y"] for r in flat])) if flat else 0.0
        gaps.append(bc - br)
    return (round(float(np.percentile(gaps, 2.5)), 4), round(float(np.percentile(gaps, 97.5)), 4))


def overtrust_verdict(summ: dict, cis: dict) -> str:
    """Over-trust = search-chosen overridden positions have positive bias significantly
    larger than raw ones (bootstrap CI of the bias gap excludes 0 from below, and the
    chosen bias CI is above 0)."""
    gap_lo, gap_hi = cis.get("bias_gap", (None, None))
    ch_lo, ch_hi = cis.get("chosen_overridden", {}).get("bias", (None, None))
    if gap_lo is None or ch_lo is None:
        return "no verdict (no overridden decisions)"
    if gap_lo > 0 and ch_lo > 0:
        return (f"OVER-TRUST: chosen bias {summ['chosen_overridden']['bias']:.3f} "
                f"[{ch_lo:.3f}, {ch_hi:.3f}] exceeds raw by a gap "
                f"[{gap_lo:.3f}, {gap_hi:.3f}] (95% CI over games)")
    return (f"no significant over-trust: chosen bias {summ['chosen_overridden']['bias']:.3f} "
            f"[{ch_lo:.3f}, {ch_hi:.3f}], bias gap [{gap_lo:.3f}, {gap_hi:.3f}]")


def replay_config(rec: dict):
    """DuelConfig for a search_match record (same defaults as search_match deals)."""
    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig

    d0, d1 = rec["decks"]
    return DuelConfig(load_ydk(deck_path(d0)), load_ydk(deck_path(d1)), seed=int(rec["seed"]))


def dry_run(recs: list[dict]) -> None:
    """Engine-only replay of the first 2 games (no network): prints decision counts."""
    from decklight.engine.cpp_env import CppEnv
    from decklight.engine.duel import Engine

    engine = Engine()
    for rec in recs[:2]:
        env = CppEnv(engine, replay_config(rec), True)
        obs = env.reset()
        for c in rec["choices"]:
            if obs is None:
                fail(f"game {rec.get('game_idx')}: replay ended early at choice {len(env.choices)}")
            obs = env.step(int(c))
        print(f"game {rec.get('game_idx')}: {len(rec['choices'])} decisions, "
              f"{int(rec.get('n_searched', 0))} searched, {int(rec.get('n_override', 0))} overridden, "
              f"winner {env.winner} (record {rec['winner']}, truncated={rec.get('truncated')})", flush=True)
        if not rec.get("truncated") and env.winner != rec["winner"]:
            fail(f"game {rec.get('game_idx')}: replay winner {env.winner} != record {rec['winner']}")
    print("dry-run ok: 2 games replay in the engine", flush=True)


def raw_picks(rec: dict, obss: list[dict], policy) -> dict[int, int]:
    """The raw policy's argmax at each searched decision, behind a shared LoopGuard.

    One batched logits forward over the game's decisions, then an in-order guard walk
    noting the played choice (exactly search_game_gen with guard_mode="shared").
    Returns {decision k: raw option}. `obss` parallels rec["choices"].
    """
    from decklight.eval.loopguard import LoopGuard

    want = {int(e["k"]) for e in rec.get("search", []) if int(e.get("seat", rec["s_seat"])) == int(rec["s_seat"])}
    lg = policy.logits({k: np.stack([o[k] for o in obss]) for k in obss[0]})
    guard = LoopGuard()
    out: dict[int, int] = {}
    for k, o in enumerate(obss):
        z = np.array(lg[k][: o["action_mask"].shape[0]], dtype=np.float64)
        z[~o["action_mask"].astype(bool)] = -np.inf
        zg, gkey = guard.mask(0, o, z)
        if k in want:
            out[k] = int(np.argmax(zg))
        guard.note(0, gkey, int(rec["choices"][k]))
    return out


def attribute(pending: list, rec: dict, cfg, engine, s_seat: int, values) -> list[tuple]:
    """Per searched decision: (entry, raw, chosen, cands, p_search, p_raw|None).

    Fresh engine replays to the positions after the chosen / raw moves; non-terminal
    positions are value-batched in one call, converted to S's view (1 - p when the next
    actor is the opponent; moves that end the game score the outcome).
    """
    from decklight.engine.cpp_env import CppEnv

    # (entry, raw, chosen, cands, obs_after_chosen|None, (obs_after_raw, actor)|None,
    #  actor_after_chosen, terminal raw p|None); obs None = the move ended the game.
    jobs: list[tuple] = []
    for e, raw, chosen, cands in pending:
        pre = [int(c) for c in rec["choices"][: int(e["k"])]]
        probe = CppEnv(engine, cfg, True)
        after_c = probe.replay_from_reset(pre + [chosen])
        if after_c is None:
            w = probe.winner
            jobs.append((e, raw, chosen, cands, None, None, 1.0 if w == s_seat else 0.0, None))
            continue
        actor_c = int(probe.player)
        obs_r = None
        p_term = None
        if raw != chosen:
            probe2 = CppEnv(engine, cfg, True)
            after_r = probe2.replay_from_reset(pre + [raw])
            if after_r is None:
                w = probe2.winner
                p_term = 1.0 if w == s_seat else 0.0
            else:
                obs_r = (after_r, int(probe2.player))
        jobs.append((e, raw, chosen, cands, after_c, obs_r, actor_c, p_term))
    batch = [o for j in jobs for o in ([j[4]] + ([j[5][0]] if j[5] else [])) if o is not None]
    vals = values({k: np.stack([o[k] for o in batch]) for k in batch[0]}) \
        if batch else np.zeros(0, np.float32)
    it = iter(to_p(float(v)) for v in vals)
    out = []
    for e, raw, chosen, cands, obs_c, obs_r, actor_c, p_term in jobs:
        if obs_c is None:  # chosen move ended the game
            out.append((e, raw, chosen, cands, actor_c, None if raw == chosen else p_term))
            continue
        p_c = next(it)
        p_c = p_c if actor_c == s_seat else 1.0 - p_c
        if raw == chosen:
            out.append((e, raw, chosen, cands, p_c, None))
        elif obs_r is None:  # raw move ended the game
            out.append((e, raw, chosen, cands, p_c, p_term))
        else:
            p_r = next(it)
            p_r = p_r if obs_r[1] == s_seat else 1.0 - p_r
            out.append((e, raw, chosen, cands, p_c, p_r))
    return out


def collect_rows(recs: list[dict], policy, values, engine, max_games: int) -> tuple[list[dict], dict]:
    """Replay games; one row per S searched decision with p_search / p_raw / outcome.

    `played_mismatch` counts decisions where the recomputed raw argmax differs from the
    logged `played` (a high rate means --ckpt is not the games' policy).
    """
    from decklight.engine.cpp_env import CppEnv

    rows: list[dict] = []
    info = {"games": 0, "skipped_draw": 0, "played_mismatch": 0, "searched": 0}
    for gi, rec in enumerate(recs):
        if gi >= max_games:
            break
        if rec.get("truncated") or rec.get("winner") not in (0, 1):
            info["skipped_draw"] += 1
            continue
        s_seat = int(rec["s_seat"])
        slog = [e for e in rec.get("search", []) if int(e.get("seat", s_seat)) == s_seat]
        if not slog:
            continue
        info["games"] += 1
        cfg = replay_config(rec)
        env = CppEnv(engine, cfg, True)
        obs = env.reset()
        obss = []
        for c in rec["choices"]:
            if obs is None:
                fail(f"game {rec.get('game_idx')}: replay ended early at decision {len(obss)}")
            obss.append(obs)
            obs = env.step(int(c))
        raws = raw_picks(rec, obss, policy)
        pending = []
        for e in slog:
            k, chosen = int(e["k"]), int(e["chosen"])
            raw = raws[k]
            if raw != int(e.get("played", raw)):
                info["played_mismatch"] += 1
            acts = obss[k]["actions"]
            cands = [(int(x), int(acts[int(x), 0]), int(acts[int(x), 2])) for x in e["cands"]]
            pending.append((e, raw, chosen, cands))
        info["searched"] += len(pending)
        y = float(rec["winner"] == s_seat)
        for e, raw, chosen, cands, p_search, p_raw in attribute(pending, rec, cfg, engine, s_seat, values):
            rows.append({
                "game": gi, "k": int(e["k"]), "turn": int(e.get("turn", 0)),
                "kind": str(e.get("kind", "?")),
                "overridden": bool(chosen != raw),
                "chosen": chosen, "played": raw, "cands": cands,
                "p_search": p_search, "p_raw": p_raw, "y": y,
                "decks": rec["decks"], "seed": int(rec["seed"]),
                "s_seat": s_seat,
            })
    return rows, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, required=True, help="games-*.jsonl file or dir of them")
    ap.add_argument("--ckpt", type=Path, default=None, help="base policy ckpt (raw argmax)")
    ap.add_argument("--value", type=Path, default=None, help="learned value ckpt scoring positions")
    ap.add_argument("--emb", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-games", type=int, default=0, help="cap games scored (0 = all)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for the CIs")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--dry-run", action="store_true", help="engine-only replay of 2 games, no network")
    args = ap.parse_args()
    recs = load_games(args.games)
    if args.dry_run:
        dry_run(recs)
        return
    if args.ckpt is None or args.value is None:
        ap.error("--ckpt and --value are required (or --dry-run)")
    if args.emb is None:
        args.emb = Path(os.environ.get("YGO_EMB") or ROOT / "data/cards/emb/qwen3-0.6b-d512.npz")
    from decklight.engine.duel import Engine
    from decklight.eval.bc_policy import BCPolicy
    from decklight.eval.value import ValueNet

    engine = Engine()
    policy = BCPolicy(args.ckpt, args.emb, device=args.device)
    value = ValueNet(args.value, args.emb, device=args.device)
    rows, info = collect_rows(recs, policy, value.values, engine,
                              args.max_games or len(recs))
    if not rows:
        fail("no scored S decisions (all games truncated/drawn or unsearched?)")
    summ = summarize(rows)
    cis = bootstrap_cis(rows, args.boot, args.seed)
    cis["bias_gap"] = bootstrap_gap_ci(rows, args.boot, args.seed + 1)
    print(f"games {summ['n_games']} ({info['games']} scored, {info['skipped_draw']} drawn/truncated "
          f"skipped), searched S decisions {info['searched']}, overridden {summ['n_overridden']}, "
          f"agreeing {summ['n_agreeing']}, played_mismatch {info['played_mismatch']}", flush=True)
    for group in ("chosen_overridden", "raw_overridden", "agreeing"):
        m = summ[group]
        print(f"{group}: n={m['n']} brier={m['brier']:.4f} {cis.get(group, {}).get('brier')} "
              f"bias={m['bias']:+.4f} {cis.get(group, {}).get('bias')}", flush=True)
    print(overtrust_verdict(summ, cis), flush=True)
    over = sorted([r for r in rows if r["overridden"] and r["p_raw"] is not None],
                  key=lambda r: r["p_search"] - r["y"], reverse=True)[:20]
    print("most over-rated overridden decisions (turn, k, cands kinds/codes, "
          "chosen/played, p_search, p_raw, outcome):", flush=True)
    for r in over:
        p_raw = "None" if r["p_raw"] is None else f"{r['p_raw']:.3f}"
        print(f"  {r['decks']} seed={r['seed']} turn={r['turn']} k={r['k']} {r['kind']} "
              f"cands={r['cands']} chosen={r['chosen']} played={r['played']} "
              f"p_search={r['p_search']:.3f} p_raw={p_raw} "
              f"y={r['y']:.0f}", flush=True)


if __name__ == "__main__":
    main()
