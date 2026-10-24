"""Client parity audit (plan P0): how far is the stream view from the engine seat?

Replays search_match games twice: once in `PyEnv(auto_single=True)` (the truth) and once
in a host-style `PyEnv(auto_single=False)` whose exact `DuelHost` packet stream (START,
filtered messages, before/after refreshes via `filter.refresh_packet`) feeds a
`StreamView` per audited seat. At every seat decision it compares option lists and every
obs key, plus raw-policy argmax agreement with `--ckpt`.

Exit 3 when option lists ever differ or a replay's winner differs from the record's.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk
from decklight.edopro import filter as F
from decklight.edopro import protocol as P
from decklight.edopro.product_bot import product_to_host
from decklight.engine import constants as C
from decklight.engine.duel import DuelConfig, Engine
from decklight.engine.env import PyEnv
from decklight.play.stream import StreamView

KEYS = ("cards", "card_mask", "card_known", "known_code", "global", "chain", "chain_mask",
        "actions", "action_mask", "history", "history_mask", "hist_long", "hist_long_mask")


def find_deck(name: str, extra_dir: str | None) -> Path:
    dirs = [ROOT / "decks" / "pool", ROOT / "decks" / "gauntlet", ROOT / "decks" / "newcards"]
    if extra_dir:
        dirs.insert(0, Path(extra_dir))
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.glob("*.ydk"):
            if p.stem.lower() == name.lower():
                return p
    raise FileNotFoundError(f"deck {name!r} not found in {[str(d) for d in dirs]}")


def deck_candidates(name: str, extra_dir: str | None, ygoagent: bool) -> list[Path]:
    """Every dir holding `name` (case-insensitive), heuristic order first.

    ygo-agent match records name gauntlet lists; pool/newcards self-play names pool lists.
    """
    dirs = [ROOT / "decks" / "gauntlet", ROOT / "decks" / "pool", ROOT / "decks" / "newcards"]
    if not ygoagent:
        dirs = [ROOT / "decks" / "pool", ROOT / "decks" / "newcards", ROOT / "decks" / "gauntlet"]
    if extra_dir:
        dirs.insert(0, Path(extra_dir))
    out = []
    for d in dirs:
        if d.is_dir():
            out += [p for p in d.glob("*.ydk") if p.stem.lower() == name.lower()]
    return out


def resolve_cfg(engine: Engine, rec: dict, extra_dir: str | None) -> DuelConfig:
    """The record's DuelConfig, validated by exact replay (choices, actors, winner).

    games records carry no config (decks by name, default options); several deck dirs
    can hold the same stem, so every candidate combo is replayed and the first exact
    one wins. Fail loud when nothing replays: the alternative is auditing a wrong game.
    """
    ygoagent = "ya_decisions" in rec or "deck_s" in rec
    cands = [deck_candidates(n, extra_dir, ygoagent) or [None] for n in rec["decks"]]
    if any(p is None for c in cands for p in c):
        raise FileNotFoundError(f"deck {[n for n in rec['decks']]} not found "
                                f"(gauntlet/pool/newcards{(' + ' + extra_dir) if extra_dir else ''})")
    tried = []
    for p0 in cands[0]:
        for p1 in cands[1]:
            cfg = DuelConfig(load_ydk(p0), load_ydk(p1), seed=rec["seed"])
            tried.append(f"{p0.parent.name}/{p0.stem}+{p1.parent.name}/{p1.stem}")
            env = PyEnv(engine, cfg, auto_single=True)
            try:
                obs = env.reset()
                for c in rec["choices"]:
                    if obs is None:
                        break
                    obs = env.step(int(c))
                actors = [e.actor for e in env.history]
                if obs is None and env.done and env.winner == rec["winner"] and actors == rec["actors"]:
                    if len(tried) > 1:
                        print(f"note: game {rec.get('game_idx', '?')} decks resolved as {tried[-1]}",
                              flush=True)
                    return cfg
            except (IndexError, ValueError):
                continue
    raise RuntimeError(f"ALERT no deck combo replays game {rec.get('game_idx', '?')}: tried {tried}")


def load_records(pattern: str, max_games: int) -> list[dict]:
    recs = []
    for path in sorted(glob.glob(pattern)):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    out = [r for r in recs if not r.get("truncated")]
    skipped = len(recs) - len(out)
    if skipped:
        print(f"note: skipped {skipped} truncated records", flush=True)
    return out[:max_games]


def truth_replay(engine: Engine, cfg: DuelConfig, rec: dict, seats: list[int]):
    """Replay record choices in auto_single mode; obses at each audited seat's decisions.

    One obs per outer decision: intermediate multi-step states (same `Decision`, step > 0,
    which `_settle` also returns) are skipped, matching the host replay's first-pick rule.
    """
    env = PyEnv(engine, cfg, auto_single=True)
    obs = env.reset()
    obses: dict[int, list[dict]] = {s: [] for s in seats}
    last_dec = None
    if obs is not None and env.player in obses:
        obses[env.player].append(obs)
        last_dec = env.dec
    for c in rec["choices"]:
        obs = env.step(int(c))
        if obs is not None and env.player in obses and env.dec is not last_dec:
            obses[env.player].append(obs)
            last_dec = env.dec
    return env, obses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", required=True, help="glob of search_match games-*.jsonl")
    ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--seats", default="0,1", help="viewer seats to audit, e.g. 0,1")
    ap.add_argument("--ckpt", default=None, help="policy checkpoint for agreement (default: product)")
    ap.add_argument("--policy-default", action="store_true", help="use the product default ckpt")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--policy-sample", type=int, default=2000, help="max decisions for agreement")
    ap.add_argument("--deck-dir", default=None, help="extra dir to look deck names up in first")
    ap.add_argument("--json", default=None, help="write the JSON report here")
    a = ap.parse_args()

    seats = [int(s) for s in a.seats.split(",") if s.strip() != ""]
    recs = load_records(a.games, a.max_games)
    if not recs:
        sys.exit(f"no records in {a.games}")
    engine = Engine()

    key_match: Counter[str] = Counter()
    key_total = 0
    hist_split = Counter()  # own_match/own_mismatch/opp_match/opp_mismatch rows
    opt_match = 0
    opt_total = 0
    hidden_code = 0  # decisions whose actions differ only by filter-hidden codes
    resp_match = 0
    resp_total = 0
    resp_bad: list[tuple] = []
    winner_bad: list[int] = []
    replay_bad: list[int] = []
    opt_bad: list[tuple] = []
    agree: list[tuple[dict, dict, int]] = []  # (truth obs, stream obs, n_options) for policy check
    decisions = 0

    for rec in recs:
        game = rec.get("game_idx", "?")
        try:
            cfg = resolve_cfg(engine, rec, a.deck_dir)
        except (FileNotFoundError, RuntimeError) as e:
            print(str(e), flush=True)
            replay_bad.append(game)
            continue

        env_t, truth_obs = truth_replay(engine, cfg, rec, seats)
        if not env_t.done or env_t.winner != rec["winner"]:
            print(f"ALERT game {game}: truth replay done={env_t.done} winner={env_t.winner}, "
                  f"record winner {rec['winner']}", flush=True)
            replay_bad.append(game)
            continue

        host_choices = product_to_host(engine, cfg, rec["choices"])
        env = PyEnv(engine, cfg, auto_single=False)
        env.reset()
        streams = {s: StreamView({"main": list(cfg.deck0.main if s == 0 else cfg.deck1.main),
                                           "extra": list(cfg.deck0.extra if s == 0 else cfg.deck1.extra)},
                                          engine.cards, seat=s) for s in seats}
        n0, e0 = len(cfg.deck0.main), len(cfg.deck0.extra)
        n1, e1 = len(cfg.deck1.main), len(cfg.deck1.extra)
        for s, st in streams.items():
            st.feed(P.build_start(cfg.lp, n0, e0, n1, e1, s))
            for seat in (0, 1):
                p = F.refresh_packet(env.duel, ("loc", seat, C.LOCATION_EXTRA), s)
                if p is not None:
                    st.feed(p)

        sent = 0
        ti = {s: 0 for s in seats}  # consumed truth obses per seat

        def pump(_env=env, _streams=streams) -> None:
            nonlocal sent
            new = _env.duel.trace[sent:]
            sent = len(_env.duel.trace)
            for raw in new:
                for s, st in _streams.items():
                    for ref in F.before_refreshes(raw):
                        p = F.refresh_packet(_env.duel, ref, s)
                        if p is not None:
                            st.feed(p)
                    msg = F.filtered_message(raw, s)
                    if msg is not None:
                        st.feed(msg)
                    for ref in F.after_refreshes(raw):
                        p = F.refresh_packet(_env.duel, ref, s)
                        if p is not None:
                            st.feed(p)

        i, last_dec = 0, None
        while not env.done:
            pump()
            if env.done:
                break
            player, opts = env.player, env.opts
            s = player if player in streams else None
            if s is not None:
                st = streams[s]
                if len(opts) == 1:
                    if st._resp is None:
                        opt_bad.append((game, s, i, "singleton not auto-taken"))
                else:
                    if st.decision is None or not st.needs_answer:
                        opt_bad.append((game, s, i, "stream has no pending decision"))
                    elif env.dec is not last_dec:
                        # First pick of a multi-pick decision: the outer decision the
                        # truth replay also saw (singletons never reach either side).
                        so, ho = st.opts, list(opts)
                        same = len(so) == len(ho) and all(
                            (x.kind, x.desc, x.arg,
                             None if x.card is None else (x.card.controller, x.card.location, x.card.sequence))
                            == (y.kind, y.desc, y.arg,
                                None if y.card is None else (y.card.controller, y.card.location, y.card.sequence))
                            for x, y in zip(so, ho)
                        )
                        if not same:
                            opt_bad.append((game, s, i, "raw option tuples differ"))
                        # The obs actions arrays are the effective option lists (codes for
                        # hidden cards map to 0 on both sides); compare those exactly.
                        if ti[s] >= len(truth_obs[s]):
                            print(f"ALERT game {game} seat {s}: more host decisions than truth "
                                  f"({ti[s]} >= {len(truth_obs[s])})", flush=True)
                            replay_bad.append(game)
                            break
                        ta = truth_obs[s][ti[s]]
                        sa = st.obs()
                        opt_total += 1
                        acts_same = ta["actions"].shape == sa["actions"].shape and bool(
                            (ta["actions"] == sa["actions"]).all()
                            and (ta["action_mask"] == sa["action_mask"]).all()
                        )
                        opt_match += acts_same
                        if not acts_same:
                            # Fail loud unless the ONLY differences are hidden codes: the
                            # filter zeroes uncontrolled cards for the chooser (blind
                            # picks), so stream 0 vs truth code is a genuine client gap.
                            n = min(ta["actions"].shape[0], sa["actions"].shape[0])
                            hidden_only = ta["actions"].shape == sa["actions"].shape and all(
                                ta["actions"][r][2] != 0 and sa["actions"][r][2] == 0
                                and (ta["actions"][r][[0, 1, 3, 4, 5, 6, 7, 8, 9]]
                                     == sa["actions"][r][[0, 1, 3, 4, 5, 6, 7, 8, 9]]).all()
                                or (ta["actions"][r] == sa["actions"][r]).all()
                                for r in range(n)
                            )
                            if hidden_only:
                                hidden_code += 1
                            else:
                                opt_bad.append((game, s, i, "actions arrays differ"))
                        key_total += 1
                        decisions += 1
                        for k in KEYS:
                            if ta[k].shape == sa[k].shape and bool((ta[k] == sa[k]).all()):
                                key_match[k] += 1
                        for hk in ("history", "hist_long"):
                            n = int(ta[hk + "_mask"].sum())
                            for r in range(n):
                                actor = int(ta[hk][r][10])  # actor ^ me (0 = own row)
                                same_row = bool((ta[hk][r] == sa[hk][r]).all())
                                hist_split[("own" if actor == 0 else "opp")
                                           + ("_match" if same_row else "_mismatch")] += 1
                        if (a.ckpt is not None or a.policy_default) and len(agree) < a.policy_sample:
                            agree.append((ta, sa, len(ho)))
                        ti[s] += 1
            last_dec = env.dec
            c = int(host_choices[i])
            if s is not None and streams[s].needs_answer:
                # Take the current pick (singletons inside a decision resolve here,
                # possibly completing the stream side before the host's step does).
                streams[s].take(c)
            n_resp = len(env.duel.responses)
            env.step(c)
            i += 1
            if s is not None and len(env.duel.responses) > n_resp:
                # The host answered the core for seat s: the stream must hold the same
                # response (it takes every seat-s pick, auto-taking singletons).
                resp_total += 1
                same_resp = streams[s]._resp == env.duel.responses[-1]
                resp_match += same_resp
                if not same_resp:
                    resp_bad.append((game, s, i))
        pump()

        host_winner = env.duel.winner
        if host_winner != rec["winner"]:
            print(f"ALERT game {game}: host replay winner {host_winner} != record {rec['winner']}",
                  flush=True)
            replay_bad.append(game)
            continue
        for s, st in streams.items():
            if st.winner != rec["winner"]:
                print(f"ALERT game {game} seat {s}: stream winner {st.winner} != record {rec['winner']}",
                      flush=True)
                winner_bad.append((game, s))
        for s in seats:
            if ti[s] != len(truth_obs[s]):
                print(f"ALERT game {game} seat {s}: consumed {ti[s]} of {len(truth_obs[s])} truth obses",
                      flush=True)
                replay_bad.append(game)

    print(f"\ndecisions compared: {decisions} over {len(recs)} records, keys {dict(key_match)}/{key_total}")
    table = [("key", "exact/total", "rate")]
    for k in KEYS:
        table.append((k, f"{key_match[k]}/{key_total}",
                      f"{key_match[k] / key_total:.4f}" if key_total else "n/a"))
    table.append(("options(actions)", f"{opt_match}/{opt_total}",
                  f"{opt_match / opt_total:.4f}" if opt_total else "n/a"))
    table.append(("options(hidden-code)", f"{hidden_code}/{opt_total}",
                  f"{hidden_code / opt_total:.4f}" if opt_total else "n/a"))
    table.append(("response-bytes", f"{resp_match}/{resp_total}",
                  f"{resp_match / resp_total:.4f}" if resp_total else "n/a"))
    w = max(len(r[0]) for r in table)
    for r in table:
        print(f"  {r[0]:<{w}}  {r[1]:>14}  {r[2]}")
    print(f"history rows: {dict(hist_split)}")
    if opt_bad[:5]:
        print(f"option problems (first 5): {opt_bad[:5]}")
    if resp_bad[:5]:
        print(f"response-byte mismatches (first 5): {resp_bad[:5]}")

    agreement = None
    if a.ckpt is not None or a.policy_default:
        from decklight.eval.bc_policy import BCPolicy
        from decklight.search.product import resolve_product_paths

        ckpt_p, _v, emb_p = resolve_product_paths(a.ckpt, None, a.emb)
        pol = BCPolicy(ckpt_p, emb_p, threads=2)
        n = len(agree)
        hit = 0
        for ta, sa, _m in agree:
            zb = np.asarray(pol.logits({k: v[None] for k, v in ta.items()}))[0]
            zs = np.asarray(pol.logits({k: v[None] for k, v in sa.items()}))[0]
            m = ta["action_mask"].astype(bool)
            zb, zs = np.array(zb, float), np.array(zs, float)
            zb[~m], zs[~m] = -np.inf, -np.inf
            hit += int(np.argmax(zb) == np.argmax(zs))
        agreement = hit / n if n else 0.0
        print(f"policy agreement: {hit}/{n} = {agreement:.4f} ({ckpt_p})")

    report = {
        "records": len(recs), "decisions": decisions,
        "keys": {k: {"match": int(key_match[k]), "total": key_total} for k in KEYS},
        "options": {"match": opt_match, "total": opt_total, "hidden_code": hidden_code},
        "response_bytes": {"match": int(resp_match), "total": int(resp_total),
                           "mismatches": resp_bad[:20]},
        "history_rows": dict(hist_split),
        "option_problems": opt_bad[:20],
        "winner_mismatches": winner_bad,
        "replay_mismatches": replay_bad,
        "policy_agreement": agreement,
        "policy_n": len(agree),
    }
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
        print(f"wrote {a.json}")

    if opt_bad or winner_bad or replay_bad or resp_bad:
        print(f"ALERT parity failed: {len(opt_bad)} option, {len(resp_bad)} response, "
              f"{len(winner_bad)} winner, {len(replay_bad)} replay problems", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
