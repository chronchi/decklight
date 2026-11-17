"""Step 1b of the opponent-turn plan (docs/plans/2026-11-03-opp-turn-interaction.md): going-second starts for the
combo-line opponents. One game record per (candidate deck, our deck, seed): our deck plays turn 1 with P (argmax) while
the candidate passes, and the record stops at the candidate's first turn-2 Main Phase 1 idle (plus that one choice, so
`plan_turns.sample_states` finds the state in its "second" bucket). `scripts/plan_turns.py --buckets second` then asks
the LLM for the candidate's turn 2 into our board and verifies it in the engine from that exact state (ADR 0003).
These records are the league's turn-2 starts: line replay from our real P board, our policy answers in the windows.

    uv run python scripts/audit/second_starts.py --decks branded --seeds 2 --out /tmp/second-starts.jsonl
    uv run python scripts/audit/second_starts.py --admit config/league/combo_pool_v2.json --seeds 6 \
        --out data/demos/second-starts/starts.jsonl

Exits 3 when no record was written or a record does not reach the state on replay (CppEnv, as plan_turns replays).
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import combo_pool as CP  # noqa: E402
import plan_turns as PTN  # noqa: E402

from decklight.decks import load_ydk  # noqa: E402
from decklight.demos.dataset import record_config  # noqa: E402
from decklight.demos.game import Responder  # noqa: E402
from decklight.engine import constants as C  # noqa: E402
from decklight.engine.cpp_env import CppEnv  # noqa: E402
from decklight.engine.duel import DuelConfig, Engine  # noqa: E402
from decklight.engine.env import PyEnv  # noqa: E402
from decklight.eval.bc_policy import BCPolicy  # noqa: E402
from decklight.search.product import resolve_product_paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OURS = ("decks/custom/skydeck-tcg2026-10.ydk", "decks/custom/light-magistus-f10-tcg2026-10.ydk")


def is_turn2_start(obs: dict, player: int) -> bool:
    g = obs["global"]
    return (player == 1 and int(g[2]) == 2 and int(g[3]) == PTN.PHASE_MAIN1 and int(g[21]) == C.MSG_SELECT_IDLECMD
            and int(g[20]) == 0)  # fmt: skip


def make_start(engine, pol: BCPolicy, ours_name: str, ours, name: str, theirs, seed: int) -> dict | None:
    """Our turn 1 by P (argmax), the candidate passive; stop after the candidate's first turn-2 MP1 choice (P's)."""
    if getattr(pol, "guard", None) is not None:
        pol.guard.reset()
    cfg = DuelConfig(deck0=ours, deck1=theirs, seed=seed)
    env = PyEnv(engine, cfg, auto_single=True)
    obs, passive, choices = env.reset(), Responder(seed), []
    while obs is not None and len(choices) < CP.MAX_DECISIONS and int(env.duel.turn) <= 2:
        start = is_turn2_start(obs, env.player)
        a = CP._act(pol, obs) if (env.player == 0) == (int(env.duel.turn) == 1) else passive.passive(obs)
        choices.append(int(a))
        if start:
            return {"game_id": f"s2-{Path(ours_name).stem}-{name}-{seed}", "decks": [Path(ours_name).stem, name],
                    "config": PTN.config_dict_of(cfg), "auto_single": True, "choices": choices,
                    "source": "second_starts"}  # fmt: skip
        obs = env.step(a)
    return None


def replays_to_start(engine, rec: dict) -> bool:
    env = CppEnv(engine, record_config(rec), rec["auto_single"])
    obs = env.reset()
    for c in rec["choices"][:-1]:
        if obs is None:
            return False
        obs = env.step(int(c))
    return obs is not None and is_turn2_start(obs, env.player)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", default="")
    ap.add_argument("--admit", type=Path, help="combo_admit output: its going_second_candidate decks")
    ap.add_argument("--seeds", type=int, default=6, help="starts per (candidate, our deck)")
    ap.add_argument("--ours", default=",".join(OURS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    names = [n for n in a.decks.split(",") if n]
    if a.admit:
        names += [d["deck"] for d in json.loads(a.admit.read_text())["decks"] if d.get("going_second_candidate")]
    hold, new = PTN.eval_only_decks()
    skipped = sorted(n for n in names if n in hold or n in new)
    names = [n for n in dict.fromkeys(names) if n not in hold and n not in new]
    if skipped:
        print(f"eval-only decks skipped (never trained on): {skipped}", flush=True)
    engine = Engine()
    ckpt, _value, emb = resolve_product_paths(None, None, None)
    pol = BCPolicy(ckpt, emb, device="cpu")
    ours = [(o, load_ydk(ROOT / o)) for o in a.ours.split(",") if o]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_bad, missing = 0, 0, []
    with a.out.open("w") as fo:
        for name in names:
            theirs = load_ydk(ROOT / "decks/pool" / f"{name}.ydk")
            got = 0
            for o_name, o_deck in ours:
                for s in range(a.seeds):
                    seed = 91000 + zlib.crc32(f"{name}/{o_name}".encode()) % 1000 * 10 + s
                    rec = make_start(engine, pol, o_name, o_deck, name, theirs, seed)
                    if rec is None:
                        missing.append(f"{name}/{Path(o_name).stem}/{seed}")
                        continue
                    if not replays_to_start(engine, rec):
                        n_bad += 1
                        print(f"ALERT second_starts: {rec['game_id']} does not replay to the turn-2 start", flush=True)
                        continue
                    fo.write(json.dumps(rec) + "\n")
                    got += 1
            n_ok += got
            print(f"{name:44s} {got} starts", flush=True)
    if missing:
        print(f"no turn-2 start (duel ended on turn 1 or cut): {len(missing)} {missing[:10]}", flush=True)
    print(f"{n_ok} starts over {len(names)} decks -> {a.out}", flush=True)
    if n_ok == 0 or n_bad:
        print(f"ALERT second_starts: {n_ok} written, {n_bad} failed replay", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
