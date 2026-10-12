"""Mid-game turn planning from recorded states (scripts/plan_turns.py; no LLM)."""

import importlib.util
import json
import sys
from pathlib import Path

from ygo_zero.demos.dataset import iter_samples, sample_kind
from ygo_zero.demos.game import legal_actions
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.plan import card_sheet
from ygo_zero.engine import constants as C
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "decks"


def _load():
    spec = importlib.util.spec_from_file_location("plan_turns", ROOT / "scripts" / "plan_turns.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plan_turns"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_record(engine, decks, seed=0, max_turn=4) -> dict:
    """A short real game of the fixture decks with an always-option-0 policy."""
    d0, d1 = decks["BlueEyes"], decks["Branded"]
    cfg = DuelConfig(d0, d1, seed=seed)
    env = CppEnv(engine, cfg)
    obs = env.reset()
    choices, actors, turns = [], [], []
    for _ in range(2000):
        if obs is None or int(obs["global"][2]) > max_turn:
            break
        if int(obs["action_mask"].sum()) == 0:
            break
        actors.append(env.player)
        turns.append(int(obs["global"][2]))
        choices.append(0)
        obs = env.step(0)
    assert any(t >= 2 for t in turns), "option-0 play must reach turn 2"
    return {
        "config": {"deck0_main": cfg.main_of(0), "deck0_extra": list(d0.extra),
                   "deck1_main": cfg.main_of(1), "deck1_extra": list(d1.extra),
                   "seed": seed, "hand": 5, "draw": 1, "shuffle": False},
        "auto_single": True, "choices": choices, "actors": actors, "turns": turns,
        "decks": [d0.name, d1.name], "game_id": f"t{seed}", "line_id": f"t{seed}",
    }


def _sample_kwargs():
    return {"pool_dir": FIXTURES, "eval_sets": (set(), set())}


def _obs_at(engine, state):
    env = CppEnv(engine, state["config"], state["auto_single"])
    obs = env.reset()
    for c in state["prefix"]:
        obs = env.step(int(c))
    return obs, env


def test_bucket_of():
    mod = _load()
    assert mod.bucket_of(2, False, 8000, 8000, 30) == "second"
    assert mod.bucket_of(2, True, 8000, 8000, 30) is None
    assert mod.bucket_of(4, True, 8000, 8000, 30) == "mid"
    assert mod.bucket_of(7, True, 8000, 8000, 30) is None
    assert mod.bucket_of(5, True, 2000, 8000, 30) == "end"
    assert mod.bucket_of(5, True, 8000, 1000, 30) == "end"
    assert mod.bucket_of(3, False, 8000, 8000, 8) == "end"
    assert mod.bucket_of(4, True, 2000, 2000, 30) == "end"  # end wins over mid


def test_sampling_only_open_mp1_idle(engine, decks):
    mod = _load()
    recs = [_tiny_record(engine, decks, s) for s in range(3)]
    states = mod.sample_states(engine, recs, seed=0, per_bucket=10, **_sample_kwargs())
    assert states, "option-0 fixture games must yield MP1 states"
    seen_game_bucket = set()
    for s in states:
        assert (s["game_id"], s["bucket"]) not in seen_game_bucket, "one state per game per bucket"
        seen_game_bucket.add((s["game_id"], s["bucket"]))
        obs, _ = _obs_at(engine, s)
        g = obs["global"]
        assert bool(int(g[5])), "own turn"
        assert int(g[3]) == 0x4, "Main Phase 1"
        assert int(g[21]) == C.MSG_SELECT_IDLECMD and int(g[20]) == 0, "open idle, chain empty"
        went_first = bool(int(g[4]))
        assert s["bucket"] == mod.bucket_of(int(g[2]), went_first, int(g[0]), int(g[1]), int(g[6]))
    # the acting deck is the record deck at the acting seat
    for s in states:
        env = CppEnv(engine, s["config"], s["auto_single"])
        obs = env.reset()
        for c in s["prefix"]:
            obs = env.step(int(c))
        assert obs is not None
        assert s["decks"][env.player] == s["deck"]


def test_sampling_caps_and_filters(engine, decks):
    mod = _load()
    recs = [_tiny_record(engine, decks, s) for s in range(3)]
    one = mod.sample_states(engine, recs, seed=0, per_bucket=1, **_sample_kwargs())
    counts: dict = {}
    for s in one:
        counts[s["deck"], s["bucket"]] = counts.get((s["deck"], s["bucket"]), 0) + 1
    assert all(v <= 1 for v in counts.values())
    mid = mod.sample_states(engine, recs, seed=0, per_bucket=10, buckets=("mid",),
                            **_sample_kwargs())
    assert mid and all(s["bucket"] == "mid" for s in mid)
    # either deck eval-only skips the whole game
    evald = mod.sample_states(engine, recs, seed=0, per_bucket=10,
                              pool_dir=FIXTURES, eval_sets=({"Branded"}, set()))
    assert evald == []
    # allow-list keeps only that acting deck
    allow = mod.sample_states(engine, recs, seed=0, per_bucket=10, decks_allow={"BlueEyes"},
                              **_sample_kwargs())
    assert allow and all(s["deck"] == "BlueEyes" for s in allow)


def _summon_line_json(names: CardNames, action: str, card: str) -> str:
    return json.dumps({"lines": [{
        "title": "fake", "hand": [], "requires": [],
        "steps": [{"i": 0, "action": action, "card": card, "materials": [], "targets": [],
                   "results": [card], "text": "fake"}],
        "end_board": [], "end_board_text": "", "going": "unknown"}]})


def _fake_many(names, action: str, card: str):
    body = _summon_line_json(names, action, card)
    return lambda prompts: [body for _ in prompts]


def _pick_summon(mod, engine, names, state):
    """A (summon-action, card) the engine offers right now, from the legal list."""
    obs, _ = _obs_at(engine, state)
    legal, _ = legal_actions(obs, names.names)
    for action, tag in (("normal_summon", "Normal Summon"), ("special_summon", "Special Summon")):
        for part in legal.split(";"):
            if part.strip().startswith(tag + ":"):
                first = part.split(":", 1)[1].split(",")[0].strip()
                if first and first != "none":
                    return action, first
    raise AssertionError(f"no summon offered at {state['deck']}/{state['bucket']}: {legal}")


def test_verified_record_replays_prefix_weightless(engine, decks, tmp_path):
    mod = _load()
    names = CardNames()
    recs = [_tiny_record(engine, decks, s) for s in range(6)]
    states = mod.sample_states(engine, recs, seed=0, per_bucket=10, **_sample_kwargs())
    assert states
    pool_decks = {"BlueEyes": decks["BlueEyes"], "Branded": decks["Branded"]}
    sheets = {n: card_sheet(engine, d, names) for n, d in pool_decks.items()}
    picked = None
    for s in states:
        try:
            picked = (s, *_pick_summon(mod, engine, names, s))
            break
        except AssertionError:
            continue
    assert picked is not None, "some sampled state must offer a summon"
    s, action, card = picked
    results = mod.plan_states(engine, names, sheets, pool_decks, [s],
                              _fake_many(names, action, card), budget=4000, repairs=0,
                              model="fake")
    rec, status = results[0]
    assert status == "ok" and rec is not None, status
    assert len(rec["labels"]) == len(rec["choices"])
    assert rec["labels"][: s["k"]] == ["prefix"] * s["k"]
    assert len(rec["actors"]) == len(rec["choices"]) == len(rec["turns"])
    p = tmp_path / "turn.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    samples = list(iter_samples(engine, p))
    assert samples, "the verified plan must yield BC samples"
    assert all(m["label"] != "prefix" for _, _, _, m in samples), "prefix decisions do not train"


def test_sample_kind_turns():
    assert sample_kind("data/demos/turns/mid/x.jsonl", "stated") == "turn_mid"
    assert sample_kind("data/demos/turns/end/y.jsonl", "prefix") == "turn_end"
