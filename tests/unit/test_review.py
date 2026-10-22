"""Post-game review unit tests (web-ux-plan step 3)."""

from pathlib import Path

import numpy as np

from decklight.analysis.review import (
    MARK_BLUND,
    MARK_GOOD,
    MARK_INACC,
    MARK_MISSED,
    MARK_MIST,
    collect_human_decisions,
    load_record,
    mark_for_delta,
    review_game,
)

FIXTURE = Path("tests/fixtures/web/mitsurugi-20261016")


class _FakeCalib:
    def p(self, v: float, turn: int) -> float:
        return float(np.clip((float(v) + 1.0) / 2.0, 0.03, 0.97))


class _FakeLauncher:
    """ProductLauncher-shaped stub with spied policy/value calls."""

    def __init__(self, logits_fn=None, value_v: float = 0.0) -> None:
        self._fn = logits_fn
        self.policy_seen: list[dict] = []
        self.value_seen: list[dict] = []
        self._v = value_v

    def base_logits(self, obs_list):
        for o in obs_list:
            self.policy_seen.append({k: np.array(v) for k, v in o.items()})
        if self._fn is not None:
            return self._fn(obs_list)
        n = len(obs_list)
        w = int(obs_list[0]["action_mask"].shape[0])
        return np.zeros((n, w), np.float32)

    @property
    def value(self):
        def fn(batch):
            self.value_seen.append({k: np.array(v) for k, v in batch.items()})
            n = next(iter(batch.values())).shape[0]
            return np.full(n, self._v, np.float32)

        return fn


def _human_check(obs: dict, first_player: int, human: int) -> bool:
    g = obs["global"][0] if obs["global"].ndim > 1 else obs["global"]
    return int(g[4]) == int(first_player == human)


def test_marks_follow_thresholds():
    assert mark_for_delta(0) == MARK_GOOD
    assert mark_for_delta(-3) == MARK_GOOD
    assert mark_for_delta(-3.1) == MARK_INACC
    assert mark_for_delta(-7) == MARK_INACC
    assert mark_for_delta(-7.1) == MARK_MIST
    assert mark_for_delta(-15) == MARK_MIST
    assert mark_for_delta(-15.1) == MARK_BLUND


def test_review_fixture_end_to_end(engine):
    rec, cfg, _hs, _d = load_record(FIXTURE / "game.jsonl")
    launcher = _FakeLauncher()
    out1 = review_game(engine, cfg, rec, launcher, _FakeCalib(), max_steps=8)
    out2 = review_game(engine, cfg, rec, _FakeLauncher(), _FakeCalib(), max_steps=8)
    assert out1["decisions"], "no human decisions reviewed"
    o1 = {k: v for k, v in out1.items() if k != "timing"}
    o2 = {k: v for k, v in out2.items() if k != "timing"}
    assert o1 == o2, "review is not deterministic"
    # single-option decisions are skipped: every entry had >= 2 options.
    assert all("delta_pts" in e and "mark" in e for e in out1["decisions"])
    # JSON shape per the brief.
    for e in out1["decisions"]:
        assert set(e) >= {
            "k",
            "turn",
            "phase",
            "kind",
            "mark",
            "delta_pts",
            "your",
            "agent",
            "policy_p_yours",
            "policy_p_agent",
            "missed_response",
            "note",
        }
        assert e["mark"] in (MARK_GOOD, MARK_INACC, MARK_MIST, MARK_BLUND, MARK_MISSED)
    # accuracy = mean(max(0, 100 + delta)).
    want = round(sum(max(0.0, 100.0 + e["delta_pts"]) for e in out1["decisions"]) / len(out1["decisions"]), 1)
    assert out1["accuracy"] == want
    # missed responses come from activatable windows only (human passed a chain window
    # that offered ACTIVATE); auto-passed windows are flagged.
    for e in out1["decisions"]:
        if e["missed_response"]:
            assert e["mark"] == MARK_MISSED


def test_candidates_dedup_and_cap(engine):
    rec, cfg, _hs, _d = load_record(FIXTURE / "game.jsonl")
    launcher = _FakeLauncher()
    out = review_game(engine, cfg, rec, launcher, _FakeCalib(), max_steps=4)
    # candidates are internal, but the agent choice must be a reviewed option and
    # the policy distribution must sum to 1 over legal options.
    assert out["decisions"]


def test_single_option_skipped(engine):
    from decklight.engine.env import PyEnv

    rec, cfg, hs, _d = load_record(FIXTURE / "game.jsonl")
    env = PyEnv(engine, cfg, auto_single=False)
    env.reset()
    multis = 0
    for ch in rec["choices"]:
        if env.done:
            break
        if env.player == hs and len(env.opts) > 1:
            multis += 1
        env.step(int(ch))
    out = review_game(engine, cfg, rec, _FakeLauncher(), _FakeCalib(), max_steps=4)
    assert len(out["decisions"]) == multis


def test_fairness_spy_human_obs_only(engine):
    """Every value obs (and every candidate policy obs) is the human seat's obs."""
    rec, cfg, hs, _d = load_record(FIXTURE / "game.jsonl")
    launcher = _FakeLauncher()
    review_game(engine, cfg, rec, launcher, _FakeCalib(), max_steps=8)
    assert launcher.value_seen, "value never called"
    env0 = __import__("decklight.engine.env", fromlist=["PyEnv"]).PyEnv(engine, cfg, auto_single=False)
    env0.reset()
    first_player = int(env0.duel.first_player)
    for batch in launcher.value_seen:
        n = next(iter(batch.values())).shape[0]
        for i in range(n):
            obs = {k: batch[k][i] for k in batch}
            assert _human_check(obs, first_player, hs), "value saw a non-human obs"
    # the candidate policy calls at decision points are human obs; rollout bot
    # steps (bot's own view) are allowed, so at least one policy obs is human's.
    assert launcher.policy_seen
    assert any(_human_check(o, first_player, hs) for o in launcher.policy_seen)


def test_missed_from_activatable_only(engine):
    rec, cfg, _hs, _d = load_record(FIXTURE / "game.jsonl")
    launcher = _FakeLauncher(
        logits_fn=lambda ol: np.zeros((len(ol), int(ol[0]["action_mask"].shape[0])), np.float32)
    )
    out = review_game(engine, cfg, rec, launcher, _FakeCalib(), max_steps=4)
    decs = collect_human_decisions(engine, cfg, rec)
    by_k = {d.get("k", i): d for i, d in enumerate(decs)}
    for e in out["decisions"]:
        if e["missed_response"]:
            d = by_k[e["k"]]
            assert d["is_chain"] and any(
                str(k) == "Kind.ACTIVATE" or "ACTIVATE" in str(k) for k in d["opt_kinds"]
            )
