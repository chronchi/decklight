"""Memory v2 step B: card-knowledge memory over the gold games.

Replays every gold game in the fast C++ env and reports, over all observations, how often the
viewer remembers a hidden card. Two hand-checked events pin the semantics: a search-and-reveal
(Deck -> hand + confirm) and a bounce (face-up field -> hand) show `card_known == 1` with the
right `known_code` from the opponent's next observation, and the flag is gone once the card
leaves the hand (with resyncs at 0, disappearance implies leaving: nothing else clears a flag).
"""

from pathlib import Path

import pytest

from decklight.demos.dataset import load_records, record_config
from decklight.engine.cpp_env import CppEnv

GOLD = Path(__file__).resolve().parents[2] / "data" / "demos" / "gold"

# (file, k_obs, viewer, code): hand-checked via scratch in data/cache/mem2b/ (*.py):
# - abc: player 1 searches 76375976 (Deck -> hand) and confirms it to player 0 (trace k=20);
#   player 0's observation k=20 flags the hidden row; the card is summoned right after (k=22).
# - labrynth: 62089826 is bounced from player 0's face-up field to the hand (trace k=62);
#   player 1's observation k=63 flags it; it is summoned again before k=79.
SEARCH = ("abc-dragon-vs-exodia.jsonl", 20, 0, 76375976)
BOUNCE = ("blue-eyes-vs-labrynth.jsonl", 63, 1, 62089826)


def _env_at(engine, fname, k):
    """CppEnv replayed so the pending decision is `k` (k choices consumed); returns (env, obs)."""
    rec = load_records(GOLD / fname)[0]
    env = CppEnv(engine, record_config(rec, rec["config"]["seed"]), rec.get("auto_single", True))
    obs = env.reset()
    for choice in rec["choices"][:k]:
        obs = env.step(choice)
        assert obs is not None, f"{fname}: duel ended before decision {k}"
    return env, obs, rec


def _flagged(obs, code):
    ck, kc = obs["card_known"], obs["known_code"]
    return [i for i in range(len(ck)) if ck[i] == 1 and kc[i] == code]


def test_gold_knowledge_stats(engine):
    files = sorted(GOLD.glob("*.jsonl"))
    assert len(files) == 103, f"expected 103 gold games, got {len(files)}"
    n_obs = n_with_known = n_known_rows = resyncs = 0
    for path in files:
        env, obs, rec = _env_at(engine, path.name, 0)
        for k in range(len(rec["choices"]) + 1):
            n_obs += 1
            n = int((obs["card_known"] == 1).sum())
            n_known_rows += n
            n_with_known += n > 0
            # the remembered code is never on a currently visible row
            assert not ((obs["card_known"] == 1) & (obs["cards"][:, 0] != 0)).any(), path.name
            # own-known flags only sit on own rows
            assert (obs["cards"][obs["card_known"] == 2, 3] == 0).all(), path.name  # controller == me
            if k < len(rec["choices"]):
                obs = env.step(rec["choices"][k])
                if obs is None:
                    break
        resyncs += env.knowledge_resyncs
    assert resyncs == 0, f"knowledge resyncs: {resyncs}"
    assert n_with_known > 0, "no remembered card anywhere: tracking is dead"
    print(f"obs {n_obs}, with known>0: {n_with_known} ({n_with_known / n_obs:.3f}), "
          f"mean known rows/obs {n_known_rows / n_obs:.4f}, resyncs {resyncs}")


@pytest.mark.parametrize("fname,k_obs,viewer,code", [SEARCH, BOUNCE])
def test_hand_checked_events(engine, fname, k_obs, viewer, code):
    env, obs, rec = _env_at(engine, fname, k_obs)
    assert env.player == viewer
    rows = _flagged(obs, code)
    assert len(rows) == 1, f"{fname} k={k_obs}: flagged rows for {code}: {rows}"
    assert int(obs["cards"][rows[0]][0]) == 0, "remembered row must be hidden (code 0)"
    assert int(obs["cards"][rows[0]][1]) == 2, "remembered row must be an opponent hand row"
    # the flag is gone from the viewer's observations once the card leaves the hand
    gone = None
    for k, choice in enumerate(rec["choices"][k_obs:], start=k_obs):
        obs = env.step(choice)
        if obs is None:
            break
        if env.player == viewer and not _flagged(obs, code):
            gone = k + 1
            break
    assert gone is not None, f"{fname}: {code} never leaves {viewer}'s knowledge"
    assert env.knowledge_resyncs == 0
    print(f"{fname}: code {code} flagged at k={k_obs} (viewer {viewer}), gone at k={gone}")
