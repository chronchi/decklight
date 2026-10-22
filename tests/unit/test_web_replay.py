"""Spectator replay frames (game_video.py input): engine only, no browser."""

import json
from pathlib import Path

from decklight.decks import load_ydk
from decklight.engine.duel import DuelConfig
from decklight.web.replay import replay_frames

FIXTURE = Path("tests/fixtures/web/mitsurugi-20261016")


def test_replay_frames_fixture(engine):
    rec = json.loads((FIXTURE / "game.jsonl").read_text())
    human, bot = load_ydk(FIXTURE / "human.ydk"), load_ydk(FIXTURE / "bot.ydk")
    cfg = (DuelConfig(human, bot, seed=rec["seed"]) if rec["human_seat"] == 0
           else DuelConfig(bot, human, seed=rec["seed"]))
    frames = replay_frames(
        engine, cfg, rec["choices"], ("Human", "Bot"),
        expect_winner=rec["winner"], auto_single=rec["auto_single"],
        surrendered=rec.get("surrendered", False))
    # one frame per choice plus the final board
    assert len(frames) == len(rec["choices"]) + 1
    for f in frames:
        assert f["t"] == "state" and f["prompt"] is None and f["thinking"] is None
        assert f["names"] == ["Human", "Bot"]
    # frame 0 is the title card: no caption bar; the rest narrate their choice
    assert frames[0]["caption"] is None and frames[0].get("title")
    for f in frames[1:-1]:
        assert (f.get("caption") or {}).get("text")
    # the final board carries the recorded winner by deck name
    want = "Human" if rec["winner"] == rec["human_seat"] else "Bot"
    assert frames[-1]["result"] == {"winner": want, "reason": "surrender"
                                    if rec.get("surrendered") else "win"}
    # narration-free kinds never reach the video log (the caption says it)
    kinds = {e["kind"] for f in frames for e in f["log"]}
    assert "select" not in kinds and "position" not in kinds
    # spectator: seat 1's opening hand is fully visible
    hand = frames[0]["view"]["bot"]["hand"]
    assert hand and all(c is not None and c["code"] for c in hand)
    # every step after the first carries the engine cues it caused (regression: cursor moved after the step)
    with_cues = sum(1 for f in frames[1:] if f["cues"])
    assert with_cues >= 10  # 15 of 49 on this fixture; 0 with the bug
