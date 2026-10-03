"""improve_lines.py: stronger-or-drop rule, idempotent skip, dedupe, prompt content.

LLM and verifier are monkeypatched (no network, no engine search). Fast and
pure: at most 2 worker processes are used (here: none, everything is faked).
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from ygo_zero.decks import Deck
from ygo_zero.demos.names import CardNames
from ygo_zero.demos.plan import compact_line_json, improve_prompt
from ygo_zero.demos.schema import ComboLine
from ygo_zero.demos.verify import Verdict

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


imp = _load("improve_lines")

NAMES = CardNames()
SAGE, PWWB, BLUE_EYES = 8240199, 63356631, 89631139
SAGE_N, PWWB_N, BEWD_N = "Sage with Eyes of Blue", "Phoenix Wing Wind Blast", "Blue-Eyes White Dragon"
FAKE_ENGINE = SimpleNamespace(cards={})


def _deck() -> Deck:
    main = (SAGE, PWWB, BLUE_EYES, SAGE, PWWB, BLUE_EYES, 48800175, 97268402, 38517737, 89631139)
    return Deck(main, (), (), "test-deck")


def _old(id_: str, end: list[str], steps: int = 2) -> ComboLine:
    line = ComboLine(
        id=id_,
        deck="test-deck",
        source_file="planner",
        hand=[SAGE_N, PWWB_N, BEWD_N, SAGE_N, PWWB_N],
        end_board=list(end),
        going="first",
    )
    from ygo_zero.demos.schema import Step

    line.steps = [
        Step(i=j, action="normal_summon" if j == 0 else "activate",
             card=SAGE_N if j == 0 else PWWB_N, text="x")
        for j in range(steps)
    ]
    line.end_board_codes = [None] * len(end)
    return line


def _candidate_json(end: list[str]) -> str:
    steps = [
        {"i": j, "action": "normal_summon" if j == 0 else "activate",
         "card": SAGE_N if j == 0 else PWWB_N, "text": "x"}
        for j in range(max(2, len(end)))
    ]
    return json.dumps(
        {"lines": [{"title": "stronger", "hand": ["PLACEHOLDER"], "steps": steps,
                    "end_board": end, "going": "first"}]}
    )


def _verdict(ok: bool, line_id: str, reason: str = "ok", executed: int = 2) -> Verdict:
    v = Verdict(line_id, ok, reason, [0], ["stated", "auto"], 0, 3, [], f"{reason} detail", [])
    v.executed_steps = executed
    return v


# --- prompt ---


def test_improve_prompt_contains_hand_and_old_line():
    old = _old("test-deck/plan-h0-a0", [SAGE_N])
    p = improve_prompt("test-deck", "SHEET", list(old.hand), compact_line_json(old), list(old.end_board))
    for name in old.hand:
        assert name in p
    assert compact_line_json(old)[:80] in p  # the verified line itself
    assert SAGE_N in p  # old end board
    assert '{"lines": []}' in p  # the no-stronger-line escape hatch
    assert "strictly stronger" in p
    assert "ONLY the cards that THIS step itself puts into" in p  # schema rules reused


# --- stronger-or-drop rule ---


def test_is_stronger_bigger_board_wins():
    assert imp.is_stronger(_old("n", [SAGE_N, BEWD_N]), 2, _old("o", [SAGE_N]), 5)
    assert not imp.is_stronger(_old("n", [SAGE_N]), 9, _old("o", [SAGE_N, BEWD_N]), 1)


def test_is_stronger_equal_board_needs_more_steps():
    assert imp.is_stronger(_old("n", [SAGE_N]), 3, _old("o", [SAGE_N]), 2)
    assert not imp.is_stronger(_old("n", [SAGE_N]), 2, _old("o", [SAGE_N]), 2)
    assert not imp.is_stronger(_old("n", [SAGE_N]), 1, _old("o", [SAGE_N]), 2)


def test_keep_new_needs_verified_and_stronger():
    old = _old("d/l0", [SAGE_N])
    assert imp.keep_new({"ok_line": None, "ok_verdict": None, "old_line": old, "old_executed": 2}) is False
    strong = _old("d/l0+imp", [SAGE_N, BEWD_N])
    assert imp.keep_new({"ok_line": strong, "ok_verdict": _verdict(True, strong.id),
                         "old_line": old, "old_executed": 2}) is True
    weak = _old("d/l0+imp", [SAGE_N])
    assert imp.keep_new({"ok_line": weak, "ok_verdict": _verdict(True, weak.id, executed=2),
                         "old_line": old, "old_executed": 2}) is False


def test_improve_roundtrip_keeps_stronger_drops_weaker():
    deck = _deck()
    olds = [(_old("d/l0", [SAGE_N]), 2), (_old("d/l1", [SAGE_N, BEWD_N]), 2)]
    responses = [_candidate_json([SAGE_N, BEWD_N, PWWB_N]), _candidate_json([SAGE_N])]

    def fake_complete(prompts: list[str]) -> list[str]:
        assert len(prompts) == 2
        assert SAGE_N in prompts[0]  # hand present
        return list(responses)

    def fake_verify(items: list[tuple]) -> list[Verdict]:
        return [_verdict(True, ln.id, executed=3) for ln, _, _ in items]

    res = imp.improve_old_lines(
        FAKE_ENGINE, deck, "test-deck", olds, NAMES, fake_complete, fake_verify, repairs=0)
    assert imp.keep_new(res[0]) is True  # 3 vs 1: stronger
    assert res[0]["ok_line"].id == "d/l0+imp"
    assert res[0]["ok_line"].hand == list(olds[0][0].hand)  # same hand forced
    assert imp.keep_new(res[1]) is False  # 1 vs 2: weaker, dropped


def test_empty_answer_is_terminal_drop():
    deck = _deck()
    olds = [(_old("d/l0", [SAGE_N]), 2)]

    def fake_verify(items: list[tuple]) -> list[Verdict]:  # pragma: no cover - must not be called
        raise AssertionError("no candidates to verify")

    res = imp.improve_old_lines(
        FAKE_ENGINE, deck, "d", olds, NAMES, lambda prompts: ['{"lines": []}'],
        fake_verify, repairs=1)
    assert res[0]["ok_line"] is None and res[0]["terminal"] == "empty"
    assert res[0]["attempts"][-1]["reason"] == "empty"
    assert imp.keep_new(res[0]) is False


def test_rejection_goes_to_repair():
    deck = _deck()
    olds = [(_old("d/l0", [SAGE_N]), 2)]
    seen: list[str] = []
    verifications = {"n": 0}

    def fake_complete(prompts: list[str]) -> list[str]:
        seen.extend(prompts)
        return [_candidate_json([SAGE_N, BEWD_N])]

    def fake_verify(items: list[tuple]) -> list[Verdict]:
        out = []
        for ln, _, _ in items:
            verifications["n"] += 1
            first_round = verifications["n"] == 1
            out.append(_verdict(not first_round, ln.id, "ok" if not first_round else "no_anchor@1"))
        return out

    res = imp.improve_old_lines(
        FAKE_ENGINE, deck, "d", olds, NAMES, fake_complete, fake_verify, repairs=1)
    assert [a["reason"] for a in res[0]["attempts"]] == ["no_anchor@1", "ok"]
    assert "engine rejected it" in seen[1]
    assert imp.keep_new(res[0]) is True


# --- idempotent skip + dedupe (file level) ---


def test_collect_olds_order_dedupe_and_missing(tmp_path: Path):
    src = tmp_path / "planned-s9"
    (src / "lines").mkdir(parents=True)
    (src / "verified").mkdir(parents=True)
    l0, l1 = _old("d/l0", [SAGE_N]), _old("d/l1", [SAGE_N, BEWD_N])
    (src / "lines" / "d.jsonl").write_text(
        json.dumps(l0.to_json()) + "\n" + json.dumps(l1.to_json()) + "\n")
    recs = [
        {"line_id": "d/l1", "executed_steps": 4},  # verified-file order first
        {"line_id": "d/l0", "executed_steps": 2},
        {"line_id": "d/l0", "executed_steps": 2},  # duplicate record
        {"line_id": "d/ghost", "executed_steps": 1},  # no line object
    ]
    (src / "verified" / "d.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    olds = imp.collect_olds(src, "d")
    assert [(ln.id, ex) for ln, ex in olds] == [("d/l1", 4), ("d/l0", 2)]


def test_attempted_ids_skip(tmp_path: Path):
    report = tmp_path / "report.json"
    assert imp.attempted_ids(report) == set()
    report.write_text(json.dumps([{"old_id": "d/l0", "new_id": None, "reason": "x",
                                           "old_end": [], "new_end": []}]))
    assert imp.attempted_ids(report) == {"d/l0"}
    # main's skip rule: olds whose id is already reported are not retried
    olds = [(_old("d/l0", [SAGE_N]), 2), (_old("d/l1", [SAGE_N]), 2)]
    pending = [(ln, ex) for ln, ex in olds if ln.id not in imp.attempted_ids(report)]
    assert [ln.id for ln, _ in pending] == ["d/l1"]


def test_append_dedupe(tmp_path: Path):
    path = tmp_path / "verified.jsonl"
    path.write_text(json.dumps({"line_id": "d/l0+imp"}) + "\n")
    n = imp.append_records_dedupe(
        path, [{"line_id": "d/l0+imp"}, {"line_id": "d/l1+imp"}])
    assert n == 1
    assert [json.loads(r)["line_id"] for r in path.read_text().splitlines()] == ["d/l0+imp", "d/l1+imp"]
    assert imp.append_records_dedupe(path, [{"line_id": "d/l1+imp"}]) == 0  # rerun appends nothing
