import json

from ygo_zero.demos.extract import extract_lines, sequence_overlap, split_sections
from ygo_zero.demos.names import CardNames, normalize
from ygo_zero.demos.schema import ComboLine, Step

NAMES = CardNames()


def test_names_known_cards():
    # canonical code first; alt prints may follow under the same name
    assert NAMES.resolve("Ash Blossom & Joyous Spring")[0] == 14558127
    assert NAMES.resolve("Blue-Eyes White Dragon")[0] == 89631139


def test_names_insensitive_and_unknown():
    assert NAMES.resolve("blue-eyes white dragon")[0] == 89631139
    assert NAMES.resolve("Blue Eyes White Dragon")[0] == 89631139
    assert NAMES.resolve("Not A Real Card Xyz") == []
    assert NAMES.resolve("") == []


def test_normalize():
    assert normalize("Ash Blossom & Joyous Spring") == normalize("ash-blossom joyous spring")


def test_schema_round_trip():
    line = ComboLine(
        id="blue-eyes/x:0",
        deck="blue-eyes",
        source_file="x",
        source_url="u",
        format="TCG",
        title="t",
        hand=["Sage with Eyes of Blue"],
        hand_codes=[None],
        requires=[],
        requires_codes=[],
        steps=[Step(i=0, action="normal_summon", card="Sage with Eyes of Blue", text="Normal Summon Sage.")],
        end_board=["Blue-Eyes Spirit Dragon"],
        end_board_codes=[None],
        end_board_text="board",
        claimed_interruptions=None,
        going="first",
        unresolved_names=[],
        notes="",
    )
    back = ComboLine.from_json(json.loads(json.dumps(line.to_json())))
    assert back == line
    assert back.steps[0].action == "normal_summon"


def test_schema_coerces_bad_action():
    s = Step.from_json({"i": 0, "action": "activate_spell", "card": "X"})
    assert s.action == "other"
    line = ComboLine.from_json(
        {"id": "a", "deck": "d", "source_file": "f", "format": "BOGUS", "going": "sideways"}
    )
    assert line.format == "unknown" and line.going == "unknown"


MOCK_JSON = json.dumps(
    {
        "lines": [
            {
                "title": "Sage opener",
                "hand": ["Sage with Eyes of Blue"],
                "requires": ["Blue-Eyes White Dragon"],
                "steps": [
                    {
                        "i": 0,
                        "action": "normal_summon",
                        "card": "Sage with Eyes of Blue",
                        "text": "Normal Summon Sage with Eyes of Blue.",
                    },
                    {
                        "i": 1,
                        "action": "activate",
                        "card": "Sage with Eyes of Blue",
                        "effect": "search Maiden of White",
                        "results": ["Maiden of White"],
                        "text": "Activate Sage effect; search Maiden of White.",
                    },
                ],
                "end_board": ["Blue-Eyes Spirit Dragon"],
                "end_board_text": "Spirit Dragon pass",
                "claimed_interruptions": None,
                "going": "first",
            }
        ]
    }
)


def _mock_complete(prompt, system=None, **kw):
    assert "Deck:" in prompt
    return MOCK_JSON


def test_extract_with_mocked_llm(tmp_path):
    guide = tmp_path / "g.txt"
    guide.write_text(
        "URL: https://example.com/g\n" + "=" * 70 + "\nCOMBO 1: Sage opener\n"
        "Normal Summon Sage, search Maiden, end on Spirit Dragon.\n"
    )
    lines = extract_lines(guide, "blue-eyes", {8240199, 89631139}, names=NAMES, complete_fn=_mock_complete)
    assert len(lines) == 1
    line = lines[0]
    assert line.id == "blue-eyes/g:0" and line.deck == "blue-eyes"
    assert line.hand_codes == [8240199]  # Sage resolved
    assert line.steps[1].results_codes == [NAMES.resolve("Maiden of White")[0]]
    assert "outside decklist" not in line.notes  # all in decklist
    assert line.going == "first"


def test_extract_flags_off_decklist(tmp_path):
    guide = tmp_path / "g.txt"
    guide.write_text("COMBO 1: stuff\nDo the thing with Blue-Eyes White Dragon.\n")
    lines = extract_lines(guide, "blue-eyes", {12345}, names=NAMES, complete_fn=_mock_complete)
    assert "outside decklist" in lines[0].notes
    assert "Blue-Eyes White Dragon" in lines[0].notes
    assert "Sage with Eyes of Blue" in lines[0].notes  # hand card not in decklist either


def test_split_sections():
    secs = split_sections("COMBO 1: opener\nDo A.\n\nCOMBO 2: closer\nDo B.\n")
    assert len(secs) == 2
    assert split_sections("just some prose with no headers")[0][0] == "full"


def test_sequence_overlap():
    assert sequence_overlap(["a", "b", "b"], ["b", "b", "c"]) == 4 / 6
    assert sequence_overlap([], ["a"]) == 0.0
