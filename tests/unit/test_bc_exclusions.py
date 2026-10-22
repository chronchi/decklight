"""Retrain stage D exclusions in the shard build (model/bc_data.load_exclusions / _skip_for)."""

import json

from decklight.model import bc_data as b


def test_exclusions_scope_ids_and_bases(tmp_path):
    (tmp_path / "d.jsonl").write_text(json.dumps({"line": "x/a#h1"}) + "\n")
    (tmp_path / "j.json").write_text(json.dumps({"muse": {"lines": ["x/b"]}}))
    ex = b.load_exclusions([f"data/demos/*,!*/strict-d/*@{tmp_path}/d.jsonl", f"data/demos/*,!*/strict-d/*@{tmp_path}/j.json"])
    old = b._skip_for("data/demos/planned-s1/verified/x.jsonl", ex)
    assert b._skip_for("data/demos/planned-s1/strict-d/verified/x.jsonl", ex) is None  # the replacement records stay
    assert old({"line_id": "x/a#h1"}) and not old({"line_id": "x/a#h2"})  # .jsonl: exact record ids
    assert old({"line_id": "x/b#h3"}) and old({"line_id": "x/b+imp"})  # .json: base line ids, every variant
    assert not old({"line_id": "x/c"})


def test_exclusion_file_must_list_lines(tmp_path):
    (tmp_path / "e.jsonl").write_text("")
    try:
        b.load_exclusions([f"*@{tmp_path}/e.jsonl"])
    except ValueError:
        return
    raise AssertionError("an empty exclusion file must fail loud")
