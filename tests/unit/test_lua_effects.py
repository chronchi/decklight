"""Unit tests for the deterministic Lua per-effect extractor (no engine build needed)."""

import pytest

from ygo_zero.cards import lua_effects as L

CONST_LUA = L.REPO_ROOT / "third_party" / "CardScripts" / "official" / ".." / "constant.lua"
OFFICIAL = L.OFFICIAL_DIR
ASH = 14558127

SYNTH_CONSTS = {
    "EFFECT_TYPE_QUICK_O": 0x100,
    "EVENT_CHAINING": 1027,
    "EFFECT_FLAG_DELAY": 0x10000,
    "EFFECT_FLAG_CARD_TARGET": 0x10,
    "EFFECT_FLAG_DAMAGE_STEP": 0x4000,
}

SYNTH = """--[[ block comment with a fake effect that must be ignored:
local ex=Effect.CreateEffect(c)
ex:SetCode(9999)
]]
local s,id=GetID()
function s.initial_effect(c)
    -- line comment with a fake: local ey=Effect.CreateEffect(c)
    local e1=Effect.CreateEffect(c)
    e1:SetType(EFFECT_TYPE_QUICK_O)
    e1:SetCode(EVENT_CHAINING)
    e1:SetProperty(EFFECT_FLAG_DELAY+EFFECT_FLAG_CARD_TARGET)
    e1:SetCountLimit(1,{id,1})
    e1:SetCondition(s.con)
    e1:SetCost(s.cost)
    e1:SetTarget(s.tg)
    e1:SetOperation(s.op)
    c:RegisterEffect(e1)
    local e2=e1:Clone()
    e2:SetProperty(EFFECT_FLAG_DELAY|EFFECT_FLAG_DAMAGE_STEP)
    e2:SetCode(s.custom)
    c:RegisterEffect(e2)
end
"""


def _needs_scripts():
    if not (OFFICIAL / f"c{ASH}.lua").is_file():
        pytest.skip("CardScripts submodule not checked out")


def test_constants():
    if not CONST_LUA.is_file():
        pytest.skip("CardScripts submodule not checked out")
    consts = L.load_constants()
    assert consts["EFFECT_TYPE_IGNITION"] == 0x40
    assert consts["EFFECT_FLAG_DELAY"] == 0x10000
    assert consts["EVENT_SUMMON_SUCCESS"] == 1100


def test_ash_blossom():
    _needs_scripts()
    consts = L.load_constants()
    recs = L.script_effects(ASH, constants=consts)
    assert len(recs) == 1
    (rec,) = recs
    assert rec.var == "e1"
    assert rec.type == consts["EFFECT_TYPE_QUICK_O"]
    assert rec.code == consts["EVENT_CHAINING"]
    assert rec.range == consts["LOCATION_HAND"]
    assert rec.count_limit == (1, ASH)
    assert rec.description == 0
    assert rec.category == consts["CATEGORY_DISABLE"]
    assert rec.has_condition and rec.has_cost and rec.has_target and rec.has_operation
    assert rec.registered
    assert rec.cloned_from is None
    assert rec.unresolved == []


def test_clone_property_countlimit_unresolved_and_block_comment():
    recs = L.parse_script(SYNTH, 12345678, SYNTH_CONSTS)
    assert [r.var for r in recs] == ["e1", "e2"]  # block/line-comment fakes ignored
    e1, e2 = recs
    assert e1.property == 0x10000 + 0x10  # A+B form
    assert e1.count_limit == (1, 12345678, 1)  # {id,1} flattened
    assert e1.registered and e1.cloned_from is None and e1.unresolved == []
    assert e2.cloned_from == "e1"
    assert e2.type == e1.type  # inherited from the clone source
    assert e2.property == 0x10000 | 0x4000  # A|B form overrides the copy
    assert e2.code == 1027  # failed SetCode keeps the cloned value
    assert e2.unresolved == ["SetCode:s.custom"]
    assert e2.registered


def test_procedures():
    text = (
        "local s,id=GetID()\n"
        "function s.initial_effect(c)\n"
        "  -- Fusion.AddProcedure(c) in a comment must be ignored\n"
        "  Fusion.AddProcMix(c,true,true,aux.FilterBoolFunction(Card.IsCode,123))\n"
        "  Xyz.AddProcedure(c,aux.FilterBoolFunction(Card.IsRace,RACE_MACHINE),4,2)\n"
        "end\n"
    )
    assert L.procedures(text) == ["Fusion.AddProcMix", "Xyz.AddProcedure"]
    assert L.procedures("-- just a comment\n") == []
