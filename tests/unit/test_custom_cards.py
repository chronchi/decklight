"""Custom cards: `YGO_EXTRA_CARDS` dirs flow into the engine, embeddings and lua_desc merges.

Fast (fake vectors; the real Qwen model never loads on the laptop).
"""

import os
import sqlite3
import sys

import numpy as np
import pytest

from decklight.cards import effect_strings as ES
from decklight.cards import embed as E
from decklight.engine import resources as R

DIM = 8
CODE = 999999001
STRS = [f"Fake effect string {i}." for i in range(16)]

FAKE_SCRIPT = """\
local s,id=GetID()
function s.initial_effect(c)
\tlocal e1=Effect.CreateEffect(c)
\te1:SetDescription(aux.Stringid(id,0))
\te1:SetType(EFFECT_TYPE_ACTIVATE)
\te1:SetCode(EVENT_FREE_CHAIN)
\te1:SetCategory(CATEGORY_DRAW)
\te1:SetCountLimit(1,id)
\tc:RegisterEffect(e1)
end
"""


def _make_extra(tmp_path, code=CODE, script_name=None) -> object:
    """A minimal extra card dir: `test.cdb` (datas + texts) and `script/c<code>.lua`."""
    d = tmp_path / "extra"
    (d / "script").mkdir(parents=True)
    con = sqlite3.connect(d / "test.cdb")
    con.execute("CREATE TABLE datas (id INT, alias INT, setcode INT, type INT, atk INT, def INT, "
                "level INT, race INT, attribute INT)")
    con.execute("INSERT INTO datas VALUES (?, 0, 0, 33, 1000, 1000, 4, 1, 16)", (code,))
    cols = ", ".join(["id", "name", "desc"] + [f"str{i}" for i in range(1, 17)])
    con.execute(f"CREATE TABLE texts ({cols})")
    con.execute(f"INSERT INTO texts VALUES ({', '.join('?' * 19)})",
                (code, "Fake Custom", "Fake custom card text.", *STRS))
    con.commit()
    con.close()
    (d / "script" / (script_name or f"c{code}.lua")).write_text(FAKE_SCRIPT)
    return d


def _fake_encode(texts):
    rng = np.random.default_rng(sum(hash(t) & 0xFFFF for t in texts) or 1)
    v = rng.normal(size=(len(texts), DIM)).astype(np.float16)
    return v / np.linalg.norm(v, axis=1, keepdims=True).astype(np.float16)


def _card_base(tmp_path, exclude=(CODE,)):
    base = tmp_path / "base.npz"
    rng = np.random.default_rng(0)
    codes = [c for c in sorted(E.card_texts()) if c not in exclude]
    n = len(codes)
    card = (rng.normal(size=(n, DIM)) / 4).astype(np.float16)
    effects = (rng.normal(size=(n, E.MAX_EFFECTS, DIM)) / 4).astype(np.float16)
    np.savez_compressed(base, codes=np.array(codes, dtype=np.int64), card=card, effects=effects,
                        n_effects=np.zeros(n, dtype=np.int8), model=E.MODEL, dim=DIM)
    return base


def test_unset_env_is_byte_identical(monkeypatch, tmp_path):
    """Env var unset: official-only lists; set: official prefix identical + extras appended."""
    monkeypatch.delenv(R.EXTRA_CARDS_ENV, raising=False)
    assert R.extra_card_dirs() == []
    cdbs_off, scripts_off = R.default_cdbs(), R.index_scripts()
    assert not [p for p in cdbs_off if "extra" in str(p)]
    assert f"c{CODE}.lua" not in scripts_off

    d = _make_extra(tmp_path)
    monkeypatch.setenv(R.EXTRA_CARDS_ENV, str(d))
    cdbs_on, scripts_on = R.default_cdbs(), R.index_scripts()
    assert cdbs_on[: len(cdbs_off)] == cdbs_off
    assert cdbs_on[len(cdbs_off):] == [d / "test.cdb"]
    off_names = {k: str(v) for k, v in scripts_off.items()}
    on_names = {k: str(v) for k, v in scripts_on.items()}
    assert {k: v for k, v in on_names.items() if k in off_names} == off_names
    assert scripts_on[f"c{CODE}.lua"] == d / "script" / f"c{CODE}.lua"


def test_extra_script_never_overrides_official(monkeypatch, tmp_path):
    """An extra script with an official name loses (setdefault)."""
    d = _make_extra(tmp_path, script_name="constant.lua")
    monkeypatch.setenv(R.EXTRA_CARDS_ENV, str(d))
    idx = R.index_scripts()
    assert idx["constant.lua"] == R.SCRIPTS_DIR / "constant.lua"


def test_custom_codes_load(monkeypatch, tmp_path):
    """The custom code lands in `load_cards()` and the script index."""
    d = _make_extra(tmp_path)
    monkeypatch.setenv(R.EXTRA_CARDS_ENV, str(d))
    cards = R.load_cards()
    assert cards[CODE].attack == 1000 and cards[CODE].defense == 1000
    assert R.index_scripts()[f"c{CODE}.lua"] == d / "script" / f"c{CODE}.lua"
    assert E.card_texts()[CODE] == ("Fake Custom", "Fake custom card text.")
    assert ES.card_strings()[ES.desc_key(CODE, 0)] == STRS[0]


def test_only_missing_cards_merge(monkeypatch, tmp_path):
    """Card merge: base rows byte-identical, the custom code appended (fake vectors)."""
    d = _make_extra(tmp_path)
    monkeypatch.setenv(R.EXTRA_CARDS_ENV, str(d))
    base = _card_base(tmp_path)
    missing, names = E.missing_card_codes(base)
    assert missing == [CODE] and names[CODE] == "Fake Custom"
    out = tmp_path / "merged.npz"
    E.append_card_embeddings(base, out, dim=DIM, encode=_fake_encode)
    z0, z1 = np.load(base), np.load(out)
    assert z1["codes"].tolist() == z0["codes"].tolist() + [CODE]
    n0 = len(z0["codes"])
    assert np.array_equal(z1["card"][:n0], z0["card"])
    assert np.array_equal(z1["effects"][:n0], z0["effects"])
    assert np.array_equal(z1["n_effects"][:n0], z0["n_effects"])
    assert z1["n_effects"][n0] > 0 and np.abs(z1["card"][n0]).sum() > 0
    # nothing missing -> a copy of base
    E.append_card_embeddings(out, tmp_path / "merged2.npz", dim=DIM, encode=_fake_encode)
    z2 = np.load(tmp_path / "merged2.npz")
    assert z2["codes"].tolist() == z1["codes"].tolist() and np.array_equal(z2["card"], z1["card"])


def test_only_missing_strings_merge(monkeypatch, tmp_path):
    """String merge: every base key maps to a byte-identical vector, new keys appended (fake)."""
    d = _make_extra(tmp_path)
    monkeypatch.setenv(R.EXTRA_CARDS_ENV, str(d))
    base = tmp_path / "sbase.npz"
    rng = np.random.default_rng(1)
    vecs = (rng.normal(size=(2, DIM)) / 4).astype(np.float16)
    k_sys = ES.desc_key(0, 1)
    np.savez_compressed(base, keys=np.array([k_sys, 100], dtype=np.int64),
                        rows=np.array([0, 1], dtype=np.int32), vecs=vecs, model=E.MODEL, dim=DIM)
    missing, texts = ES.missing_string_keys(base)
    assert ES.desc_key(CODE, 0) in missing and texts[ES.desc_key(CODE, 0)] == STRS[0]
    out = tmp_path / "smerged.npz"
    ES.append_string_embeddings(base, out, dim=DIM, encode=_fake_encode)
    z0, z1 = np.load(base), np.load(out)
    assert (np.diff(z1["keys"]) > 0).all()  # still sorted for DescTables._lookup
    assert set(z0["keys"].tolist()) < set(z1["keys"].tolist())
    tab0 = {int(k): vecs[i] for i, k in enumerate(z0["keys"])}
    tab1 = {int(k): z1["vecs"][r] for k, r in zip(z1["keys"].tolist(), z1["rows"].tolist())}
    for k, v in tab0.items():
        assert np.array_equal(tab1[k], v)
    assert np.array_equal(z1["vecs"][: len(vecs)], vecs)  # base vecs: byte-identical prefix


def test_lua_desc_merge(monkeypatch, tmp_path):
    """lua_desc merge: base rows kept, the custom script's descriptions appended."""
    script_dir = tmp_path / "extra" / "script"
    script_dir.mkdir(parents=True)
    (script_dir / f"c{CODE}.lua").write_text(FAKE_SCRIPT)
    base = tmp_path / "lua.npz"
    feats = np.arange(len(ES.LUA_COLS), dtype=np.int64).reshape(1, -1)
    np.savez_compressed(base, keys=np.array([123], dtype=np.int64), feats=feats,
                        cols=np.array(ES.LUA_COLS))
    out = tmp_path / "lua-custom.npz"
    ES.merge_lua_desc(base, out, extra_dirs=[script_dir])
    z1 = np.load(out)
    assert 123 in z1["keys"].tolist()
    i = z1["keys"].tolist().index(123)
    assert np.array_equal(z1["feats"][i], feats[0])
    new_key = ES.desc_key(CODE, 0)
    assert new_key in z1["keys"].tolist()
    assert (np.diff(np.sort(z1["keys"])) > 0).all()
    # official-named scripts lose: parsing the official dir adds nothing new here
    n = len(z1["keys"])
    ES.merge_lua_desc(base, tmp_path / "lua2.npz", extra_dirs=[script_dir])
    assert len(np.load(tmp_path / "lua2.npz")["keys"]) == n


def test_never_overwrites_stock_lua(monkeypatch, tmp_path):
    """Default lua_desc.npz is never overwritten with an extras build unless --in-place."""
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ES.build_lua(extra_dirs=[script_dir])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ES.merge_lua_desc(tmp_path / "b.npz", ES.EMB_DIR.parent / "lua_desc.npz",
                           extra_dirs=[script_dir])


def test_fake_encode_never_touches_model(monkeypatch, tmp_path):
    """Guard: the merge tests above would fail loudly if the real model tried to load."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    d = _make_extra(tmp_path)
    monkeypatch.setenv(R.EXTRA_CARDS_ENV, str(d))
    E.append_card_embeddings(_card_base(tmp_path), tmp_path / "m.npz", dim=DIM, encode=_fake_encode)
    assert (tmp_path / "m.npz").is_file()


def test_env_overrides_for_merged_tables(monkeypatch, tmp_path):
    """`$YGO_EMB` / `$YGO_STRINGS` / `$YGO_LUA` redirect the product's table loading."""
    from decklight.model import policy_v3 as P3
    from decklight.search import product as PR

    emb = tmp_path / "emb.npz"
    rng = np.random.default_rng(3)
    np.savez(emb, codes=np.array([7], dtype=np.int64), card=rng.normal(size=(1, DIM)).astype(np.float16),
             effects=np.zeros((1, E.MAX_EFFECTS, DIM), np.float16), n_effects=np.zeros(1, np.int8),
             model=E.MODEL, dim=DIM)
    monkeypatch.setenv("YGO_EMB", str(emb))
    assert E.CardEmbeddings().codes.tolist() == [7]
    assert PR.resolve_product_paths(ckpt=__file__, value=__file__, emb=None)[2] == emb
    s = tmp_path / "s.npz"
    np.savez(s, keys=np.array([1]), rows=np.array([0]), vecs=np.zeros((1, DIM), np.float16))
    lu = tmp_path / "l.npz"
    np.savez_compressed(lu, keys=np.array([1]), feats=np.zeros((1, len(ES.LUA_COLS)), dtype=np.int64))
    monkeypatch.setenv("YGO_STRINGS", str(s))
    monkeypatch.setenv("YGO_LUA", str(lu))
    t = P3.load_desc_tables()
    assert t["str_keys"].tolist() == [1] and t["lua_keys"].tolist() == [1]


def test_no_extra_dirs_without_env(monkeypatch):
    monkeypatch.delenv(R.EXTRA_CARDS_ENV, raising=False)
    assert R.extra_card_dirs() == [] and R.extra_script_dirs() == []


def test_pathsep_splits_and_skips_missing(monkeypatch, tmp_path):
    d = _make_extra(tmp_path)
    monkeypatch.setenv(R.EXTRA_CARDS_ENV, os.pathsep.join([str(d), "/nonexistent-xyz"]))
    assert R.extra_card_dirs() == [d]
