"""Bot bundles: explicit custom cards, no env vars (brief §Tests).

- load valid / invalid bundles (every problem listed at once);
- relative paths (bundle-relative, then repo-relative; may point outside);
- duplicate names name both paths;
- `engine_for` caches (same key -> same object, different custom dirs -> different engines);
- a bundle with the fixture custom cards builds an engine that knows those codes,
  with `YGO_EXTRA_CARDS` unset;
- no env read at import (`decklight.edopro.host` imports clean with env unset or garbage);
- `write_bundle` (finetune export) round-trips through `Bundle.load`;
- the web model list includes a bundle (no server start).
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from decklight.play.bundle import Bundle, BundleError, discover_bundles, resolve_bundle

ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_load_shipped_bundles():
    catalog = discover_bundles()
    assert {"random-snake-eye", "raw-blueeyes"} <= set(catalog)
    raw = catalog["raw-blueeyes"]
    assert raw.agent == "raw" and raw.deck.endswith("blue-eyes.ydk")
    assert raw.deck_path().is_file()
    deck = raw.load_deck()
    assert len(deck.main) >= 40


def test_load_invalid_lists_every_problem(tmp_path):
    bad = _write(tmp_path / "bad.toml",
                 'agent = "nope"\ndeck = "missing.ydk"\nckpt = "no.pt"\n'
                 'cards = ["nodir"]\nunknownkey = 1\n')
    with pytest.raises(BundleError, match="unknown key 'unknownkey'") as e:
        Bundle.load(bad)
    msg = str(e.value)
    assert "not registered" in msg and "does not exist" in msg
    assert msg.count("  - ") >= 5  # one line per problem


def test_load_bundle_dir_and_relative_paths(tmp_path):
    from test_custom_cards import CODE, _make_extra

    extra = _make_extra(tmp_path / "stock")
    bundle_dir = tmp_path / "bots" / "mine"
    bundle_dir.mkdir(parents=True)
    deck = bundle_dir / "deck.ydk"
    deck.write_text("#main\n" + "\n".join(["89631139"] * 39 + [str(CODE)]) +
                    "\n#extra\n!side\n")
    _write(bundle_dir / "bot.toml",
           'name = "mine"\nagent = "random"\ndeck = "deck.ydk"\n'
           f'cards = ["{extra}"]\n')
    bundle = Bundle.load(bundle_dir)
    assert bundle.name == "mine" and bundle.dir == bundle_dir
    assert bundle.deck_path() == deck
    assert bundle.card_dirs() == [extra]
    assert bundle.load_deck().main.count(CODE) == 1


def test_resolve_bundle_path_and_ydk(tmp_path):
    bundle_dir = tmp_path / "mine"
    bundle_dir.mkdir()
    (bundle_dir / "deck.ydk").write_text("#main\n" + "\n".join(["89631139"] * 40) +
                                         "\n#extra\n!side\n")
    (bundle_dir / "bot.toml").write_text('name = "mine"\nagent = "random"\ndeck = "deck.ydk"\n')
    path, bundle = resolve_bundle(str(bundle_dir), {})
    assert bundle is not None and bundle.name == "mine" and path == bundle.deck_path()
    path, bundle = resolve_bundle(str(bundle_dir / "bot.toml"), {})
    assert bundle is not None and bundle.name == "mine"
    assert resolve_bundle("no-such-bot", {}) == (None, None)


def test_duplicate_names_name_both_paths(tmp_path):
    _write(tmp_path / "a.toml", 'name = "dup"\nagent = "random"\n'
                                'deck = "decks/pool/snake-eye.ydk"\n')
    sub = tmp_path / "sub"
    sub.mkdir()
    _write(sub / "bot.toml", 'name = "dup"\nagent = "random"\n'
                             'deck = "decks/pool/snake-eye.ydk"\n')
    with pytest.raises(BundleError, match="a.toml.*bot.toml|bot.toml.*a.toml"):
        discover_bundles([tmp_path])


def test_engine_for_caches_per_cardset(tmp_path):
    from test_custom_cards import CODE, _make_extra

    from decklight.engine.resources import CardSet, engine_for

    extra = _make_extra(tmp_path)
    official, custom = CardSet(()), CardSet((extra,))
    assert official.key() != custom.key()
    assert engine_for(official) is engine_for(CardSet(()))
    assert engine_for(custom) is engine_for(CardSet((extra,)))
    assert CODE in engine_for(custom).cards
    assert CODE not in engine_for(official).cards


def test_custom_bundle_engine_with_env_unset(monkeypatch, tmp_path):
    """Explicit `cards` work with no env vars (the old code path needed them)."""
    from test_custom_cards import CODE, _make_extra

    from decklight.engine import resources as R

    monkeypatch.delenv(R.EXTRA_CARDS_ENV, raising=False)
    extra = _make_extra(tmp_path)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    deck = bundle_dir / "deck.ydk"
    deck.write_text("#main\n" + "\n".join(["89631139"] * 39 + [str(CODE)]) +
                    "\n#extra\n!side\n")
    _write(bundle_dir / "bot.toml",
           'name = "custom"\nagent = "random"\ndeck = "deck.ydk"\n'
           f'cards = ["{extra}"]\n')
    bundle = Bundle.load(bundle_dir)
    engine = bundle.engine()
    assert CODE in engine.cards  # the custom card is playable
    assert R.extra_card_dirs() == []  # ...with nothing leaking in from the env


def test_no_env_read_at_import(monkeypatch):
    """`decklight.edopro.host` imports with env unset or garbage (lazy EMB_PATH)."""
    import decklight.edopro.host as H

    monkeypatch.delenv("YGO_EMB", raising=False)
    monkeypatch.delenv("YGO_EXTRA_CARDS", raising=False)
    importlib.reload(H)
    assert not hasattr(H, "EMB_PATH")
    monkeypatch.setenv("YGO_EMB", "/nonexistent-xyz")
    monkeypatch.setenv("YGO_EXTRA_CARDS", "/nonexistent-xyz")
    importlib.reload(H)  # must not raise: nothing reads env at import
    from decklight.engine import resources as R

    assert R.extra_card_dirs() == [] and R.default_cdbs() == R.default_cdbs(extra_dirs=[])


def test_make_agent_passes_ckpt_explicitly(tmp_path, monkeypatch):
    from decklight.play import agents as AG

    seen: dict = {}

    class FakePolicy:
        def __init__(self, ckpt, emb, **k):
            seen.update(ckpt=ckpt, emb=emb)

    monkeypatch.setattr("decklight.search.product.load_policy",
                        lambda ckpt, emb, **k: FakePolicy(ckpt, emb))
    monkeypatch.setattr("decklight.search.product.resolve_product_paths",
                        lambda *a: (a[0], None, a[2]))
    bundle = Bundle.load(ROOT / "config" / "bots" / "raw-blueeyes.toml")
    agent = bundle.make_agent(ckpt=tmp_path / "p.pt", emb=tmp_path / "e.npz")
    assert isinstance(agent, AG.RawPolicyAgent)
    assert seen["ckpt"] == tmp_path / "p.pt" and seen["emb"] == tmp_path / "e.npz"


def test_product_agent_needs_engine(monkeypatch):
    from decklight.play import agents as AG

    agent = AG.make_agent("product")
    assert agent.needs == frozenset({"engine"})
    with pytest.raises(AG.MissingCapability):
        AG.check_needs(agent, frozenset(), "EDOPro client")
    seen: dict = {}

    class FakeLauncher:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("decklight.edopro.product_bot.ProductLauncher", FakeLauncher)
    prod = AG.ProductAgent(ckpt="c", value="v", emb="e", device="cpu", fast=True)
    prod.launcher()
    assert seen == {"ckpt": "c", "value_ckpt": "v", "emb": "e", "device": "cpu", "fast": True}
    with pytest.raises(RuntimeError, match="driver"):
        prod.act(None)


def test_web_models_include_bundles():
    from decklight.web.models import discover_models

    models = discover_models([ROOT / "runs"], [], None, None,
                             bundle_dirs=[ROOT / "config" / "bots"])
    bundles = [m for m in models if m["kind"] == "bundle"]
    assert {m["id"] for m in bundles} >= {"bundle-random-snake-eye", "bundle-raw-blueeyes"}
    raw = next(m for m in bundles if m["id"] == "bundle-raw-blueeyes")
    assert raw["agent"] == "raw" and raw["ckpt"] is not None
    # the lobby payload carries no local paths (backwards compatible: new fields only)
    from decklight.web.server import Server

    public = Server._public_model(raw)
    assert "ckpt" not in public and "bundle" not in public and public["kind"] == "bundle"


def test_finetune_write_bundle_round_trip(tmp_path):
    import importlib.util

    import torch

    mod = importlib.util.spec_from_file_location(
        "finetune_deck_mod", ROOT / "scripts" / "finetune_deck.py")
    ft = importlib.util.module_from_spec(mod)
    mod.loader.exec_module(ft)
    init_d, value_d = tmp_path / "P", tmp_path / "value_p2"
    for d in (init_d, value_d):
        d.mkdir(parents=True)
        (d / "metrics.jsonl").write_text("{}\n")
    torch.save({"trunk.weight": torch.ones(2)}, init_d / "best.pt")
    torch.save({"trunk.weight": torch.ones(2)}, value_d / "best.pt")
    out = tmp_path / "ft"
    (out / "policy").mkdir(parents=True)
    (out / "value").mkdir(parents=True)
    policy = out / "policy" / "best.pt"
    value = out / "value" / "best.pt"
    policy.write_bytes((init_d / "best.pt").read_bytes())
    value.write_bytes((value_d / "best.pt").read_bytes())
    deck = ROOT / "decks" / "pool" / "blue-eyes.ydk"
    bundle_dir = ft.write_bundle(out, deck=deck, policy_pt=policy, value_pt=value, cards=[])
    assert (bundle_dir / "deck.ydk").is_file()
    bundle = Bundle.load(bundle_dir)
    assert bundle.agent == "product" and bundle.deck == "deck.ydk"
    assert bundle._abs(bundle.ckpt) == policy
    assert bundle._abs(bundle.value) == value
