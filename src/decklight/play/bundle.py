"""Bot bundles: one folder per bot, no env vars (docs/plans/2026-10-24-edopro-client.md §3.5).

A bundle is a dir with `bot.toml`, or a bare `.toml` (the shipped examples in
`config/bots/`):

    name = "raw-blueeyes"
    description = "Product default policy (no search) on Blue-Eyes."
    agent = "raw"            # random | raw | product
    deck = "deck.ydk"        # repo-relative, absolute, or bundle-relative
    ckpt = "../policy/best.pt"   # optional (raw, product)
    value = "../value/best.pt"   # optional (product)
    emb = "../emb.npz"           # optional (merged card embeddings)
    cards = ["../cards"]         # optional custom card dirs (*.cdb + script/*.lua)
    device = "cpu"           # optional

Paths are relative to the bundle dir and may point outside it (the shipped
examples reference the stock checkpoints). `finetune_deck.py` writes one at the
end of every fine-tune.

Transports resolve bundles, never env vars:
- EDOPro host (`--bot`): the bundle's deck, engine (custom cards) and
  checkpoints (`make_bot` for random/raw, `make_launcher` for product).
- Web lobby: `web.models` lists bundles; the duel uses
  `engine_for(bundle.cardset())`.
- EDOPro client (`Deck=<name>` / `--bot`): `make_agent` with ckpt/emb passed
  explicitly; StreamView over the bundle's engine cards.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_DIRS = (REPO_ROOT / "bots", REPO_ROOT / "config" / "bots")

MODEL_AGENTS = ("raw", "product")  # agents that read policy checkpoints/embeddings


class BundleError(ValueError):
    pass


def _resolve(ref: str | Path, base: Path, root: Path = REPO_ROOT) -> Path:
    """A bundle path ref -> absolute path (bundle-relative, then repo-relative)."""
    p = Path(ref)
    if p.is_absolute():
        return p
    if (base / p).exists() or not (root / p).exists():
        return base / p
    return root / p


@dataclass
class Bundle:
    """One validated bot: agent + deck + explicit checkpoint/embedding/card files."""

    name: str
    description: str = ""
    agent: str = "random"
    deck: str = ""
    ckpt: str | None = None
    value: str | None = None
    emb: str | None = None
    cards: list[str] = field(default_factory=list)
    device: str = "cpu"
    seed: int = 0
    dir: Path = field(default_factory=Path)
    source: Path = field(default_factory=Path)

    # --- loading ---

    @classmethod
    def load(cls, path: str | Path, root: Path = REPO_ROOT) -> Bundle:
        """Load + validate (every problem listed in one message)."""
        path = Path(path)
        if path.is_dir():
            toml_path = path / "bot.toml"
            base = path
        else:
            toml_path = path
            base = path.parent
        problems: list[str] = []
        try:
            data = tomllib.loads(toml_path.read_bytes().decode("utf-8"))
        except OSError as e:
            raise BundleError(f"bundle {path}: cannot read {toml_path}: {e}") from None
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            raise BundleError(f"bundle {path}: bad TOML in {toml_path}: {e}") from None
        if not isinstance(data, dict):
            raise BundleError(f"bundle {path}: {toml_path} must be a TOML table")
        known = {"name", "description", "agent", "deck", "ckpt", "value", "emb",
                 "cards", "device", "seed"}
        for k in sorted(set(data) - known):
            problems.append(f"unknown key {k!r}")

        def _path(key: str) -> Path | None:
            v = data.get(key)
            if v is None:
                return None
            if not isinstance(v, str):
                problems.append(f"{key} must be a path string, got {v!r}")
                return None
            return _resolve(v, base, root)

        name = data.get("name", toml_path.stem)
        if not isinstance(name, str) or not name:
            problems.append(f"name must be a non-empty string, got {data.get('name')!r}")
            name = toml_path.stem
        agent = data.get("agent", "random")
        from decklight.play.agents import AGENTS

        if agent not in AGENTS:
            problems.append(f"agent {agent!r} is not registered (known: {sorted(AGENTS)})")
        deck_p = _path("deck")
        if deck_p is None and data.get("deck") is None:
            problems.append("deck is required")
        elif deck_p is not None and not deck_p.is_file():
            problems.append(f"deck {deck_p} does not exist")
        ckpt_p, value_p, emb_p = _path("ckpt"), _path("value"), _path("emb")
        for key, p in (("ckpt", ckpt_p), ("value", value_p), ("emb", emb_p)):
            if p is not None and not p.is_file():
                problems.append(f"{key} {p} does not exist")
        raw_cards = data.get("cards", [])
        if isinstance(raw_cards, str):
            raw_cards = [raw_cards]
        if not isinstance(raw_cards, list) or any(not isinstance(c, str) for c in raw_cards):
            problems.append(f"cards must be a list of dir strings, got {data.get('cards')!r}")
            raw_cards = []
        card_dirs = [_resolve(c, base, root) for c in raw_cards]
        for d in card_dirs:
            if not d.is_dir():
                problems.append(f"cards dir {d} does not exist")
            elif not list(d.glob("*.cdb")):
                problems.append(f"cards dir {d} has no *.cdb files")

        # Deck parsable + covered by the card set (explicit dirs, no env involved).
        deck_codes: list[int] = []
        if deck_p is not None and deck_p.is_file():
            try:
                from decklight.decks import load_ydk

                deck = load_ydk(deck_p)
                deck_codes = [*deck.main, *deck.extra]
            except (OSError, ValueError) as e:
                problems.append(f"deck {deck_p} does not parse: {e}")
        cards_db: dict | None = None
        if not any(p.startswith("cards dir") for p in problems):
            try:
                from decklight.engine.resources import default_cdbs, load_cards

                cards_db = load_cards(default_cdbs(extra_dirs=card_dirs))
            except OSError as e:
                problems.append(f"cannot load card DBs: {e}")
        if cards_db is not None:
            missing = sorted({c for c in deck_codes if c not in cards_db})
            if missing:
                problems.append(f"deck has {len(missing)} codes missing from the card set, "
                                f"e.g. {missing[:5]}")
        # Embedding coverage when the bot reads a model (or emb is pinned).
        if data.get("emb") is not None or agent in MODEL_AGENTS:
            emb_path = emb_p
            if emb_path is None:
                try:
                    from decklight.search.product import resolve_product_paths

                    _, _, emb_path = resolve_product_paths(None, None, None)
                except (FileNotFoundError, OSError) as e:
                    problems.append(f"no embeddings resolve: {e}")
                    emb_path = None
            if emb_path is not None and emb_path.is_file():
                try:
                    import numpy as np

                    have = {int(c) for c in np.load(emb_path)["codes"]}
                    missing = sorted({c for c in deck_codes if c not in have})
                    if missing:
                        problems.append(f"deck has {len(missing)} codes missing from "
                                        f"embeddings {emb_path}, e.g. {missing[:5]}")
                except OSError as e:
                    problems.append(f"cannot read embeddings {emb_path}: {e}")
            elif emb_path is not None:
                problems.append(f"embeddings {emb_path} do not exist")

        if problems:
            raise BundleError(f"bundle {path} is invalid:\n" + "\n".join(f"  - {p}" for p in problems))
        return cls(name=name, description=str(data.get("description", "")),
                   agent=agent, deck=str(data.get("deck", "")),
                   ckpt=None if ckpt_p is None else str(data["ckpt"]),
                   value=None if value_p is None else str(data["value"]),
                   emb=None if emb_p is None else str(data["emb"]),
                   cards=list(raw_cards), device=str(data.get("device", "cpu")),
                   seed=int(data.get("seed", 0) or 0),
                   dir=base, source=toml_path)

    # --- resolved paths ---

    def _abs(self, ref: str | None) -> Path | None:
        return _resolve(ref, self.dir) if ref else None

    def deck_path(self) -> Path:
        return _resolve(self.deck, self.dir)

    def card_dirs(self) -> list[Path]:
        return [_resolve(c, self.dir) for c in self.cards]

    def cardset(self):
        """This bundle's explicit card set (official cards + its custom dirs)."""
        from decklight.engine.resources import CardSet

        return CardSet(tuple(self.card_dirs()))

    def engine(self):
        """This bundle's engine (cached per card set: `engine_for`)."""
        from decklight.engine.resources import engine_for

        return engine_for(self.cardset())

    def load_deck(self):
        """The bundle's deck (parsed .ydk)."""
        from decklight.decks import load_ydk

        return load_ydk(self.deck_path())

    # --- agents ---

    def _ckpt_paths(self, **overrides):
        ckpt = overrides.get("ckpt", self._abs(self.ckpt))
        value = overrides.get("value", self._abs(self.value))
        emb = overrides.get("emb", self._abs(self.emb))
        device = overrides.get("device", self.device or "cpu")
        return ckpt, value, emb, device

    def make_agent(self, **overrides):
        """The bundle's agent (ckpt/emb passed explicitly; no env).

        random/raw return a `SeatView` agent; product returns a `ProductLauncher`
        (search runs lockstep with an engine through the host/web driver).
        """
        from decklight.play.agents import make_agent

        if self.agent == "product":
            return self.make_launcher(**overrides)
        ckpt, _value, emb, device = self._ckpt_paths(**overrides)
        cfg: dict = {"seed": overrides.get("seed", self.seed), "device": device}
        if ckpt is not None:
            cfg["ckpt"] = ckpt
        if emb is not None:
            cfg["emb"] = emb
        return make_agent(self.agent, **cfg)

    def make_launcher(self, **overrides):
        """The bundle's product search launcher (agent == product)."""
        from decklight.edopro.product_bot import ProductLauncher

        ckpt, value, emb, device = self._ckpt_paths(**overrides)
        return ProductLauncher(ckpt=ckpt, value_ckpt=value, emb=emb, device=device,
                               fast=bool(overrides.get("fast", False)))

    def make_bot(self, **overrides):
        """The bundle's host-side bot (`next_choice(env)`; random/raw only)."""
        from decklight.edopro.bot import RandomBot
        from decklight.play.view import EngineView

        if self.agent == "product":
            raise BundleError(f"bundle {self.name!r}: product search joins the host via "
                              "make_launcher, not make_bot")
        if self.agent == "random":
            bot = RandomBot(overrides.get("seed", self.seed))
        else:
            agent = self.make_agent(**overrides)

            class _AgentBot:
                def __init__(self, agent) -> None:
                    self._agent = agent

                def next_choice(self, env) -> int:
                    return self._agent.act(EngineView(env))

                def rps(self) -> int:
                    return self._agent.rps()

            bot = _AgentBot(agent)
        return bot


def default_bundle_for_ydk(path: str | Path) -> Bundle:
    """A bare `.ydk` -> an on-the-fly random bundle (no file needed)."""
    p = Path(path)
    return Bundle(name=p.stem, description=f"Bare deck {p} (default random bot).",
                  agent="random", deck=str(p), dir=p.parent if p.parent != Path("") else Path("."),
                  source=p)


def discover_bundles(dirs=None, root: Path = REPO_ROOT) -> dict[str, Bundle]:
    """All bundles under `dirs` (default: `bots/` + `config/bots/`), keyed by name.

    Scans `*/bot.toml` (bundles) and `*.toml` (bare files). Duplicate names are an
    error naming both sources.
    """
    if dirs is None:
        dirs = [d for d in DEFAULT_BUNDLE_DIRS if d.is_dir()]
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    out: dict[str, Bundle] = {}
    sources: dict[str, Path] = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        cands = sorted(d.glob("*.toml")) + sorted(p for p in d.glob("*/bot.toml"))
        for cand in cands:
            bundle = Bundle.load(cand, root=root)
            if bundle.name in out:
                raise BundleError(f"duplicate bot name {bundle.name!r}: "
                                  f"{sources[bundle.name]} and {cand}")
            out[bundle.name] = bundle
            sources[bundle.name] = cand
    return out


def resolve_bundle(ref: str, catalog: dict[str, Bundle],
                   root: Path = REPO_ROOT) -> tuple[Path | None, Bundle | None]:
    """A `Deck=` value -> (ydk path, bundle or None).

    A catalog name wins; otherwise `ref` is a `.ydk` path (repo-relative or
    absolute, `.ydk` suffix optional) with an on-the-fly default bundle.
    """
    if ref in catalog:
        bundle = catalog[ref]
        return bundle.deck_path(), bundle
    p = Path(ref)
    if p.is_dir() and (p / "bot.toml").is_file():
        bundle = Bundle.load(p, root=root)
        return bundle.deck_path(), bundle
    if p.is_file() and p.suffix == ".toml":
        bundle = Bundle.load(p, root=root)
        return bundle.deck_path(), bundle
    if p.suffix != ".ydk":
        p = p.with_suffix(".ydk")
    full = p if p.is_absolute() else root / p
    if full.is_file():
        return full, default_bundle_for_ydk(full)
    return None, None


def bundles_text(catalog: dict[str, Bundle]) -> str:
    """One line per bundle (for `--list-bots`)."""
    lines = []
    for name in sorted(catalog):
        b = catalog[name]
        extra = f" [{b.agent}] {b.deck}"
        if b.description:
            extra += f" — {b.description.splitlines()[0][:100]}"
        lines.append(f"{name}{extra}")
    return "\n".join(lines) + ("\n" if lines else "")
