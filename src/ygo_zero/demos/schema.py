"""Stage-2 contract for the demonstration pipeline (ADR 0002).

Structured combo lines extracted from guide text. Field names here are the
contract for the stage-3 verifier — do not rename them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

ACTIONS = (
    "normal_summon",
    "set_monster",
    "special_summon",
    "activate",
    "link_summon",
    "synchro_summon",
    "xyz_summon",
    "fusion_summon",
    "ritual_summon",
    "set_spelltrap",
    "attack",
    "detach",
    "other",
)

GOING = ("first", "second", "unknown")

FORMATS = ("TCG", "OCG", "MD", "unknown")


@dataclass
class Step:
    i: int
    action: str  # one of ACTIONS (anything else is coerced to "other")
    card: str = ""
    card_code: int | None = None
    effect: str | None = None  # free-text hint, e.g. "search Maiden of White"
    materials: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    materials_codes: list[int | None] = field(default_factory=list)
    targets_codes: list[int | None] = field(default_factory=list)
    results_codes: list[int | None] = field(default_factory=list)
    text: str = ""  # the original sentence(s)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> Step:
        d = dict(d)
        d["i"] = int(d.get("i", 0))
        if d.get("action") not in ACTIONS:
            d["action"] = "other"
        for k in ("materials", "targets", "results"):
            v = d.get(k) or []
            d[k] = [str(x) for x in v] if isinstance(v, list) else [str(v)]
        for k in ("materials_codes", "targets_codes", "results_codes"):
            v = d.get(k)
            if v is None:
                d[k] = [None] * len(d[k[:-6]])
            else:
                d[k] = [None if x is None else int(x) for x in v]
        if d.get("card_code") is not None:
            d["card_code"] = int(d["card_code"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ComboLine:
    id: str = ""  # deck + source file + index (filled by the extractor)
    deck: str = ""
    source_file: str = ""
    source_url: str = ""
    format: str = "unknown"  # one of FORMATS
    title: str = ""
    hand: list[str] = field(default_factory=list)
    hand_codes: list[int | None] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)  # cards the line needs in deck/extra
    requires_codes: list[int | None] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    end_board: list[str] = field(default_factory=list)
    end_board_codes: list[int | None] = field(default_factory=list)
    end_board_text: str = ""
    claimed_interruptions: str | None = None
    going: str = "unknown"  # one of GOING
    unresolved_names: list[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["steps"] = [s.to_json() if isinstance(s, Step) else s for s in self.steps]
        return d

    @classmethod
    def from_json(cls, d: dict) -> ComboLine:
        d = dict(d)
        steps = [s if isinstance(s, Step) else Step.from_json(s) for s in d.get("steps", [])]
        d["steps"] = steps
        if d.get("format") not in FORMATS:
            d["format"] = "unknown"
        if d.get("going") not in GOING:
            d["going"] = "unknown"
        for k in ("hand_codes", "requires_codes", "end_board_codes"):
            v = d.get(k)
            base = {"hand_codes": "hand", "requires_codes": "requires", "end_board_codes": "end_board"}[k]
            if v is None:
                d[k] = [None] * len(d.get(base, []))
            else:
                d[k] = [None if x is None else int(x) for x in v]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
