"""Timing labels for the card-text embedding probe (laptop: text parsing only, no ML).

Per card (rows aligned with the embedding table's `codes`), labels from the engine's Lua scripts (ground truth):
quick (QUICK_O/F), hand_quick (quick effect usable from the hand: hand traps), trigger (TRIGGER_O/F), ignition,
hopt (a count limit keyed to the card's id: "you can only use this effect of X once per turn"), cost, target.
Per effect sentence (rows aligned with `effects[:, j]`), labels from PSCT punctuation: colon (an activation
condition or trigger timing precedes ':'), semicolon (a cost or target precedes ';'), quick_tag ('(Quick Effect)'),
opt ('once per turn'), either_turn ("either player's turn" / "opponent's turn"), plus the sentence text itself for a
bag-of-words baseline.

    uv run python scripts/diag/text_labels.py  -> data/cache/diag/text_labels.npz
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from decklight.cards.embed import MAX_EFFECTS, card_texts, split_effects
from decklight.cards.lua_effects import OFFICIAL_DIR, load_constants, parse_script

ROOT = Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "cards" / "emb" / "qwen3-0.6b-d512.npz"
OUT = ROOT / "data" / "cache" / "diag" / "text_labels.npz"
QUICK, TRIG, IGN = 0x100 | 0x400, 0x80 | 0x200, 0x40
HAND = 0x2
CARD_KEYS = ("quick", "hand_quick", "trigger", "ignition", "hopt", "cost", "target")
EFF_KEYS = ("colon", "semicolon", "quick_tag", "opt", "either_turn")


def main() -> int:
    z = np.load(EMB)
    codes, n_eff = z["codes"], z["n_effects"]
    texts = card_texts()
    consts = load_constants()
    lab = np.full((len(codes), len(CARD_KEYS)), -1, np.int8)  # -1 = no Lua script parsed
    eff = np.full((len(codes), MAX_EFFECTS, len(EFF_KEYS)), -1, np.int8)
    eff_text: list[list[str]] = []
    n_lua = 0
    for i, code in enumerate(codes):
        code = int(code)
        name, desc = texts.get(code, ("", ""))
        sents = split_effects(desc)[:MAX_EFFECTS]
        eff_text.append(sents)
        if len(sents) != int(n_eff[i]):
            sents = []  # misaligned with the embedding table (text changed since embedding): skip effect labels
        for j, s in enumerate(sents):
            eff[i, j] = [":" in s, ";" in s, "(Quick Effect)" in s, "once per turn" in s.lower(),
                         bool(re.search(r"either player's turn|opponent's turn", s))]
        p = OFFICIAL_DIR / f"c{code}.lua"
        if not p.is_file():
            continue
        recs = parse_script(p.read_text(encoding="utf-8", errors="replace"), code, consts)
        if not recs:
            continue
        n_lua += 1
        q = [r for r in recs if r.type & QUICK]
        lab[i] = [bool(q), any(r.range & HAND for r in q), any(r.type & TRIG for r in recs),
                  any(r.type & IGN for r in recs),
                  any(r.count_limit is not None and len(r.count_limit) > 1 and code in r.count_limit for r in recs),
                  any(r.has_cost for r in recs), any(r.property & 0x10 for r in recs)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, codes=codes, card_lab=lab, eff_lab=eff, card_keys=np.array(CARD_KEYS),
                        eff_keys=np.array(EFF_KEYS), eff_text=np.array(json.dumps(eff_text)))
    has = lab[:, 0] >= 0
    print(f"{len(codes)} cards, {n_lua} with Lua labels; card label rates:",
          {k: round(float(lab[has, j].mean()), 3) for j, k in enumerate(CARD_KEYS)})
    e = eff.reshape(-1, len(EFF_KEYS)); e = e[e[:, 0] >= 0]
    print(f"{len(e)} effect sentences; rates:", {k: round(float(e[:, j].mean()), 3) for j, k in enumerate(EFF_KEYS)})
    if n_lua < 0.5 * len(codes) or len(e) < 10000:
        print("ALERT text_labels: too few labels")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
