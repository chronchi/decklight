# ADR 0003 — LLM-planned combo lines as the main demonstration source

Status: **proposed**, 2026-10-02. Go-ahead from the user ("go ahead with this whole thing"). It becomes accepted (or is revised) after the
pilot below reports.

## Context
ADR 0002 sources lines from internet guides. After one day on 8 decks: 18 verified lines and 1,958 BC decisions.
Phase 3's exit target is ≥10k decisions for 2–3 decks. Guide lines are rare, inconsistent (nicknames, fragments,
missing steps) and do not exist for unseen decks, which are the project goal. The engine verifier (ADR 0002 stage 3) is the
reusable core: it turns any proposed line into ground-truth choices or rejects it.

## Decision
Make **"LLM plans, engine verifies"** the main source of demonstrations:
1. Sample opening hands from a deck (reproducible).
2. An LLM (muse; cheaper models gated on yield) gets the hand, the decklist and the card texts, plus a few verified lines as
   examples, and writes one going-first `ComboLine` (same schema as extraction).
3. `verify_line` checks it. On failure, the LLM gets the failure point and the legal options there
   (`demos/explain.py`) and revises, at most N times.
4. Verified lines become BC records (tag `+planned`) and go through the existing hand-variant expansion.

Guides stay as few-shot examples and as a quality reference (do planned lines reach guide-level end boards?).
Per-decision LLM play is rejected: about 38 s per call × 50–100 decisions per turn is too slow, and the LLM would get
low-level choices (zones, positions, copies) wrong, which the verifier's search already fills in.

Later, in the same loop: going-second / matchup lines (plan against a verified opponent board, then hand traps), and
engine search (PLAN Phase 3 track 2) to improve on planned lines.

## Pilot (gate)
Blue-Eyes + Salamangreat, 20 hands each, `--repairs 2`. Measure verified/hand by attempt, decisions per line, calls and
minutes per verified line, and the end-board strength compared with the guide lines verified on the same decks. Go: ≥30% of hands verify
with non-trivial boards. Then scale to all pool decks and pause guide-yield work.
