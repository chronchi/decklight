"""Mid-game puzzle starts for the search teachers (ADR 0006 amendment 2026-10-05).

An LLM designs situations (data/scenarios/<id>.json); the engine validates
them here; the search teachers label the decisions (scripts/gen_window.py).

    python scripts/scenarios.py validate data/scenarios/*.json
    python scripts/scenarios.py play data/scenarios/*.json --out <dir> --games G [--policy random|bc|greedy] [--seed S]
    python scripts/scenarios.py list-cards <deck>

A scenario JSON: {id, theme, deck0, deck1, focus, lp, start, p0, p1, why},
where p0/p1 = {hand: [names], mzone: [{card, pos}], szone: [{card, pos}],
grave: [names], removed: [names]}. See data/scenarios/ for examples.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.decks import Deck, load_ydk
from ygo_zero.demos.names import normalize
from ygo_zero.engine import constants as C
from ygo_zero.engine.actions import Kind
from ygo_zero.engine.cpp_env import CppEnv
from ygo_zero.engine.duel import DuelConfig, Engine, Placement

POOL = ROOT / "decks" / "pool"
MAX_DECISIONS = 600

MZONE_POS = {"atk": C.POS_FACEUP_ATTACK, "def": C.POS_FACEUP_DEFENSE, "set": C.POS_FACEDOWN_DEFENSE}
SZONE_POS = {"up": C.POS_FACEUP, "set": C.POS_FACEDOWN}

# Preference order when skipping a turn (answer every prompt with PASS / end-turn).
_SKIP_WANTS = (Kind.TO_END, Kind.TO_MAIN2, Kind.TO_BATTLE, Kind.PASS, Kind.NO, Kind.FINISH, Kind.CANCEL)


def pool_decks() -> tuple[list[str], set[str]]:
    pool = sorted(p.stem for p in POOL.glob("*.ydk"))
    hold = set((POOL / "HOLDOUT").read_text().split()) | {"kewl-tune"}
    return pool, hold


def load_names():
    from ygo_zero.demos.names import CardNames

    return CardNames()


def exact_index(names) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for code, name in names.names.items():
        index.setdefault(normalize(name), []).append(code)
    return index


def resolve_exact(name: str, names, index: dict[str, list[int]]) -> int | None:
    """Exact (case/punctuation-insensitive) name -> canonical code; None when unknown. Never guesses."""
    key = normalize(name)
    if not key or key not in index:
        return None
    cands = index[key]
    alias = getattr(names, "alias", {})
    return min(cands, key=lambda c: (alias.get(c, 0) != 0, c))


def _entry_name(e) -> str:
    return e["card"] if isinstance(e, dict) else e


def _entry_pos(e) -> str | None:
    return e.get("pos") if isinstance(e, dict) else None


def build_config(
    sc: dict, seed: int = 0, names=None
) -> tuple[DuelConfig, dict, list[str]]:
    """Build the puzzle DuelConfig + its record config dict. Returns (config, config_dict, errors)."""
    names = names or load_names()
    index = exact_index(names)
    errors: list[str] = []
    for key in ("id", "theme", "deck0", "deck1", "focus", "start", "p0", "p1"):
        if key not in sc:
            errors.append(f"missing key {key}")
    if errors:
        return None, None, errors
    if sc["start"] not in ("turn1", "turn2"):
        errors.append(f"bad start {sc['start']!r}")
    if sc["focus"] not in (0, 1):
        errors.append(f"bad focus {sc['focus']!r}")
    lp = sc.get("lp", [8000, 8000])
    if isinstance(lp, (list, tuple)):
        if len(lp) != 2:
            errors.append(f"bad lp {lp!r}")
            lp_val = 8000
        elif lp[0] != lp[1]:
            errors.append(f"unequal lp {lp!r}: the engine supports one LP for both players")
            lp_val = int(lp[0])
        else:
            lp_val = int(lp[0])
    else:
        lp_val = int(lp)
    decks = {}
    for side, key in ((0, "deck0"), (1, "deck1")):
        p = POOL / f"{sc[key]}.ydk"
        if not p.is_file():
            errors.append(f"unknown deck {sc[key]!r}")
            continue
        decks[side] = load_ydk(p)
    if errors:
        return None, None, errors

    placements: list[Placement] = []
    placed_codes: dict[int, list[int]] = {0: [], 1: []}
    for team in (0, 1):
        state = sc[f"p{team}"]
        for zone in ("hand", "mzone", "szone", "grave", "removed"):
            entries = state.get(zone, [])
            if not isinstance(entries, list):
                errors.append(f"p{team}.{zone} is not a list")
                continue
            for i, e in enumerate(entries):
                nm = _entry_name(e)
                code = resolve_exact(nm, names, index)
                if code is None:
                    errors.append(f"unknown card {nm!r} (p{team}.{zone}[{i}])")
                    continue
                placed_codes[team].append(code)
                pos_name = _entry_pos(e)
                if zone == "mzone":
                    seq = sum(1 for p in placements if p.team == team and p.location == C.LOCATION_MZONE)
                    if seq >= 5:
                        errors.append(f"too many mzone cards (p{team})")
                        continue
                    pos = MZONE_POS.get(pos_name or "atk", C.POS_FACEUP_ATTACK)
                    if pos_name and pos_name not in MZONE_POS:
                        errors.append(f"bad mzone pos {pos_name!r} (p{team}[{i}])")
                    placements.append(Placement(team, code, C.LOCATION_MZONE, seq, pos))
                elif zone == "szone":
                    seq = sum(1 for p in placements if p.team == team and p.location == C.LOCATION_SZONE)
                    if seq >= 5:
                        errors.append(f"too many szone cards (p{team})")
                        continue
                    pos = SZONE_POS.get(pos_name or "set", C.POS_FACEDOWN)
                    if pos_name and pos_name not in SZONE_POS:
                        errors.append(f"bad szone pos {pos_name!r} (p{team}[{i}])")
                    placements.append(Placement(team, code, C.LOCATION_SZONE, seq, pos))
                elif zone == "hand":
                    seq = sum(1 for p in placements if p.team == team and p.location == C.LOCATION_HAND)
                    placements.append(Placement(team, code, C.LOCATION_HAND, seq, 0))
                elif zone == "grave":
                    seq = sum(1 for p in placements if p.team == team and p.location == C.LOCATION_GRAVE)
                    placements.append(Placement(team, code, C.LOCATION_GRAVE, seq, 0))
                elif zone == "removed":
                    seq = sum(1 for p in placements if p.team == team and p.location == C.LOCATION_REMOVED)
                    placements.append(Placement(team, code, C.LOCATION_REMOVED, seq, 0))
    if errors:
        return None, None, errors

    # Every placed card must come from its side's own deck, within its copy count.
    reduced = {}
    for team in (0, 1):
        deck = decks[team]
        main = list(deck.main)
        extra = list(deck.extra)
        for code in placed_codes[team]:
            if code in main:
                main.remove(code)
            elif code in extra:
                extra.remove(code)
            else:
                have = Counter(deck.main + deck.extra)
                nm = names.names.get(code, str(code))
                if have.get(code, 0) <= 0:
                    errors.append(f"card not in deck: {nm!r} (p{team}, deck {sc[f'deck{team}']})")
                else:
                    errors.append(f"too many copies: {nm!r} (p{team}, deck {sc[f'deck{team}']})")
        reduced[team] = Deck(tuple(main), tuple(extra))
    if errors:
        return None, None, errors

    config = DuelConfig(reduced[0], reduced[1], seed=seed, lp=lp_val, hand=0, setup=tuple(placements))
    cfg_dict = {
        "deck0_main": list(reduced[0].main),
        "deck0_extra": list(reduced[0].extra),
        "deck1_main": list(reduced[1].main),
        "deck1_extra": list(reduced[1].extra),
        "seed": seed,
        "hand": 0,
        "draw": 1,
        "lp": lp_val,
        "setup": [[p.team, p.code, p.location, p.sequence, p.position] for p in placements],
        "shuffle": True,
    }
    return config, cfg_dict, []


def skip_option(obs) -> int:
    """A PASS / end-turn answer: TO_END first, then passive kinds, else the first legal option."""
    n = int(obs["action_mask"].sum())
    kinds = [int(obs["actions"][i][0]) for i in range(n)]
    legal = [i for i in range(n) if bool(obs["action_mask"][i])]
    order = list(legal)
    for want in _SKIP_WANTS:
        for i in legal:
            if kinds[i] == int(want):
                return i
    return order[0]


def random_option(obs, rng: random.Random) -> int:
    import numpy as np

    legal = np.flatnonzero(obs["action_mask"])
    return int(rng.choice(legal))


def validate_files(files: list[Path], engine: Engine, names) -> int:
    fails = 0
    for f in files:
        try:
            sc = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {f.name}: bad json ({e})")
            fails += 1
            continue
        config, _, errs = build_config(sc, seed=0, names=names)
        if errs:
            print(f"FAIL {sc.get('id', f.name)}: {errs[0]}")
            fails += 1
            continue
        try:
            env = CppEnv(engine, config)
            obs = env.reset()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {sc['id']}: engine error ({e!r}"[:300] + ")")
            fails += 1
            continue
        if obs is None:
            print(f"FAIL {sc['id']}: the duel ended immediately")
            fails += 1
            continue
        if sc["start"] == "turn2":
            steps = 0
            try:
                while obs is not None and int(obs["global"][2]) < 2:
                    obs = env.step(skip_option(obs))
                    steps += 1
                    if steps > MAX_DECISIONS:
                        break
                # The skipped turn may end on the opponent's decision; advance to focus on turn 2.
                guard = 0
                while obs is not None and not (
                    int(obs["global"][2]) >= 2 and int(env.player) == int(sc["focus"])
                ):
                    if int(obs["global"][2]) > 2:
                        break
                    obs = env.step(skip_option(obs))
                    guard += 1
                    if guard > MAX_DECISIONS:
                        break
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {sc['id']}: engine error in skipped turn ({e!r}"[:200] + ")")
                fails += 1
                continue
            if obs is None:
                print(f"FAIL {sc['id']}: the duel ended immediately (during the skipped turn)")
                fails += 1
                continue
            if int(obs["global"][2]) < 2 or int(env.player) != int(sc["focus"]):
                print(
                    f"FAIL {sc['id']}: no decision for player {sc['focus']} on turn 2 "
                    f"(turn {int(obs['global'][2])}, player {int(env.player)})"
                )
                fails += 1
                continue
        print(f"OK {sc['id']}")
    return fails


def play_files(
    files: list[Path],
    out: Path,
    games: int,
    policy: str,
    ckpt,
    device: str,
    seed: int,
    engine: Engine,
    names,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pol = None
    if policy == "bc":
        from ygo_zero.eval.bc_policy import BCPolicy

        pol = BCPolicy(ckpt, ROOT / "data/cards/emb/qwen3-0.6b-d512.npz", temperature=1.0, device=device)
    elif policy == "greedy":
        sys.path.insert(0, str(ROOT / "scripts"))
        from gen_search import greedy_logits as _greedy

        def pol(obs):  # type: ignore[no-redef]
            import numpy as np

            buf = {k: np.expand_dims(v, 0) for k, v in obs.items() if isinstance(v, np.ndarray)}
            return int(np.argmax(_greedy(buf)[0][obs["action_mask"]]))
    for f in files:
        sc = json.loads(f.read_text())
        sid = sc["id"]
        _, _, errs = build_config(sc, seed=seed, names=names)
        if errs:
            raise ValueError(f"{sid}: {errs[0]}")
        dest = out / f"{sid}.jsonl"
        if dest.exists():
            dest.unlink()
        for g in range(games):
            gseed = seed + g
            config, cfg_dict, _ = build_config(sc, seed=gseed, names=names)
            assert config is not None and cfg_dict is not None
            env = CppEnv(engine, config)
            obs = env.reset()
            choices: list[int] = []
            actors: list[int] = []
            turns: list[int] = []
            rng = random.Random(gseed * 1_000_003 + 17)
            argmax_game = g == 0
            steps = 0
            while obs is not None and steps < MAX_DECISIONS:
                actor, turn = int(env.player), int(obs["global"][2])
                if sc["start"] == "turn2" and turn < 2:
                    idx = skip_option(obs)
                elif policy == "random":
                    idx = random_option(obs, rng)
                elif policy == "greedy":
                    idx = pol(obs)
                else:  # bc: argmax for the first game, temperature-1 sampling after
                    import numpy as np

                    buf = {k: np.expand_dims(v.copy(), 0) for k, v in obs.items() if isinstance(v, np.ndarray)}
                    z = np.asarray(pol.logits(buf))[0].astype(np.float64)
                    mask = np.asarray(obs["action_mask"]).astype(bool)
                    z[~mask] = -np.inf
                    if argmax_game:
                        idx = int(np.argmax(z))
                    else:
                        z = z - np.nanmax(z)
                        p = np.exp(z)
                        p = p / p.sum()
                        idx = int(rng.choices(range(len(p)), weights=p)[0])
                choices.append(idx)
                actors.append(actor)
                turns.append(turn)
                obs = env.step(idx)
                steps += 1
            winner = int(env.winner) if obs is None and env.winner in (0, 1, 2) else (int(env.winner) if obs is None else 3)
            rec = {
                "game_id": f"sc-{sid}-{gseed}",
                "line_id": f"{sid}/sc-{gseed}",
                "decks": [sc["deck0"], sc["deck1"]],
                "config": cfg_dict,
                "auto_single": True,
                "choices": choices,
                "actors": actors,
                "turns": turns,
                "labels": ["skip"] * len(choices),
                "winner": winner,
                "scenario": sid,
                "theme": sc.get("theme"),
            }
            with dest.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"wrote {dest.name}: {games} games")


def list_cards(deck: str, names) -> None:
    p = POOL / f"{deck}.ydk"
    if not p.is_file():
        raise SystemExit(f"unknown deck {deck!r}")
    d = load_ydk(p)
    for label, codes in (("main", d.main), ("extra", d.extra)):
        for code, n in sorted(Counter(codes).items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{names.names.get(code, code)}\t{n}\t{label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("files", nargs="+", type=Path)
    p = sub.add_parser("play")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--policy", default="random", choices=["random", "bc", "greedy"])
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    lc = sub.add_parser("list-cards")
    lc.add_argument("deck")
    args = ap.parse_args()

    names = load_names()
    if args.cmd == "list-cards":
        list_cards(args.deck, names)
        return
    engine = Engine()
    if args.cmd == "validate":
        fails = validate_files(args.files, engine, names)
        if fails:
            print(f"{fails} FAIL(s)")
            raise SystemExit(1)
    elif args.cmd == "play":
        if args.policy == "bc" and args.ckpt is None:
            raise SystemExit("--ckpt is required with --policy bc")
        play_files(args.files, args.out, args.games, args.policy, args.ckpt, args.device, args.seed, engine, names)


if __name__ == "__main__":
    main()
