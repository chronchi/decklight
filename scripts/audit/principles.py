"""Principle audit of training data and games (retrain plan v2 §3.1 / stage D; docs/play_principles.md).

Engine replay only (laptop OK under `taskset -c 0-3`). Card classes come from the engine's Lua scripts and are used for
auditing only; the policy never sees a rule. A "hand trap" here = a monster with a Quick Effect whose range includes the
hand (Lua QUICK_O/F with LOCATION_HAND): Maxx "C", Ash Blossom, Mulcharmy Fuwalos/Purulia, Ghost Ogre, ...

Per source it reports, as fires / opportunities (an opportunity = a decision where such an option was legal):
- ht_own_t1 / ht_own_t2p: hand trap activated on its own turn with no opponent chain link to answer (turn 1 vs later);
- ht_opp: hand trap activated on the opponent's turn (the intended use; anti-passivity reference);
- empty_draw: any activation at an empty-chain Draw/Standby window (own / opponent turn);
- interrupt_opp: any activation at an opponent-turn chain window where activating was legal (take rate);
- ht_summoned: hand-trap monsters Normal Summoned or Set (opportunities = decisions where that was legal);
- inferred: share of `inferred` labels among labelled decisions (data sources only).
It also writes the per-line drop list (`--drops`): lines that fire a hand trap on their own turn with no opponent chain
link (retrain plan v2 stage D per-line principle drop), with the card names, for review before use.

Sources: `--records <glob> ...` (BC record files, any kind; decisions = labelled samples via demos.dataset.iter_samples)
or `--games <dir> ...` (search_match/playtest games-*.jsonl: decisions of the seat `s_seat`, i.e. our pilot; with
`--seat ya` the other seat, i.e. ygo-agent in `--opp ygoagent` cards).

Usage:
    taskset -c 0-3 uv run python scripts/audit/principles.py --name teacher --records 'data/demos/teacher/*.jsonl' \
        --max-files 40 --out data/cache/audit/principles-teacher.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ygo_zero.cards.lua_effects import OFFICIAL_DIR, load_constants, parse_script  # noqa: E402
from ygo_zero.engine import constants as C  # noqa: E402
from ygo_zero.engine.actions import Kind  # noqa: E402
from ygo_zero.engine.duel import Engine  # noqa: E402

QUICK, HAND = 0x100 | 0x400, 0x2
LOC_HAND = 2  # obs/action location numbering (docs/obs_spec.md)
DRAW_STANDBY = (0x1, 0x2)


class HandTraps:
    """Lazy Lua classification: monster card with a quick effect usable from the hand."""

    def __init__(self, engine: Engine) -> None:
        self.engine, self.consts, self.cache = engine, load_constants(), {}

    def __call__(self, code: int) -> bool:
        code = int(code)
        if code not in self.cache:
            cd = self.engine.cards.get(code)
            ok = False
            p = OFFICIAL_DIR / f"c{code}.lua"
            if cd is not None and cd.type & C.TYPE_MONSTER and p.is_file():
                recs = parse_script(p.read_text(encoding="utf-8", errors="replace"), code, self.consts)
                ok = any(r.type & QUICK and r.range & HAND for r in recs)
            self.cache[code] = ok
        return self.cache[code]


class Audit:
    def __init__(self, ht: HandTraps, names) -> None:
        self.ht, self.names = ht, names
        self.n = Counter()  # metric -> opportunities
        self.k = Counter()  # metric -> fires
        self.cards = defaultdict(Counter)  # metric -> code -> fires
        self.labels = Counter()
        self.drops: dict[str, Counter] = defaultdict(Counter)  # line id -> hand-trap codes fired on own turn

    def decision(self, obs: dict, a: int, line: str | None = None, label: str | None = None) -> None:
        g = obs["global"]
        acts, mask = obs["actions"], obs["action_mask"].astype(bool)
        legal = [i for i in range(len(mask)) if mask[i]]
        if not legal or a not in legal:
            return
        if label is not None:
            self.labels[label] += 1
        own, turn, phase, msg = int(g[5]) == 1, int(g[2]), int(g[3]), int(g[21])
        cm = obs["chain_mask"].astype(bool)
        opp_link = bool((obs["chain"][cm][:, 4] == 1).any()) if cm.any() else False
        kind = lambda i: int(acts[i, 0])  # noqa: E731
        code = lambda i: int(acts[i, 2])  # noqa: E731
        is_ht_act = lambda i: kind(i) == Kind.ACTIVATE and int(acts[i, 7]) == LOC_HAND and self.ht(code(i))  # noqa: E731
        ht_opts = [i for i in legal if is_ht_act(i)]
        if ht_opts:
            if own and not opp_link:
                m = "ht_own_t1" if turn == 1 else "ht_own_t2p"
            elif not own:
                m = "ht_opp"
            else:
                m = "ht_own_answer"  # own turn, answering an opponent chain link
            self.n[m] += 1
            if a in ht_opts:
                self.k[m] += 1
                self.cards[m][code(a)] += 1
                if m in ("ht_own_t1", "ht_own_t2p") and line is not None:
                    self.drops[line][code(a)] += 1
        acts_legal = [i for i in legal if kind(i) == Kind.ACTIVATE]
        if msg == C.MSG_SELECT_CHAIN and acts_legal:
            if phase in DRAW_STANDBY and int(g[20]) == 0:
                m = "empty_draw_own" if own else "empty_draw_opp"
                self.n[m] += 1
                if kind(a) == Kind.ACTIVATE:
                    self.k[m] += 1
                    self.cards[m][code(a)] += 1
            if not own:
                self.n["interrupt_opp"] += 1
                if kind(a) == Kind.ACTIVATE:
                    self.k["interrupt_opp"] += 1
        summ = [i for i in legal if kind(i) in (Kind.SUMMON, Kind.MSET) and self.ht(code(i))]
        if summ:
            self.n["ht_summoned"] += 1
            if a in summ:
                self.k["ht_summoned"] += 1
                self.cards["ht_summoned"][code(a)] += 1

    def report(self) -> dict:
        out = {}
        for m in sorted(set(self.n) | set(self.k)):
            n, k = self.n[m], self.k[m]
            out[m] = {"fires": k, "opps": n, "rate": round(k / n, 4) if n else None,
                      "top": [(self.names(c), v) for c, v in self.cards[m].most_common(8)]}
        lab = sum(self.labels.values())
        if lab:
            out["labels"] = dict(self.labels)
            out["inferred_share"] = round(self.labels["inferred"] / lab, 4)
        out["lines_with_own_turn_ht"] = len(self.drops)
        out["card_counts"] = {m: {str(c): v for c, v in cs.items()} for m, cs in self.cards.items()}  # full, by code
        return out


def iter_games(dirs: list[Path], engine: Engine, seat: str):
    from ygo_zero.decks import load_ydk
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import DuelConfig

    for d in dirs:
        for f in sorted(d.glob("games-*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                me = int(r["s_seat"]) if seat == "s" else 1 - int(r["s_seat"])
                ddir = ROOT / ("decks/gauntlet" if (ROOT / "decks/gauntlet" / f"{r['decks'][0]}.ydk").exists()
                               else "decks/pool")
                cfg = DuelConfig(load_ydk(ddir / f"{r['decks'][0]}.ydk"), load_ydk(ddir / f"{r['decks'][1]}.ydk"),
                                 seed=int(r["seed"]))
                env = CppEnv(engine, cfg, True)
                obs = env.reset()
                for c in r["choices"]:
                    if obs is None:
                        break
                    if int(env.player) == me:
                        yield obs, int(c), f"{f.name}:{r.get('game_idx')}", None
                    obs = env.step(int(c))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--records", nargs="*", default=[])
    ap.add_argument("--games", nargs="*", type=Path, default=[])
    ap.add_argument("--seat", choices=("s", "ya"), default="s")
    ap.add_argument("--strict", action="store_true", help="iter_samples strict label weights (weight-0 labels skipped)")
    ap.add_argument("--max-files", type=int, default=0, help="random sample of record files (0 = all)")
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drops", type=Path, default=None, help="write the per-line drop list (jsonl)")
    args = ap.parse_args()

    from ygo_zero.cards.embed import card_texts

    engine = Engine()
    texts = card_texts()
    audit = Audit(HandTraps(engine), lambda c: texts.get(int(c), (str(c), ""))[0] or str(c))
    t0, n = time.time(), 0
    if args.records:
        from ygo_zero.demos.dataset import iter_samples

        files = sorted({Path(f) for g in args.records for f in glob.glob(g)})
        if args.max_files:
            random.Random(0).shuffle(files)
            files = files[: args.max_files]
        for f in files:
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                break
            try:
                for obs, a, w, meta in iter_samples(engine, f, strict=args.strict):
                    if args.strict and w == 0:
                        continue
                    audit.decision(obs, int(a), meta.get("line_id"), meta.get("label"))
                    n += 1
            except Exception as e:  # a record file that no longer replays is reported, not fatal
                print(f"skip {f}: {e}", file=sys.stderr)
    for obs, a, gid, _ in iter_games(args.games, engine, args.seat):
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            break
        audit.decision(obs, a, gid)
        n += 1
    if n == 0:
        print(f"ALERT principles: 0 decisions audited for {args.name}")
        return 2
    rep = {"name": args.name, "decisions": n, "seconds": round(time.time() - t0), **audit.report()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1))
    if args.drops is not None:
        with args.drops.open("w") as fh:
            for lid, cs in sorted(audit.drops.items()):
                fh.write(json.dumps({"line": lid, "cards": {audit.names(c): v for c, v in cs.items()}}) + "\n")
    short = {m: (v["fires"], v["opps"], v["rate"]) for m, v in rep.items() if isinstance(v, dict) and "rate" in v}
    print(json.dumps({"name": args.name, "decisions": n, **short, "inferred_share": rep.get("inferred_share")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
