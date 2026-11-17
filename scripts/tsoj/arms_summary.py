#!/usr/bin/env python3
"""Pair a forced-activation arm against P on the same openings, and summarise.

    uv run python scripts/tsoj/arms_summary.py TAG [ARM ...]

Reads `runs/tsoj-arms/TAG/<arm>.jsonl` (arbiter records) and `arms.jsonl` (one line per
client game). Pairs games by opening (seed + both deck names) and seat order. For each pair
it checks the arbiter responses are identical up to the forced decision (determinism),
then reports win(arm) - win(p) with a paired bootstrap CI and a sign test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OURS = {"skydeck-tcg2026-10", "light-magistus-f10-tcg2026-10"}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def canon(r: dict) -> str:
    return json.dumps(r.get("response"), sort_keys=True, default=str)


def first_diff(a: list[dict], b: list[dict]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x.get("player") != y.get("player") or canon(x) != canon(y):
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def p_resp_ordinal(rec: dict, pseat: int, pos: int) -> int:
    """How many of P's responses are before response index `pos`."""
    return sum(1 for r in rec.get("responses", [])[:pos] if r.get("player") == pseat)


def bad_result(rec: dict) -> bool:
    reason = (rec.get("result") or {}).get("reason")
    return isinstance(reason, str) and reason.startswith(("bot_error", "illegal"))


def groups(recs: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        names = tuple(d.get("name") for d in r.get("decks", []))
        # one game per key: the arbiter writes games in finishing order, so the seat order (`agents`) is part of the key
        out[(int(r.get("seed", 0)), *names, tuple(r.get("agents") or ()))].append(r)
    return out


def bootstrap_ci(x, n_boot: int = 4000, seed: int = 0):
    x = np.asarray(x, float)
    if not len(x):
        return None
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def sign_p(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    return round(sum(math.comb(n, k) for k in range(better, n + 1)) / 2**n, 4)


def win(rec: dict, pseat: int) -> float:
    w = (rec.get("result") or {}).get("winner")
    if w is None:
        return 0.5
    return 1.0 if int(w) == pseat else 0.0


def opening_hand(rec: dict, pseat: int) -> list[int]:
    """Our starting hand: the top 5 of the record's deck order (bottom first)."""
    order = (rec.get("deck_orders") or [[], []])[pseat][-5:]
    return sorted(int(c) for c in reversed(order))


def find_log(logs: list[dict], rec: dict, pseat: int, used: set[int]) -> dict | None:
    pdeck = rec["decks"][pseat].get("name")
    winner = (rec.get("result") or {}).get("winner")
    turns = (rec.get("result") or {}).get("turns")
    hand = opening_hand(rec, pseat)
    for i, l in enumerate(logs):
        if i in used:
            continue
        if (l.get("seat") == pseat and l.get("deck") == pdeck
                and l.get("winner") == winner and l.get("turns") == turns
                and (not l.get("hand") or l.get("hand") == hand)):
            used.add(i)
            return l
    return None


def stats(rows: list[dict]) -> dict:
    diffs = [r["diff"] for r in rows]
    better = sum(d > 0 for d in diffs)
    worse = sum(d < 0 for d in diffs)
    return {
        "pairs": len(rows),
        "win_p": round(float(np.mean([r["p"] for r in rows])), 4) if rows else None,
        "win_arm": round(float(np.mean([r["arm"] for r in rows])), 4) if rows else None,
        "diff": round(float(np.mean(diffs)), 4) if rows else None,
        "ci95": bootstrap_ci(diffs),
        "arm_better": better, "arm_worse": worse, "sign_p": sign_p(better, worse),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = ROOT / "runs" / "tsoj-arms" / args.tag
    p_path = d / "p.jsonl"
    if not p_path.is_file():
        print(f"missing {p_path}", file=sys.stderr)
        return 3
    p_recs = load(p_path)
    p_groups = groups(p_recs)
    all_logs = load(d / "arms.jsonl") if (d / "arms.jsonl").is_file() else []
    arms = args.arms or [p.stem for p in sorted(d.glob("*.jsonl"))
                         if p.stem not in ("p", "p2", "arms") and not p.stem.startswith("ALERT")]
    report = {"tag": args.tag, "arms": {}}
    p2_path = d / "p2.jsonl"
    if p2_path.is_file():
        # Determinism (independent of the arms): a second run of arm p must repeat every game.
        same = diff = 0
        p2_groups = groups(load(p2_path))
        for key, prs in p2_groups.items():
            for s, rec in enumerate(prs[:1]):
                if key not in p_groups:
                    continue
                if bad_result(rec) or bad_result(p_groups[key][s]):
                    report["p_repeat_bad_skipped"] = report.get("p_repeat_bad_skipped", 0) + 1
                    continue  # a crashed game is not a determinism failure (run 5: 1 bot_error in p)
                if first_diff(p_groups[key][s].get("responses", []), rec.get("responses", [])) is None:
                    same += 1
                else:
                    diff += 1
        report["p_repeat_identical"] = f"{same}/{same + diff}"
        if diff:
            report["p_repeat_alert"] = "arm p is not deterministic: paired diffs are invalid"
    for arm in arms:
        a_path = d / f"{arm}.jsonl"
        if not a_path.is_file():
            report["arms"][arm] = {"error": f"missing {a_path.name}"}
            continue
        a_groups = groups(load(a_path))
        kind, _, kraw = arm.partition("@")
        k = int(kraw) if kraw.isdigit() else None
        a_logs = [l for l in all_logs
                  if l.get("arm") == kind and (k is None or l.get("k") == k)]
        used: set[int] = set()
        valid, earlier, no_log, skipped_orientation, unmatched = [], 0, 0, 0, 0
        prefix_ok = prefix_n = 0
        by_our, by_opp, by_card, by_turn = (defaultdict(list) for _ in range(4))
        for key, prs in p_groups.items():
            if key not in a_groups:
                continue
            for _ in (0,):
                prec, arec = prs[0], a_groups[key][0]
                pseat = list(prec.get("agents") or ["ab", "dl"]).index("dl")
                pdeck = prec["decks"][pseat].get("name")
                oppdeck = prec["decks"][1 - pseat].get("name")
                log = find_log(a_logs, arec, pseat, used)
                if log is None:
                    no_log += 1
                pos = first_diff(prec.get("responses", []), arec.get("responses", []))
                forced = bool(log and log.get("forced"))
                ok = True
                if log is not None:
                    if pos is None:
                        ok = not forced
                    elif forced:
                        # The arbiter auto-passes hidden empty-chain windows without telling
                        # the client, so the record's P-response ordinal of the forced decision
                        # is >= the client's own decision index (never smaller).
                        before = p_resp_ordinal(arec, pseat, pos)
                        p_resp = (prec.get("responses") or [{}] * (pos + 1))[pos].get("response", {})
                        ok = (arec["responses"][pos].get("player") == pseat
                              and before >= int(log["decision"])
                              and p_resp.get("index") == -1)
                elif pos is not None:
                    ok = False  # no client log: cannot validate the divergence point
                if not ok:
                    earlier += 1
                if log is not None:
                    prefix_n += 1
                    prefix_ok += int(ok)
                if pdeck not in OURS:
                    skipped_orientation += 1  # P holds the opponent's deck: kept, split by_our_deck
                if log is None:
                    unmatched += 1
                    continue
                row = {"p": win(prec, pseat), "arm": win(arec, pseat),
                       "diff": win(arec, pseat) - win(prec, pseat), "log": log}
                valid.append(row)
                by_our[pdeck].append(row)
                by_opp[oppdeck].append(row)
                by_card[log.get("card") or "(none)"].append(row)
                by_turn[str(log.get("turn"))].append(row)
        if not valid:
            report["arms"][arm] = {"error": "0 valid pairs", "no_log": no_log,
                                   "earlier_divergence": earlier}
            continue
        forced = sum(1 for r in valid if r["log"].get("forced"))
        report["arms"][arm] = {
            **stats(valid),
            "prefix_identical": f"{prefix_ok}/{prefix_n}",
            "earlier_divergence": earlier,
            "no_log": no_log,
            "unmatched_logs": unmatched,
            "skipped_orientation": skipped_orientation,
            "forced_share": round(forced / len(valid), 4),
            "by_our_deck": {k: stats(v) for k, v in sorted(by_our.items())},
            "by_opp_deck": {k: stats(v) for k, v in sorted(by_opp.items())},
            "by_forced_card": {k: stats(v) for k, v in sorted(by_card.items())},
            "by_turn": {k: stats(v) for k, v in sorted(by_turn.items(), key=lambda kv: str(kv[0]))},
        }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if "p_repeat_alert" in report:
        return 3
    return 0 if all("error" not in v for v in report["arms"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
