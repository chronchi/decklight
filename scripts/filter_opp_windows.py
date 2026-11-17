"""Seed-resampled filter for the opponent-turn window labels (NOTEBOOK 2026-11-04 19:30, option 2; user: "Sounds good").

`gen_opp_windows.py` decided each disagreement (search's move vs P's) from ONE deterministic play-out per arm; G1 of
the fine-tune on those labels moved the opponent's field the right way but not the games (NOTEBOOK 19:30). Here every
"search"/"p" label is played out again `--variants` times per arm, each variant with fresh randomness after the
window:
  - both seats' P samples at `--temp` from the window on (argmax before it, so the window is reached as recorded);
    the two arms of a variant share the sampling seed;
  - on "first" starts (decks in .ydk order, opening hands placed by `setup`) the main decks are also shuffled with a
    variant seed, so the cards still hidden at the window (both decks' order) are resampled. Only reshuffles whose
    engine replay of the recorded prefix reaches the identical window observation are used (up to `--tries`); when
    fewer than `--variants` reach (lines that draw or mill), the rest keep the recorded order (`topup`, sampling
    only). An arm that still leaves the recorded position is `diverged` and not counted.
Per variant the arm with the better `gen_opp_windows.outcome` (game result, then the opponent's field at the end of
the line's turn) wins; equal outcomes are ties. A label is KEPT for an arm when valid variants >= `--min-valid`, that
arm wins >= `--min-wins` variants and >= `--ratio` x the other arm's wins (pre-registered, NOTEBOOK 2026-11-04 20:00).
The kept arm can differ from the original label (`flipped`). Everything else becomes "none" (weight 0); "agree"
anchors are copied unchanged.

    uv run python scripts/filter_opp_windows.py --win runs/<opp-windows job>/window \
        --starts data/cache/line_starts/starts-v2.jsonl --shard 0/1 --out /tmp/f/s0.jsonl --max-labels 2
    # merge: --merge <dir>/s*.json --win ... --out <dir>/window/filtered.jsonl

Exit 3: no label played out, > 10% of labels failed, or a record does not match its start.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "audit"))
import gen_opp_windows as GW
import line_eval as LE
import window_read as WR

from decklight.demos.dataset import record_config
from decklight.engine.duel import Engine
from decklight.engine.env import PyEnv
from decklight.eval.bc_policy import BCPolicy
from decklight.search.product import resolve_product_paths

MIN_VALID = [8]  # set from --min-valid (summary)


class Resampled(WR.Forced):
    """`Forced`, and from the forced decision on the shared base policy samples at `temp` with `rng` (our seat, and the
    opponent once its script breaks). The forced decision must see exactly the recorded observation (`want`, every
    key), so a variant is only counted when the public position at the window is the recorded one."""

    def __init__(self, base, at, a, want: dict, temp: float, rng):
        super().__init__(base, at, a, np.asarray(want["actions"]))
        self.want, self.temp, self.rng = want, temp, rng

    def act(self, obs) -> int:
        if self.n == self.at and any(not np.array_equal(np.asarray(obs[x]), self.want[x]) for x in self.want):
            raise RuntimeError("forced decision reached a different position (observation)")
        a = super().act(obs)
        if self.hit and self.base.temperature != self.temp:
            self.base.temperature, self.base.rng = self.temp, self.rng
        return a


def training_starts(path: Path) -> list[dict]:
    """The deduped training starts, indexed as `gen_opp_windows` indexed them (`oppwin-<idx>-...`)."""
    seen, out = set(), []
    for x in path.read_text().splitlines():
        if x.strip() and x not in seen:
            seen.add(x)
            out.append(json.loads(x))
    return out


def variant_start(st: dict, v: int, idx: int) -> dict:
    s = copy.deepcopy(st)
    if s.get("kind") == "first" and not s["config"].get(
        "shuffle", False
    ):  # .ydk order: resample it per variant
        s["config"]["shuffle"] = True
        s["config"]["seed"] = (idx * 1_000_003 + v * 7919 + 1) & 0x7FFFFFFF
    return s


TOPUP = 10_000  # variant ids from here keep the recorded deck order


def same_obs(o, want: dict) -> bool:
    return o is not None and all(np.array_equal(np.asarray(o[x]), want[x]) for x in want)


def window_obs(engine, st: dict, prefix: list[int]):
    """The observation after replaying `prefix` from `st` (None if a recorded choice is no longer legal)."""
    env = PyEnv(engine, record_config(st), st.get("auto_single", True))
    obs = env.reset()
    for c in prefix:
        if obs is None or not obs["action_mask"][c]:
            return None
        obs = env.step(c)
    return obs


def reaching_variants(engine, st, idx, prefix, want, n: int, tries: int) -> tuple[list[int], int]:
    """Variant ids whose start replays the recorded prefix to the identical window observation (engine only, no
    model), the first `n` of up to `tries`, topped up with unshuffled ids (>= TOPUP: the recorded deck order, varied by
    sampling only) when fewer reach; and how many were tried. Starts that are not reshuffled always reach."""
    if variant_start(st, 0, idx)["config"] == st["config"]:
        return list(range(n)), n
    ok, v = [], 0
    while len(ok) < n and v < tries:
        if same_obs(window_obs(engine, variant_start(st, v, idx), prefix), want):
            ok.append(v)
        v += 1
    return ok + [TOPUP + j for j in range(n - len(ok))], v


def play_arm(engine, st, pol, at, a, want, temp, seed, prefix) -> dict | None:
    pol.temperature = 0.0
    f = Resampled(pol, at, a, want, temp, np.random.default_rng(seed))
    try:
        g = LE.play(engine, st, f, pol)
    except (
        RuntimeError
    ) as e:  # "forced decision reached a different position": the variant left the recorded line
        if "different position" in str(e):
            return None
        raise
    finally:
        pol.temperature = 0.0
    if not f.hit or [int(c) for c in g["choices"][: len(prefix)]] != prefix:
        return None
    return g


def decide(b: int, w: int, valid: int, a: argparse.Namespace) -> str | None:
    """'search' / 'p' when that arm wins clearly over the valid variants, else None."""
    if valid < a.min_valid:
        return None
    for kind, x, y in (("search", b, w), ("p", w, b)):
        if x >= a.min_wins and x >= a.ratio * y:
            return kind
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--variants", type=int, default=12)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument(
        "--tries", type=int, default=60, help="reshuffles tried to find --variants that reach the window"
    )
    ap.add_argument("--min-valid", type=int, default=8)
    ap.add_argument("--min-wins", type=int, default=6)
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--max-labels", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--merge", nargs="*", type=Path, default=None)
    ap.add_argument("--expect-parts", type=int, default=0, help="merge: shard files that must exist")
    ap.add_argument(
        "--expect-labels", type=int, default=0, help="merge: non-agree labels the input must hold"
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    MIN_VALID[0] = a.min_valid
    recs = [
        json.loads(x)
        for f in sorted(glob.glob(str(a.win / "*.jsonl")))
        for x in Path(f).read_text().splitlines()
        if x.strip()
    ]
    if a.merge is not None:
        return merge(a, recs)
    starts = training_starts(a.starts)
    jobs = [(ri, k) for ri, r in enumerate(recs) for k, inf in sorted(r["window_info"].items(), key=lambda t: int(t[0]))
            if inf["kind"] != "agree"]  # fmt: skip
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = jobs[i::n][: a.max_labels or None]
    ckpt, _, emb = resolve_product_paths(None, None, None)
    pol, engine = BCPolicy(ckpt, emb, threads=1, device=a.device), Engine()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows, failed, bad = [], 0, 0
    for ji, (ri, ks) in enumerate(jobs):
        r, k = recs[ri], int(ks)
        inf = r["window_info"][ks]
        idx = int(r["line_id"].split("/oppwin-")[1].split("-")[0])
        st = starts[idx]
        if st["line_id"] != r["start_line_id"] or st["config"] != r["config"]:
            print(f"ALERT filter_opp_windows: record {r['line_id']} does not match start {idx}", flush=True)
            bad += 1
            continue
        try:
            me, prefix = int(st["our_seat"]), [int(c) for c in r["choices"][:k]]
            obs = window_obs(engine, st, prefix)
            if obs is None:
                raise RuntimeError("the recorded prefix does not replay")
            want = {x: np.asarray(v).copy() for x, v in obs.items()}
            at = WR.our_calls_before(engine, st, r["choices"], k, me)
            res = {"search": 0, "p": 0, "tie": 0, "diverged": 0}
            vs, tried = reaching_variants(engine, st, idx, prefix, want, a.variants, a.tries)
            res["unreached"], res["topup"] = tried - sum(v < TOPUP for v in vs), sum(v >= TOPUP for v in vs)
            for v in vs:
                sv = st if v >= TOPUP else variant_start(st, v, idx)
                g = {nm: play_arm(engine, sv, pol, at, inf[nm], want, a.temp, [idx, k, v], prefix)
                     for nm in ("search", "p")}  # fmt: skip
                if g["search"] is None or g["p"] is None:
                    res["diverged"] += 1
                    continue
                if g["search"]["cut"] or g["p"]["cut"]:
                    res["diverged"] += 1
                    continue
                os_, op = GW.outcome(g["search"]), GW.outcome(g["p"])
                res["search" if os_ > op else "p" if os_ < op else "tie"] += 1
            valid = len(vs) - res["diverged"]
            keep = decide(res["search"], res["p"], valid, a)
            rows.append({"rec": r["line_id"], "k": k, "orig": inf["kind"], "keep": keep, "valid": valid, **res,
                         "start_kind": st.get("kind")})  # fmt: skip
        except Exception as e:  # noqa: BLE001  counted; > 10% fails the shard
            failed += 1
            print(f"label {r['line_id']} k={k} failed: {e!r}", flush=True)
        print(f"{ji + 1}/{len(jobs)} {rows[-1] if rows else ''}", flush=True)
        a.out.write_text(
            json.dumps({"rows": rows, "failed": failed, "bad": bad, "jobs": len(jobs), "partial": True})
        )
    a.out.write_text(json.dumps({"rows": rows, "failed": failed, "bad": bad, "jobs": len(jobs)}, indent=1))
    print(json.dumps(summary(rows)), flush=True)
    return (
        3 if bad or (jobs and failed == len(jobs)) else 0
    )  # the merge applies the 10% tolerance over all labels


def summary(rows: list[dict]) -> dict:
    s = {"labels": len(rows), "kept": sum(r["keep"] is not None for r in rows),
         "kept_search": sum(r["keep"] == "search" for r in rows), "kept_p": sum(r["keep"] == "p" for r in rows),
         "flipped": sum(r["keep"] is not None and r["keep"] != r["orig"] for r in rows),
         "too_few_valid": sum(r["keep"] is None and r["valid"] < MIN_VALID[0] for r in rows)}  # fmt: skip
    for kind in ("first", "second"):
        rk = [r for r in rows if r["start_kind"] == kind]
        s[f"{kind}_labels"], s[f"{kind}_kept"] = len(rk), sum(r["keep"] is not None for r in rk)
        s[f"{kind}_diverged_mean"] = round(sum(r["diverged"] for r in rk) / max(len(rk), 1), 2)
    return s


def merge(a: argparse.Namespace, recs: list[dict]) -> int:
    parts = [json.loads(f.read_text()) for f in a.merge]
    rows = [r for p in parts for r in p["rows"]]
    jobs, failed, bad = (
        sum(p["jobs"] for p in parts),
        sum(p["failed"] for p in parts),
        sum(p["bad"] for p in parts),
    )
    dec = {(r["rec"], r["k"]): r for r in rows}
    out, n_lab = [], {"agree": 0, "search": 0, "p": 0}
    for r in recs:
        r = copy.deepcopy(r)
        for ks, inf in list(r["window_info"].items()):
            k = int(ks)
            if inf["kind"] == "agree":
                n_lab["agree"] += 1
                continue
            d = dec.get((r["line_id"], k))
            if d is None or d["keep"] is None:  # dropped (or never played out): weight 0, kept for the record
                r["labels"][k], r["targets"][k] = "none", None
                inf["filter"] = d and {
                    x: d[x] for x in ("search", "p", "tie", "diverged", "unreached", "topup")
                }
                inf["kind"] = "dropped"
                continue
            r["targets"][k] = int(inf[d["keep"]])
            inf["filter"] = {x: d[x] for x in ("search", "p", "tie", "diverged", "unreached", "topup")}
            inf["kind_orig"], inf["kind"] = inf["kind"], d["keep"]
            n_lab[d["keep"]] += 1
        if any(lab == "window" for lab in r["labels"]):
            out.append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    s = {
        **summary(rows),
        "jobs": jobs,
        "failed": failed,
        "bad": bad,
        "records": len(out),
        "labels_out": n_lab,
    }
    a.out.with_suffix(".summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)
    want = sum(inf["kind"] != "agree" for r in recs for inf in r["window_info"].values())
    ok = (rows and not bad and len(parts) == a.expect_parts and jobs == want == a.expect_labels
          and len(rows) + failed == jobs and failed <= 0.1 * jobs and n_lab["search"] + n_lab["p"] > 0)  # fmt: skip
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
