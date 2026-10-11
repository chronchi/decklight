"""Encoding diagnostics for the retrain plan (NOTEBOOK 2026-10-11): what does the network see and use?

Stages (`--stage`):
- collect (CPU, engine only): replay search_match games; keep every empty-chain window that offers an activation plus a
  random share of other decisions. Per state: the obs (bc_data.OBS_KEYS) and labels: whose turn, phase, went first,
  turn, Normal Summon used this turn by the actor, the actor's activations this turn, and per card row "this card's
  code was activated by the actor this turn" (a once-per-turn-used proxy). -> <out>/states-<proc>.npz
- probe (GPU): for the policy and the value net,
  (a) P(activate) at empty-chain windows by whose turn x turn 1 x phase, and how far it moves when a single input is
      flipped (whose turn, turn 1<->3, Draw<->Main 1 phase): does the network use those inputs?
  (b) linear probes (logistic regression, split by game) on the global token + mean token for each label, and on the
      card-row tokens for "activated this turn": is the fact in the representation? Baseline: the same probe on the
      raw global vector.
- text (CPU): linear probes on the frozen card-text embeddings (card vector, mean effect vector, per-effect vectors)
  for timing labels (scripts/diag/text_labels.py: Lua ground truth per card, PSCT punctuation per effect sentence),
  against a bag-of-words baseline on the same text. Does the embedding keep the timing that the text states?

    python scripts/diag/encoding_diag.py --stage collect --games data/cache/diag/games/raw ... --out runs/<n>/diag
        [--nproc 16 --proc 0] [--keep 0.25]
    python scripts/diag/encoding_diag.py --stage probe --out runs/<n>/diag --policy P.pt --value V.pt --emb E.npz
    python scripts/diag/encoding_diag.py --stage text --out runs/<n>/diag --emb E.npz --labels data/cache/diag/text_labels.npz
Results: <out>/probe.json, <out>/text.json (+ printed tables). Invariant failures print "ALERT ..." and exit 2.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

KIND_SUMMON, KIND_MSET, KIND_ACTIVATE, KIND_PASS = 1, 4, 6, 11
MSG_SELECT_CHAIN = 0x10
PH_DRAW, PH_MAIN1 = 0x1, 0x4
LABELS = ("my_turn", "went_first", "turn1", "ns_used", "any_act")


def alert(msg: str) -> None:
    print(f"ALERT encoding_diag: {msg}", flush=True)


# ---------------------------------------------------------------- collect
def collect(a) -> int:
    import show_divergence as S

    from ygo_zero.demos.game import _opts
    from ygo_zero.engine.cpp_env import CppEnv
    from ygo_zero.engine.duel import Engine
    from ygo_zero.model.bc_data import OBS_KEYS

    games = []
    for d in a.games:
        games += [g for _, g in sorted(S.load(d).items())]
    games = games[a.proc :: a.nproc]
    rng = np.random.default_rng(1000 + a.proc)
    eng = Engine()
    obs_buf: dict[str, list] = {k: [] for k in OBS_KEYS}
    meta: dict[str, list] = {k: [] for k in ("game", "seat", "turn", "phase", "my_turn", "went_first", "window",
                                             "ns_used", "n_act")}  # fmt: skip
    rowlab, actm, passm = [], [], []
    gid0 = a.proc * 100000
    for gi, g in enumerate(games):
        env = CppEnv(eng, S._config(g), True)
        obs = env.reset()
        for c in g["choices"]:
            if obs is None:
                break
            o = _opts(obs)
            glob_ = obs["global"]
            window = (int(glob_[21]) == MSG_SELECT_CHAIN and int(glob_[20]) == 0
                      and any(k == KIND_ACTIVATE for k, _ in o))  # fmt: skip
            early = window and int(glob_[2]) <= 2
            if len(o) > 1 and (early or rng.random() < (a.keep_window if window else a.keep)):
                p, turn = int(env.player), int(glob_[2])
                hl = env.history_log()
                mine = hl[(hl[:, 10] == p) & (hl[:, 11] == turn)] if len(hl) else hl
                acts = mine[mine[:, 0] == KIND_ACTIVATE] if len(mine) else mine
                codes = {int(x) for x in acts[:, 2]} if len(acts) else set()
                for k in OBS_KEYS:
                    obs_buf[k].append(obs[k])
                cards, cm = obs["cards"], obs["card_mask"]
                rl = np.full(len(cm), -1, np.int8)
                for i in np.flatnonzero(cm):
                    if int(cards[i, 3]) == 0 and int(cards[i, 1]) in (2, 3, 4, 5) and int(cards[i, 0]):
                        rl[i] = int(int(cards[i, 0]) in codes)
                rowlab.append(rl)
                am = np.zeros(len(obs["action_mask"]), bool)
                pm = np.zeros(len(obs["action_mask"]), bool)
                for i, (k, _) in enumerate(o):
                    am[i] = k == KIND_ACTIVATE
                    pm[i] = k == KIND_PASS
                actm.append(am)
                passm.append(pm)
                for k, v in (("game", gid0 + gi), ("seat", p), ("turn", turn), ("phase", int(glob_[3])),
                             ("my_turn", int(glob_[5])), ("went_first", int(glob_[4])), ("window", int(window)),
                             ("ns_used", int(len(mine) > 0 and bool(np.isin(mine[:, 0], (KIND_SUMMON, KIND_MSET)).any()))),
                             ("n_act", len(acts))):  # fmt: skip
                    meta[k].append(v)
            obs = env.step(int(c))
    n = len(meta["game"])
    if n == 0:
        alert(f"proc {a.proc}: 0 states collected from {len(games)} games")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"states-{a.proc}.npz", **{f"obs_{k}": np.stack(v) for k, v in obs_buf.items()},
                        **{k: np.array(v, np.int32) for k, v in meta.items()}, rowlab=np.stack(rowlab),
                        actm=np.stack(actm), passm=np.stack(passm))  # fmt: skip
    print(f"proc {a.proc}: {len(games)} games, {n} states, {sum(meta['window'])} windows", flush=True)
    return 0


# ---------------------------------------------------------------- probe helpers
def load_states(out: Path) -> dict[str, np.ndarray]:
    files = sorted(out.glob("states-*.npz"))
    if not files:
        raise SystemExit(f"no states-*.npz under {out}")
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def logreg(X: np.ndarray, y: np.ndarray, groups: np.ndarray, multiclass: bool = False, max_n: int = 60000) -> dict:
    """Held-out (split by game) logistic-regression probe: AUC for binary labels, accuracy (vs majority) otherwise."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    if len(y) > max_n:
        idx = rng.choice(len(y), max_n, replace=False)
        X, y, groups = X[idx], y[idx], groups[idx]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "skip": "one class"}
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, y, groups))
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[tr]), y[tr])
    Xt = sc.transform(X[te])
    if multiclass:
        acc = float((m.predict(Xt) == y[te]).mean())
        maj = float((y[te] == np.bincount(y[tr]).argmax()).mean())
        return {"n": int(len(y)), "acc": round(acc, 4), "majority": round(maj, 4)}
    if len(np.unique(y[te])) < 2:
        return {"n": int(len(y)), "skip": "one class in test"}
    auc = float(roc_auc_score(y[te], m.predict_proba(Xt)[:, 1]))
    return {"n": int(len(y)), "pos": round(float(y.mean()), 4), "auc": round(auc, 4)}


def probe(a) -> int:
    import torch

    from ygo_zero.model.policy import PolicyNet

    out = Path(a.out)
    st = load_states(out)
    n = len(st["game"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_keys = [k[4:] for k in st if k.startswith("obs_")]
    emb = dict(np.load(a.emb))
    res: dict = {"n_states": int(n), "n_windows": int(st["window"].sum())}
    print(f"{n} states, {res['n_windows']} empty-chain activation windows", flush=True)
    for name, ckpt in (("policy", a.policy), ("value", a.value), ("random_init", "RANDOM")):
        if not ckpt:
            continue
        if ckpt == "RANDOM":  # control: an untrained net of the same shape (what probes find without learning)
            torch.manual_seed(0)
            has_v = False
            net = PolicyNet(emb, d=a.d, layers=4, heads=4)
        else:
            sd = torch.load(ckpt, map_location="cpu")
            sd = sd.get("model", sd)
            has_v = any(k.startswith("value_head") for k in sd)
            has_q = any(k.startswith("q_head") for k in sd)
            net = PolicyNet(emb, d=a.d, layers=4, heads=4, value=has_v, q=has_q)
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        net.to(dev).eval()

        def run(sel: np.ndarray, glob_override=None):
            """(P(activate) or V, global+mean features, row-token features + labels) for states `sel`."""
            pa, feats, rows_f, rows_y, rows_g = [], [], [], [], []
            for i in range(0, len(sel), a.batch):
                b = sel[i : i + a.batch]
                t = {k: torch.from_numpy(np.ascontiguousarray(st["obs_" + k][b])).to(dev) for k in obs_keys}
                if glob_override is not None:
                    t["global"] = torch.from_numpy(glob_override(st["obs_global"][b].copy())).to(dev)
                with torch.no_grad():
                    tt, enc, keep, n_opt = net.encode(t)
                    if has_v and name == "value":
                        score = net.value_head(enc[:, 0]).squeeze(-1)
                    else:
                        lg = net(t).float()
                        lg = torch.where(t["action_mask"].bool(), lg, torch.full_like(lg, -1e9))
                        pr = torch.softmax(lg, -1)
                        score = (pr * torch.from_numpy(st["actm"][b]).to(dev)).sum(-1)
                    pa.append(score.float().cpu().numpy())
                    if glob_override is None:
                        km = keep.float().unsqueeze(-1)
                        pooled = (enc * km).sum(1) / km.sum(1)
                        feats.append(torch.cat([enc[:, 0], pooled], -1).half().cpu().numpy())
                        lc = tt["cards"].shape[1]
                        rl = st["rowlab"][b][:, :lc]
                        bi, ri = np.nonzero(rl >= 0)
                        if len(bi):
                            rows_f.append(enc[torch.from_numpy(bi).to(dev), torch.from_numpy(ri + 1).to(dev)].half().cpu().numpy())
                            rows_y.append(rl[bi, ri])
                            rows_g.append(st["game"][b][bi])
            cat = (lambda x: np.concatenate(x) if x else np.zeros((0,)))
            return cat(pa), cat(feats), cat(rows_f), cat(rows_y), cat(rows_g)

        allidx = np.arange(n)
        score, feats, rf, ry, rg = run(allidx)
        if not np.isfinite(score).all():
            alert(f"{name}: non-finite outputs")
            return 2
        r: dict = {}
        w = st["window"] == 1
        # (a) behaviour at empty-chain windows by group, and input-flip sensitivity
        grp = {}
        for mt in (1, 0):
            for t1 in (1, 0):
                for ph, phn in ((PH_DRAW, "draw"), (PH_MAIN1, "main1"), (0x200, "end"), (-1, "other")):
                    m = w & (st["my_turn"] == mt) & ((st["turn"] == 1) == bool(t1))
                    m &= (st["phase"] == ph) if ph > 0 else ~np.isin(st["phase"], (PH_DRAW, PH_MAIN1, 0x200))
                    if m.sum() >= 20:
                        grp[f"{'own' if mt else 'opp'}_turn{'1' if t1 else '2+'}_{phn}"] = {
                            "n": int(m.sum()), "mean": round(float(score[m].mean()), 4)}
        r["window_groups"] = grp

        def flip_turnplayer(g):
            g[:, 5] = 1 - g[:, 5]
            return g

        def flip_turn13(g):
            g[:, 2] = np.where(g[:, 2] == 1, 3, np.where(g[:, 2] >= 2, 1, g[:, 2]))
            return g

        def flip_phase(g):
            g[:, 3] = np.where(g[:, 3] == PH_DRAW, PH_MAIN1, np.where(g[:, 3] == PH_MAIN1, PH_DRAW, g[:, 3]))
            return g

        wi = np.flatnonzero(w)
        flips = {}
        for fname, fn in (("turn_player", flip_turnplayer), ("turn_1_vs_3", flip_turn13), ("draw_vs_main1", flip_phase)):
            s2 = run(wi, fn)[0]
            d = s2 - score[wi]
            own = st["my_turn"][wi] == 1
            flips[fname] = {"mean_abs_delta": round(float(np.abs(d).mean()), 4),
                            "mean_delta_own": round(float(d[own].mean()), 4) if own.any() else None,
                            "mean_delta_opp": round(float(d[~own].mean()), 4) if (~own).any() else None}
        r["flip"] = flips
        # (b) probes
        groups = st["game"]
        y = {"my_turn": st["my_turn"], "went_first": st["went_first"], "turn1": (st["turn"] == 1).astype(int),
             "ns_used": st["ns_used"], "any_act": (st["n_act"] > 0).astype(int)}
        rawg = st["obs_global"][:, :22].astype(np.float32)
        r["probe"] = {k: {"repr": logreg(feats.astype(np.float32), v, groups),
                          "raw_global": logreg(rawg, v, groups)} for k, v in y.items()}
        ph = st["phase"]
        cls = {v: i for i, v in enumerate(sorted(set(ph.tolist())))}
        yp = np.array([cls[v] for v in ph])
        r["probe"]["phase"] = {"repr": logreg(feats.astype(np.float32), yp, groups, multiclass=True),
                               "raw_global": logreg(rawg, yp, groups, multiclass=True)}
        if len(ry):
            r["probe"]["row_activated_this_turn"] = {"repr": logreg(rf.astype(np.float32), ry.astype(int), rg)}
        res[name] = r
        print(f"== {name}\n" + json.dumps(r, indent=1), flush=True)
    (out / "probe.json").write_text(json.dumps(res, indent=1))
    pol = res.get("policy", {})
    if pol and pol["flip"]["turn_player"]["mean_abs_delta"] == 0.0:
        alert("turn-player flip changed nothing at all: check the override path")
        return 2
    return 0


# ---------------------------------------------------------------- text
def text(a) -> int:
    from sklearn.feature_extraction.text import TfidfVectorizer

    z = np.load(a.emb)
    L = np.load(a.labels)
    if not np.array_equal(z["codes"], L["codes"]):
        alert("label codes do not match the embedding table")
        return 2
    lim = a.text_limit or len(z["codes"])
    z = {k: z[k][:lim] if z[k].ndim else z[k] for k in z}
    L = {k: L[k][:lim] if k in ("codes", "card_lab", "eff_lab") else L[k] for k in L}
    card = z["card"].astype(np.float32)
    effs = z["effects"].astype(np.float32)
    n_eff = z["n_effects"]
    eff_mean = effs.sum(1) / np.maximum(n_eff, 1)[:, None]
    texts = json.loads(str(L["eff_text"]))[:lim]
    whole = np.array([" ".join(t) for t in texts])
    # group = identical card text (alternate artworks share it) so the split never leaks a text into both sides
    _, grp = np.unique(whole, return_inverse=True)
    res: dict = {"card": {}, "effect": {}}
    lab = L["card_lab"]
    bow = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=30000).fit_transform(whole)
    for j, k in enumerate(L["card_keys"]):
        m = lab[:, j] >= 0
        y = lab[m, j].astype(int)
        res["card"][str(k)] = {"card_vec": logreg(card[m], y, grp[m]),
                               "eff_mean (model input)": logreg(np.concatenate([card, eff_mean], 1)[m], y, grp[m]),
                               "bag_of_words": logreg_sparse(bow[m], y, grp[m])}
    el = L["eff_lab"]
    ii, jj = np.nonzero(el[:, :, 0] >= 0)
    sent = np.array([texts[i][j] for i, j in zip(ii, jj)])
    _, sgrp = np.unique(sent, return_inverse=True)
    Xe = effs[ii, jj]
    bow_e = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=30000, token_pattern=r"[^\s]+").fit_transform(sent)
    for q, k in enumerate(L["eff_keys"]):
        y = el[ii, jj, q].astype(int)
        res["effect"][str(k)] = {"effect_vec": logreg(Xe, y, sgrp), "bag_of_words": logreg_sparse(bow_e, y, sgrp)}
    (Path(a.out) / "text.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    return 0


def logreg_sparse(X, y, groups) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    if len(np.unique(y)) < 2:
        return {"skip": "one class"}
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, y, groups))
    m = LogisticRegression(max_iter=3000, C=4.0).fit(X[tr], y[tr])
    if len(np.unique(y[te])) < 2:
        return {"skip": "one class in test"}
    return {"auc": round(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("collect", "probe", "text"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", nargs="*", default=[])
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--keep", type=float, default=0.08, help="share of other decisions kept")
    ap.add_argument("--keep-window", type=float, default=0.3, help="share of turn-3+ windows kept (turn 1-2: all)")
    ap.add_argument("--d", type=int, default=256, help="model width (tests use a tiny random net)")
    ap.add_argument("--text-limit", type=int, default=0, help="text stage on the first N cards only (smoke)")
    ap.add_argument("--policy", default="")
    ap.add_argument("--value", default="")
    ap.add_argument("--emb", default=str(ROOT / "data/cards/emb/qwen3-0.6b-d512.npz"))
    ap.add_argument("--labels", default=str(ROOT / "data/cache/diag/text_labels.npz"))
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    return {"collect": collect, "probe": probe, "text": text}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
