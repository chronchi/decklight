"""Per-deck fine-tune (user-facing): continue P's PPO with the target deck in `--share` of seats.

    uv run python scripts/finetune_deck.py --deck my.ydk --out runs/ft-my [--hours 2] [--iters 0] [--share 0.5]
        [--device cuda] [--lr 4e-5] [--kl-coef 0.1] [extra train_rl args after --]

Runs scripts/train_rl.py with --init data/cache/P/best.pt, --value-init data/cache/value_p2/best.pt,
--focus-deck/--focus-share (train_rl.py), --eval-every huge (no periodic general evals) and --no-ygoagent
unless --ygoagent is passed. Writes, fp16 weights-only, in the product's layout (P + value_p2):
<out>/policy/best.pt (policy, no value head) and <out>/value/best.pt (trunk + trained value head), each with its
base's metrics.jsonl (the loaders read the architecture from it), plus <out>/FINETUNE.json; prints the play commands.

Invariants (exit non-zero with a clear message): the deck file does not parse; train_rl failed; last.pt is
missing; the policy weights equal P's exactly (nothing trained); the value head equals value_p2's exactly; either
file fails to load through the product loaders (BCPolicy / ValueNet).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decklight.decks import load_ydk

BASE_CKPT = Path("data/cache/P/best.pt")
BASE_VALUE = Path("data/cache/value_p2/best.pt")


def fail(msg: str) -> NoReturn:
    print(f"ALERT finetune_deck: {msg}", flush=True)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state.get("model", state)


def check_invariants(final: Path, base_ckpt: Path, base_value: Path) -> None:
    """Fail loud when the fine-tune trained nothing: policy weights equal P's, or value head equal value_p2's."""
    f = load_state(final)
    b = load_state(base_ckpt)
    v = load_state(base_value)
    f_pol = {k: v for k, v in f.items() if not k.startswith("value_head.") and not k.startswith("q_head.")}
    b_pol = {k: v for k, v in b.items() if not k.startswith("value_head.") and not k.startswith("q_head.")}
    if set(f_pol) == set(b_pol) and all(torch.equal(f_pol[k], b_pol[k]) for k in f_pol):
        fail(f"{final} policy weights equal {base_ckpt} exactly (nothing trained)")
    f_val = {k: v for k, v in f.items() if k.startswith("value_head.")}
    v_val = {k: v for k, v in v.items() if k.startswith("value_head.")}
    if f_val and set(f_val) == set(v_val) and all(torch.equal(f_val[k], v_val[k]) for k in f_val):
        fail(f"{final} value head equals {base_value} exactly (value head never trained)")


def check_loads(policy_pt: Path, value_pt: Path) -> None:
    """The outputs must load through the product's own loaders (strict state-dict loads)."""
    from decklight.eval.bc_policy import BCPolicy
    from decklight.eval.value import ValueNet
    from decklight.search.product import PRODUCT_EMB

    try:
        BCPolicy(policy_pt, PRODUCT_EMB, threads=1, device="cpu", batch=8)
        ValueNet(value_pt, PRODUCT_EMB, device="cpu", batch=8)
    except Exception as e:  # noqa: BLE001 - any load failure is fatal here
        fail(f"{policy_pt} / {value_pt} do not load as the product's policy/value: {e!r}")


def export(full_state: dict, dest_dir: Path, init: Path, value_init: Path) -> tuple[Path, Path]:
    """Export one fp16 full state into the product layout under dest_dir.

    Writes dest_dir/policy/best.pt (policy, no value/q heads) and dest_dir/value/best.pt
    (trunk + trained value head), each with its base's metrics.jsonl, then runs
    check_invariants + check_loads. Returns (policy_pt, value_pt).
    """
    full = {k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v
            for k, v in full_state.items()}
    policy_pt, value_pt = dest_dir / "policy" / "best.pt", dest_dir / "value" / "best.pt"
    for pt, base in ((policy_pt, init), (value_pt, value_init)):
        m = base.parent / "metrics.jsonl"
        if not m.is_file():
            fail(f"{m} missing: {pt} would load with the wrong architecture")
        pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m, pt.parent / "metrics.jsonl")
    torch.save({k: v for k, v in full.items() if not k.startswith(("value_head.", "q_head."))}, policy_pt)
    torch.save(full, value_pt)
    check_invariants(value_pt, init, value_init)
    check_loads(policy_pt, value_pt)
    return policy_pt, value_pt


def write_bundle(out: Path, *, deck: Path, policy_pt: Path, value_pt: Path,
                 emb: str | Path | None = None, cards: list[str] | None = None,
                 name: str | None = None) -> Path:
    """Write `<out>/bundle/bot.toml` (+ a copy of the .ydk) for the fine-tune.

    `emb` defaults to the npz the product loaders actually use; `cards` defaults
    to today's `YGO_EXTRA_CARDS` fallback. Returns the bundle dir.
    """
    import shutil as _shutil

    from decklight.engine.resources import extra_card_dirs
    from decklight.search.product import resolve_product_paths

    def _rel(p: Path, base: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(base.resolve()))
        except ValueError:
            return str(p)

    bundle_dir = out / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _shutil.copyfile(deck, bundle_dir / "deck.ydk")
    if emb is None:
        _, _, emb = resolve_product_paths(None, None, None)
    if cards is None:
        cards = [str(d) for d in extra_card_dirs()]
    lines = [
        f"name = {json.dumps(name or deck.stem)}",
        f"description = {json.dumps(f'Fine-tuned on {deck.stem}')}",
        'agent = "product"',
        'deck = "deck.ydk"',
        f"ckpt = {json.dumps(_rel(policy_pt, bundle_dir))}",
        f"value = {json.dumps(_rel(value_pt, bundle_dir))}",
        f"emb = {json.dumps(str(emb))}",
    ]
    if cards:
        lines.append("cards = [")
        lines += [f"  {json.dumps(c)}," for c in cards]
        lines.append("]")
    (bundle_dir / "bot.toml").write_text("\n".join(lines) + "\n")
    return bundle_dir


def last_iter_metrics(rl_out: Path) -> dict:
    """The last training record of the train_rl metrics.jsonl (iterations, hours, focus fields).

    Eval records also carry "iter" ({"iter": N, "eval": ...}, written after the last iteration), so a training record
    is one with "samples"."""
    last: dict | None = None
    try:
        for raw in (rl_out / "metrics.jsonl").read_text().splitlines():
            if not raw.strip():
                continue
            rec = json.loads(raw)
            if "iter" in rec and "samples" in rec:
                last = rec
    except (OSError, ValueError) as e:
        fail(f"cannot read {rl_out}/metrics.jsonl: {e}")
    if last is None:
        fail(f"no iteration records in {rl_out}/metrics.jsonl")
    assert last is not None
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", type=Path, nargs="+", required=True,
                      help="target .ydk file(s), anywhere (several = multi-list fine-tune, "
                      "round-robin over --focus-deck)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", type=float, default=2.0,
                    help="training budget; 2 h beat 0.75 h on Blue-Eyes (docs/results.md)")
    ap.add_argument("--iters", type=int, default=0, help="stop after this many iterations (0 = by --hours)")
    ap.add_argument("--share", type=float, default=0.5, help="share of seats holding the deck")
    ap.add_argument("--device", default=None, help="cuda (default when available) | cpu (works, very slow)")
    ap.add_argument("--lr", type=float, default=4e-5, help="Stage P's value")
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--init", type=Path, default=BASE_CKPT)
    ap.add_argument("--value-init", type=Path, default=BASE_VALUE)
    ap.add_argument("--eval-every", type=int, default=10**9,
                    help="periodic general evals never run (the final eval still does)")
    ap.add_argument("--ygoagent", action="store_true",
                    help="keep the slow ygo-agent eval (default: --no-ygoagent)")
    ap.add_argument("--rl-out", type=Path, default=None, help="train_rl output dir (default <out>/rl)")
    ap.add_argument("--snap-every", type=int, default=0,
                    help="export a product-layout snapshot every N iterations (0 = off)")
    ap.add_argument("--extra-cards", default=None,
                    help="custom card dirs (os.pathsep-separated) for the bundle; "
                         "default: today's YGO_EXTRA_CARDS fallback")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("warning: --device cpu will be very slow (fine-tunes run on a vast GPU box)", flush=True)
    try:
        decks = [load_ydk(p) for p in args.deck]
    except OSError as e:
        fail(f"--deck: cannot read: {e}")
    for p, deck in zip(args.deck, decks, strict=True):
        if not deck.main:
            fail(f"--deck {p}: no main-deck cards (unparseable deck)")
    for p, label in ((args.init, "--init"), (args.value_init, "--value-init")):
        if not p.is_file():
            fail(f"{label} {p} missing")
    out, rl_out = args.out, args.rl_out or args.out / "rl"
    out.mkdir(parents=True, exist_ok=True)
    rl_out.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(ROOT / "scripts" / "train_rl.py"),
           "--init", str(args.init), "--value-init", str(args.value_init),
           "--focus-deck", *[str(p) for p in args.deck], "--focus-share", str(args.share),
           "--lr", str(args.lr), "--kl-coef", str(args.kl_coef),
           "--eval-every", str(args.eval_every), "--device", device,
           "--hours", str(args.hours), "--iters", str(args.iters),
           "--out", str(rl_out)]  # fmt: skip
    if not args.ygoagent:
        cmd.append("--no-ygoagent")
    if args.snap_every > 0:
        cmd += ["--save-every", str(args.snap_every)]
    cmd += extra
    print(f"[finetune] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        fail(f"train_rl exited {r.returncode}")

    last_pt = rl_out / "last.pt"
    if not last_pt.is_file():
        fail(f"{last_pt} missing (train_rl wrote no checkpoint)")
    # Two files like the product (P + value_p2): BCPolicy loads strictly without the value head, ValueNet with it.
    # Each dir gets its base's metrics.jsonl: the loaders take the architecture from <ckpt dir>/metrics.jsonl
    # (bc_policy._run_args; a missing file silently means default args, value_p2's adds `value: true`).
    policy_pt, value_pt = export(load_state(last_pt), out, args.init, args.value_init)

    snapshots: list[dict] = []
    for snap_pt in sorted(rl_out.glob("snap-*.pt")):
        try:
            it = int(snap_pt.stem.split("-")[1])
        except (IndexError, ValueError):
            fail(f"unparseable snapshot name {snap_pt}")
        dest = out / "snaps" / f"it-{it:04d}"
        p_pt, v_pt = export(load_state(snap_pt), dest, args.init, args.value_init)
        snapshots.append({"iter": it, "policy": str(p_pt), "value": str(v_pt)})
        snap_pt.unlink()

    last = last_iter_metrics(rl_out)
    if args.snap_every > 0 and not snapshots and last["iter"] >= args.snap_every:
        fail(f"--snap-every {args.snap_every}: no snapshot written although training ran {last['iter']} iterations")
    info = {
        "deck": str(args.deck[0]), "deck_sha256": sha256_file(args.deck[0]),
        "decks": [str(p) for p in args.deck],
        "deck_sha256s": [sha256_file(p) for p in args.deck],
        "base_ckpt": str(args.init), "base_sha256": sha256_file(args.init),
        "base_value": str(args.value_init), "base_value_sha256": sha256_file(args.value_init),
        "iterations": last["iter"], "hours": last.get("hours"),
        "focus_wr_vs_bc": last.get("focus_wr_vs_bc"), "focus_seat_share": last.get("focus_seat_share"),
        "snapshots": snapshots,
        "train_rl_cmd": cmd,
    }
    if info["focus_wr_vs_bc"] is None or info["focus_seat_share"] is None:
        fail(f"{rl_out}/metrics.jsonl iteration {last['iter']} has no focus fields")
    (out / "FINETUNE.json").write_text(json.dumps(info, indent=1))
    import os as _os

    from decklight.play.bundle import Bundle as _Bundle

    extra = args.extra_cards.split(_os.pathsep) if args.extra_cards else None
    bundle_dir = write_bundle(out, deck=args.deck[0], policy_pt=policy_pt, value_pt=value_pt,
                              cards=extra)
    try:
        _Bundle.load(bundle_dir)
    except Exception as e:  # noqa: BLE001 - the self-check must fail loud
        fail(f"fresh bundle {bundle_dir} does not validate: {e}")
    print(f"bot bundle: {bundle_dir} (copy it into bots/ to play it anywhere)", flush=True)
    print(f"[finetune] wrote {policy_pt}, {value_pt} + {out}/FINETUNE.json "
          f"(decks {info['decks']}, it {info['iterations']}, "
          f"focus_wr_vs_bc {info['focus_wr_vs_bc']}, "
          f"focus_seat_share {info['focus_seat_share']})", flush=True)
    print(f"play it: uv run python scripts/web_play.py --ckpt {policy_pt} --value {value_pt}", flush=True)
    print(f"or:      uv run python scripts/edopro_host.py --deck {args.deck[0]} --opponent product "
          f"--ckpt {policy_pt} --value {value_pt}", flush=True)


if __name__ == "__main__":
    main()
