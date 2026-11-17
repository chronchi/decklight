"""Report directory writer: JSON tables, `report.md`, and an optional DeepSeek interpretation brief."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "opencode-go/deepseek-v4.1-flash"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def _table(rows: list[dict], cols: list[str], limit: int = 40) -> str:
    if not rows:
        return "_(none)_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "".join(
        "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows[:limit]
    )
    return head + body


def _behaviour_md(tag: str, t: dict) -> str:
    summary = (
        f"windows {t.get('windows')} over {t.get('starts')} starts; p(activate) mean "
        f"{t.get('p_act_mean')} median {t.get('p_act_median')}; argmax is an activation in "
        f"{t.get('argmax_activation_share')} of windows."
    )
    lines = [f"### Behaviour {tag}", "", summary, ""]
    lines += [_table(t.get("by_chain_turn_phase", []), ["chain", "turn", "phase", "n", "p_act_mean",
                                                         "p_act_median", "pass_share", "activation_share"])]
    lines += ["", "By activation card (top 10):", ""]
    cards = []
    for r in t.get("by_card", [])[:10]:
        cards.append({**r, "codes": ",".join(str(c) for c in r.get("codes", []))})
    lines += [_table(cards, ["name", "codes", "windows", "n", "p_act_mean", "p_act_median"])]
    return "\n".join(lines)


def _probes_md(tag: str, t: dict) -> str:
    rows = []
    for entries in t.values():
        for key, e in entries.items():
            if e.get("skipped"):
                rows.append({"target": key, "n": e.get("n"), "note": "skipped"})
                continue
            for fs in ("base", "glob", "opt", "opt+base", "glob+base"):
                v = e.get(fs)
                if v is None:
                    continue
                rows.append({"target": key, "n": e.get("n"), "pos_rate": e.get("pos_rate"), "features": fs,
                             "score": v.get("score"), "ci": v.get("ci"), "perm_p": v.get("perm_p")})  # fmt: skip
    return f"### Probes {tag}\n\n" + _table(
        rows, ["target", "n", "pos_rate", "features", "score", "ci", "perm_p"], limit=200
    )


def _attention_md(tag: str, t: dict) -> str:
    if not t.get("available"):
        return f"### Attention {tag}\n\n_not available for this architecture_\n"
    rows = []
    for chain, groups in t.get("by_chain", {}).items():
        for g, v in groups.items():
            rows.append({
                "chain": chain, "group": g, "n": t.get("n_by_chain", {}).get(chain), "mean_mass": v,
                "spearman_p_act": t.get("spearman_by_chain", {}).get(chain, {}).get(g),
            })  # fmt: skip
    return f"### Attention {tag}\n\n" + _table(
        rows, ["chain", "group", "n", "mean_mass", "spearman_p_act"], limit=60
    )


def _interventions_md(tag: str, t: dict | list) -> str:
    if isinstance(t, list):
        rows = [dict(r) for r in t]
    else:
        rows = [{"intervention": k, "chain": "all", **v} for k, v in t.items()]
    return f"### Interventions {tag}\n\n" + _table(
        rows, ["intervention", "chain", "n", "delta_p_act_mean", "delta_p_act_median", "argmax_flip_share",
               "flips_to_activation", "noop"], limit=40
    )  # fmt: skip


def build_markdown(meta: dict, sections: dict) -> str:
    lines = ["# Policy diagnosis", ""]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    if "behaviour" in sections:
        lines += ["## Behaviour", ""]
        b = sections["behaviour"]
        if "A" in b:
            lines += [_behaviour_md("A", b["A"]), "", _behaviour_md("B", b["B"]), ""]
            lines += ["### Argmax disagreements (first 40)", ""]
            lines += [_table(b.get("disagreements", []),
                             ["group", "k", "chain", "phase", "turn", "a_label", "b_label", "a_p", "b_p"])]
            lines.append(f"\nDisagreement rate: {b.get('disagreement_rate')}\n")
        else:
            lines += [_behaviour_md("A", b), ""]
    if "probes" in sections:
        lines += ["## Probes", ""]
        p = sections["probes"]
        if "A" in p:
            lines += [_probes_md("A", p["A"]["tables"]), "", _probes_md("B", p["B"]["tables"]), ""]
        else:
            lines += [_probes_md("A", p["tables"]), ""]
        for tag, res in (p.items() if "A" in p else [("A", p)]):
            if res.get("errors"):
                lines += [f"- {tag} errors: {res['errors']}", ""]
    if "attention" in sections:
        lines += ["## Attention", ""]
        a = sections["attention"]
        if "A" in a:
            lines += [_attention_md("A", a["A"]), "", _attention_md("B", a["B"]), ""]
        else:
            lines += [_attention_md("A", a), ""]
    if "interventions" in sections:
        lines += ["## Interventions", "",
                  "_Edited inputs can be off-distribution: read these as sensitivity, not as plays._", ""]
        it = sections["interventions"]
        if "A" in it:
            lines += [_interventions_md("A", it["A"]), "", _interventions_md("B", it["B"]), ""]
        else:
            lines += [_interventions_md("A", it), ""]
    return "\n".join(lines) + "\n"


def write_report(out: Path, meta: dict, sections: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in sections.items():
        write_json(out / f"{name}.json", obj)
    md = out / "report.md"
    md.write_text(build_markdown(meta, sections))
    return md


def interpret_brief(out: Path, meta: dict, report_md: Path) -> Path:
    """A DeepSeek brief from the report (findings only), generalising docs/briefs/interpret-opp-probe.md."""
    text = f"""# Interpretation brief: read this policy diagnosis (findings only, do NOT change any file, do NOT commit)

Context: a Yu-Gi-Oh! policy network was diagnosed offline (inference only) and the tables are in the attached report.

Run metadata:
{json.dumps(meta, indent=1)}

The report `{report_md}` (and the JSON tables next to it) contains:
- behaviour: p(activate), pass share by chain length x turn owner x phase, and by activation card;
- probes: grouped nested CV (AUC/R²) of base public counts, the global token, the option-head input at the top
  activation, and their combinations, for `complete` (opponent finishes its turn) and `field` (opponent cards at the
  end of the turn), on all windows and on the counterfactual pass subset; every metric has a start-level permutation
  p-value and a bootstrap CI over starts;
- attention (when the architecture exposes it): cross-attention mass by token group for the top activation, by chain
  length and target, and Spearman with p(activate);
- interventions: inference-only input edits (mask own same-turn history, set chain length to 1/0, remove the option's
  own card context) with the change in p(activate) and argmax flips. Edited inputs are off-distribution.

Questions (answer each with evidence from the report; say "unknown" where the data cannot tell):
1. What does the option head appear to base the activation decision on? Which token groups / inputs matter?
2. Is the target information present in the representation but unused, absent, or too weak to tell? Use the probe
   margins, the permutation p-values/CIs and the counterfactual pass subset.
3. Which interventions move the decision most, and what does that imply about the mechanism?
4. Rank the levers most likely to work: (a) reward shaping / exploration, (b) architecture, (c) data.
Output a `## RESULT` block with numbered answers, each with the evidence (table + numbers) and a proposed cheap test.
"""
    brief = out / "interpret-brief.md"
    brief.write_text(text)
    return brief


def run_interpretation(out: Path, meta: dict, report_md: Path, *, model: str = DEFAULT_MODEL) -> str:
    """Write the brief and launch `scripts/delegate.sh` (background); returns the delegate tag."""
    brief = interpret_brief(out, meta, report_md)
    tag = f"diag-interpret-{out.name}"
    subprocess.run(
        [str(ROOT / "scripts" / "delegate.sh"), str(brief), model, tag],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tag
