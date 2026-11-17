"""WCAG 1.4.3 for both themes: every text token is >= 4.5:1 on every surface (bg, felt, zone, sheet), read from
web/styles.css (:root and :root[data-theme="light"]). Exits 1 on a pair below the bar.
    python3 -I scripts/app/theme_check.py"""
import re, sys
from pathlib import Path

css = Path(__file__).resolve().parents[2].joinpath("web/styles.css").read_text()
def block(sel):
    m = re.search(re.escape(sel) + r"\s*\{(.*?)\n\}", css, re.S)
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", m.group(1)))
dark = block(":root"); light = dict(dark, **block(':root[data-theme="light"]'))

def lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
def ratio(a, b):
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)

TEXT = ("text", "muted", "accent", "danger", "bot-ink", "neg", "you-text", "bot-text", "text-2", "text-3")
SURF = ("bg", "felt", "zone", "sheet")
bad = 0
# Graphics that carry state (WCAG 1.4.11, 3:1): Ask's thinking mark and the scrubber's done dots.
GRAPHIC = ("mark-edge", "mark-gold")
for name, t in (("dark", dark), ("light", light)):
    worst = {f: min((ratio(t[f], t[s]), s) for s in SURF) for f in TEXT}
    print(name, " ".join(f"{f} {r:.2f}" for f, (r, s) in worst.items()))
    for f in GRAPHIC:
        r, s = min((ratio(t[f], t[s]), s) for s in ("bg", "sheet"))
        if r < 3: print(f"  BELOW 3: {name} --{f} {t[f]} on --{s}: {r:.2f}"); bad += 1
    for f, (r, s) in worst.items():
        if r < 4.5:
            print(f"  BELOW 4.5: {name} --{f} {t[f]} on --{s} {t[s]}: {r:.2f}"); bad += 1
sys.exit(1 if bad else 0)
