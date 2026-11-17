"""Decklight logo: three fanned cards, one win-chance point per card, the last card lit (user, 2026-10-19).

Writes <name>.svg (1200x700 lockup, wordmark only) and <name>-icon.svg (512 app icon) for the gold-outlined and
the washi-outlined last card. Render with inkscape (see the README line in NOTEBOOK 2026-10-19).

    python3 make_logo.py
"""

import math

BG, WASHI, GOLD, MUTED, FAINT, MID = "#141310", "#d6cfc0", "#b89a5e", "#958f82", "#5e5a51", "#7a756a"
ANGLES = (-20, 0, 20)  # fan, degrees (four cards were too wide, user 2026-10-19)
VALUES = (0.22, 0.50, 0.98)  # win chance per card: rises, the last card is the best
XOFF = (-0.22, -0.22, 0.0)  # points sit in each card's visible left strip; the last card is fully visible

GLOW = f"""<defs><radialGradient id="glow"><stop offset="0" stop-color="{GOLD}" stop-opacity="0.5"/><stop offset="1" stop-color="{GOLD}" stop-opacity="0"/></radialGradient>
<radialGradient id="cardglow"><stop offset="0" stop-color="{GOLD}" stop-opacity="0.10"/><stop offset="1" stop-color="{GOLD}" stop-opacity="0"/></radialGradient></defs>"""


def rot(px, py, ox, oy, deg):
    a = math.radians(deg)
    dx, dy = px - ox, py - oy
    return ox + dx * math.cos(a) - dy * math.sin(a), oy + dx * math.sin(a) + dy * math.cos(a)


def fan3(cx, cy, s, gold_card):
    w, h = s * 0.52, s * 0.76
    ox, oy = cx, cy + s * 0.95
    cols = (MID, MUTED, GOLD if gold_card else WASHI)
    lx, ly = rot(cx, cy, ox, oy, ANGLES[-1])
    out = f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="{s * 0.7:.1f}" fill="url(#cardglow)"/>'
    pts = []
    for ang, col, v, xo in zip(ANGLES, cols, VALUES, XOFF):
        sw = s * (0.02 if col in (GOLD, WASHI) else 0.016)
        out += (
            f'<g transform="rotate({ang} {ox:.1f} {oy:.1f})"><rect x="{cx - w / 2:.1f}" y="{cy - h / 2:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}" rx="{s * 0.05:.1f}" fill="{BG}" stroke="{col}" stroke-width="{sw:.1f}"/></g>'
        )
        pts.append(rot(cx + xo * w, cy + h * 0.3 - v * h * 0.62, ox, oy, ang))
    out += (
        '<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="none" stroke="{GOLD}" '
        f'stroke-width="{s * 0.022:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for a, b in pts[:-1]:
        out += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{s * 0.03:.1f}" fill="{BG}" stroke="{GOLD}" stroke-width="{s * 0.015:.1f}"/>'
    a, b = pts[-1]
    out += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{s * 0.26:.1f}" fill="url(#glow)"/>'
    out += f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{s * 0.055:.1f}" fill="{GOLD}"/>'
    return out


def lockup(name, mark):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">
<title>decklight — {name}</title>{GLOW}
<rect width="1200" height="700" fill="{BG}"/>
{mark}
<text x="600" y="590" text-anchor="middle" font-family="Clear Sans Thin, Clear Sans Light, sans-serif" font-weight="200" font-size="100" letter-spacing="14" fill="{WASHI}">decklight</text>
</svg>"""


def icon(name, mark):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
<title>decklight icon — {name}</title>{GLOW}
<rect width="512" height="512" rx="112" fill="{BG}"/>
{mark}
</svg>"""


# "light" heavier than "deck", so the word is lit like the last card (user, 2026-10-22). Variants:
# (label, weight family of "light", fill of "light", warm glow behind "light")
LIGHT_VARIANTS = (
    ("6-light-a", "Clear Sans Light", WASHI, False),
    ("6-light-b", "Clear Sans", WASHI, False),
    ("6-light-c", "Clear Sans Light", "#e6d8b6", True),
)


def lockup_light(name, mark, family, fill, glow, glow_cx=None):
    halo = ""
    if glow and glow_cx is not None:
        halo = (f'<ellipse cx="{glow_cx:.1f}" cy="555" rx="230" ry="95" fill="url(#cardglow)"/>'
                f'<ellipse cx="{glow_cx:.1f}" cy="555" rx="150" ry="60" fill="url(#cardglow)"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700">
<title>decklight — {name}</title>{GLOW}
<rect width="1200" height="700" fill="{BG}"/>
{mark}{halo}
<text x="600" y="590" text-anchor="middle" font-size="100" letter-spacing="14"><tspan font-family="Clear Sans Thin" fill="{WASHI}">deck</tspan><tspan id="light" font-family="{family}" fill="{fill}">light</tspan></text>
</svg>"""


if __name__ == "__main__":
    for name, gold in (("5-fan3-gold", True), ("5-fan3-washi", False)):
        open(f"{name}.svg", "w").write(lockup(name, fan3(600, 250, 260, gold)))
        open(f"{name}-icon.svg", "w").write(icon(name, fan3(256, 262, 250, gold)))
