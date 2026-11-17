#!/usr/bin/env bash
# Android launcher icon and splash mark from the approved brand art (docs/ui/brand/decklight/final, user-approved logo).
# Adaptive icon (API 26+): background = the night colour #141310, foreground = the mark inside the 66 dp safe circle,
# monochrome = the foreground's light strokes as alpha (Android 13 themed icons). Splash (API 31+): the mark alone.
# Needs inkscape and ImageMagick. Writes into the generated Android project's res/.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
mark=$here/docs/ui/brand/decklight/final/decklight-mark.svg
icon=$here/docs/ui/brand/decklight/final/decklight-icon.svg
res=$here/rust/app/src-tauri/gen/android/app/src/main/res
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
# The mark's drawn extent is x 70..440, y 145..400 (the crop the lobby header uses). frame W: the visible width of
# that extent as a share of the canvas; returns an SVG whose viewBox centres the mark at that share.
# $3 scales the strokes and the two small dots: at launcher size (60 dp for 370 units) the art's 4-unit strokes would be
# 0.65 dp, too thin to read, so small renders get heavier lines (optical size, as for any small-size icon).
frame() {  # $1 out.svg  $2 share of the canvas the mark's width takes  $3 stroke scale
  python3 - "$mark" "$1" "$2" "$3" <<'PY'
import re, sys
src, out, share, k = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
w = 370 / share
s = open(src).read().replace('viewBox="0 0 512 512"', f'viewBox="{255 - w / 2:.1f} {272.5 - w / 2:.1f} {w:.1f} {w:.1f}"')
s = re.sub(r'stroke-width="([\d.]+)"', lambda m: f'stroke-width="{float(m[1]) * k:.2f}"', s)
s = re.sub(r'r="7\.5"', f'r="{7.5 * (1 + (k - 1) / 2):.2f}"', s)
open(out, "w").write(s)
PY
}
frame "$tmp/fg.svg" 0.56 2.2      # 108 dp canvas: 60 dp wide, inside the 66 dp safe circle
frame "$tmp/splash.svg" 0.50 1.5  # 288 dp splash canvas: 144 dp wide, inside its 192 dp circle
declare -A dens=([mdpi]=1 [hdpi]=1.5 [xhdpi]=2 [xxhdpi]=3 [xxxhdpi]=4)
for d in "${!dens[@]}"; do
  s=${dens[$d]}; out=$res/mipmap-$d; mkdir -p "$out"
  fg=$(python3 -c "print(round(108*$s))"); lg=$(python3 -c "print(round(48*$s))")
  inkscape "$tmp/fg.svg" -w "$fg" -h "$fg" -o "$out/ic_launcher_foreground.png" 2>/dev/null
  # Monochrome: white, alpha = brightness (dark card faces stay clear, the light strokes carry the shape).
  convert "$out/ic_launcher_foreground.png" -background black -alpha remove -colorspace gray -level 8%,85% \
    "$tmp/a.png"
  convert -size "${fg}x${fg}" xc:white "$tmp/a.png" -alpha off -compose copy-opacity -composite \
    "$out/ic_launcher_monochrome.png"
  inkscape "$icon" -w "$lg" -h "$lg" -o "$out/ic_launcher.png" 2>/dev/null
  sed 's|rx="112"|rx="256"|' "$icon" > "$tmp/round.svg"
  inkscape "$tmp/round.svg" -w "$lg" -h "$lg" -o "$out/ic_launcher_round.png" 2>/dev/null
done
mkdir -p "$res/drawable-nodpi" "$res/mipmap-anydpi-v26"
inkscape "$tmp/splash.svg" -w 1152 -h 1152 -o "$res/drawable-nodpi/splash_mark.png" 2>/dev/null
# The light theme's splash (Settings → Appearance; MainActivity picks the splash theme for the next launch): the same
# mark recoloured as in web/styles.css :root[data-theme="light"] --logo-*.
sed -e 's/#141310/#faf7f0/g' -e 's/#7a756a/#a9b8cd/g' -e 's/#958f82/#7d8ea8/g' -e 's/#d6cfc0/#46628c/g' -e 's/#b89a5e/#8a6a2c/g' \
  "$tmp/splash.svg" > "$tmp/splash-light.svg"
inkscape "$tmp/splash-light.svg" -w 1152 -h 1152 -o "$res/drawable-nodpi/splash_mark_light.png" 2>/dev/null
for n in ic_launcher ic_launcher_round; do
  cat > "$res/mipmap-anydpi-v26/$n.xml" <<'XML'
<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/night" />
    <foreground android:drawable="@mipmap/ic_launcher_foreground" />
    <monochrome android:drawable="@mipmap/ic_launcher_monochrome" />
</adaptive-icon>
XML
done
rm -f "$res/drawable/ic_launcher_background.xml" "$res/drawable-v24/ic_launcher_foreground.xml"
echo "icons written to $res"
