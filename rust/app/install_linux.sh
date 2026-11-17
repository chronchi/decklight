#!/usr/bin/env bash
# Build the decklight desktop app and install it for this user, no sudo: the app and its ONNX Runtime go to
# ~/.local/share/decklight/ (DECKLIGHT_HOME to change it), a menu entry to ~/.local/share/applications/.
# Needs Rust (rustup), uv, the WebKitGTK build packages (README "Install") and the app data (scripts/fetch_release.sh).
# The build cache (rust/app/target/release, ~3 GB) is deleted afterwards; --keep-build keeps it for faster rebuilds.
#     rust/app/install_linux.sh [--keep-build]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${DECKLIGHT_HOME:-$HOME/.local/share/decklight}"
KEEP=0
[ "${1:-}" = "--keep-build" ] && KEEP=1
export PATH="$HOME/.cargo/bin:$PATH"
die() { echo "ALERT install_linux: $*" >&2; exit 1; }

command -v cargo >/dev/null || die "no Rust toolchain: install it from https://rustup.rs, then run this again"
command -v uv >/dev/null || die "no uv: install it from https://docs.astral.sh/uv/, then run this again"
pkg-config --exists webkit2gtk-4.1 javascriptcoregtk-4.1 libsoup-3.0 2>/dev/null ||
  die "the WebKitGTK build packages are missing: install them as in the README's Install section, then run this again"

cd "$ROOT"
echo "install_linux: 1/4 the app's data (web client, cards, models)"
uv sync --group onnx --group web
uv run python scripts/app/build_app_assets.py
ORT="$(ls .venv/lib/python3*/site-packages/onnxruntime/capi/libonnxruntime.so.* 2>/dev/null | sort -V | tail -1)"
[ -n "$ORT" ] || die "no libonnxruntime.so in .venv: run 'uv sync --group onnx' and check it installs onnxruntime"

echo "install_linux: 2/4 building the app (the first build takes a while)"
scripts/patch_core.sh
jobs=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
(cd rust/app/src-tauri && CARGO_BUILD_JOBS=$jobs cargo build --release)
BIN=rust/app/target/release/decklight-app
[ -x "$BIN" ] || die "the build finished without $BIN"

echo "install_linux: 3/4 installing to $DEST"
mkdir -p "$DEST" "$HOME/.local/share/applications"
install -m 755 "$BIN" "$DEST/decklight-app"
install -m 644 "$ORT" "$DEST/libonnxruntime.so"   # the app loads it from its own folder
install -m 644 rust/app/src-tauri/icons/128x128@2x.png "$DEST/decklight.png"
cat > "$HOME/.local/share/applications/decklight.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=decklight
Comment=Play and study Yu-Gi-Oh! with a learned duelist
Exec=$DEST/decklight-app
Icon=$DEST/decklight.png
Terminal=false
Categories=Game;CardGame;
EOF
command -v update-desktop-database >/dev/null && update-desktop-database "$HOME/.local/share/applications" || true

if [ "$KEEP" = 0 ]; then
  echo "install_linux: 4/4 deleting the build cache (rust/app/target/release)"
  rm -rf rust/app/target/release
else
  echo "install_linux: 4/4 build cache kept (--keep-build)"
fi
echo "install_linux: OK: decklight is in your app menu, or run $DEST/decklight-app"
