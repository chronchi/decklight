#!/usr/bin/env bash
# Fetch the release model bundle after cloning: download the tarball from the
# GitHub release, verify its sha256, and extract it at the repo root.
#
# Usage: bash scripts/fetch_release.sh [TAG] [--file PATH] [--root DIR] [--force]
#   TAG default v0.2.0. --file PATH skips the download and uses a local tarball
#   (offline / test path). --root DIR extracts into DIR instead of the repo
#   root (test path). Refuses to overwrite existing files unless --force.
#   Exits 2 on checksum mismatch, before extracting anything. Works from any cwd.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"

TAG="v0.2.0"
FILE=""
DEST="."
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --file) FILE="${2:?--file needs a value}"; shift 2;;
    --file=*) FILE="${1#--file=}"; shift;;
    --root) DEST="${2:?--root needs a value}"; shift 2;;
    --root=*) DEST="${1#--root=}"; shift;;
    --force) FORCE=1; shift;;
    -*) echo "fetch_release: unknown flag $1" >&2; exit 1;;
    *) TAG="$1"; shift;;
  esac
done
# --file/--root are relative to the caller's cwd; everything else to the repo root.
if [ -n "$FILE" ]; then FILE="$(realpath -m "$FILE")"; fi
if [ "$DEST" != "." ]; then DEST="$(realpath -m "$DEST")"; fi
cd "$HERE"

NAME="decklight-models-$TAG.tar.gz"
SUMFILE="release/MODELS-$TAG.sha256"
if [ ! -f "$SUMFILE" ]; then
  echo "fetch_release: missing $SUMFILE" >&2
  exit 1
fi
WANT="$(awk '{print $1}' "$SUMFILE")"

TARBALL="$FILE"
if [ -z "$TARBALL" ]; then
  TARBALL="$(mktemp)"
  trap 'rm -f "$TARBALL"' EXIT
  curl -fL -o "$TARBALL" "https://github.com/chronchi/decklight/releases/download/$TAG/$NAME"
fi
if [ ! -f "$TARBALL" ]; then
  echo "fetch_release: tarball not found: $TARBALL" >&2
  exit 1
fi

GOT="$(sha256sum "$TARBALL" | awk '{print $1}')"
if [ "$GOT" != "$WANT" ]; then
  echo "fetch_release: checksum mismatch for $NAME (want $WANT, got $GOT)" >&2
  exit 2
fi

mkdir -p "$DEST"
if [ "$FORCE" -ne 1 ]; then
  EXISTING=""
  while IFS= read -r f; do
    if [ -e "$DEST/$f" ]; then EXISTING="$EXISTING$DEST/$f
"; fi
  done < <(tar -tzf "$TARBALL")
  if [ -n "$EXISTING" ]; then
    echo "fetch_release: refusing to overwrite existing files (use --force):" >&2
    printf '%s' "$EXISTING" >&2
    exit 1
  fi
fi
tar -xzf "$TARBALL" -C "$DEST"
echo "installed into $DEST:"
tar -tzf "$TARBALL"
