#!/usr/bin/env bash
# Pack the release model bundle: the 9 untracked product files (+ the Forge base r1 from v0.2) into one
# deterministic tarball, plus checksums.
#
# Usage: bash scripts/release/pack_models.sh [TAG]   (default TAG v0.1)
#   Writes data/cache/release/decklight-models-$TAG.tar.gz,
#   data/cache/release/SHA256SUMS and the tracked release/MODELS-$TAG.sha256.
#   Exits 3 naming the first missing file. Works from any cwd.
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${1:-v0.1}"
FILES=(
  data/cache/P/best.pt
  data/cache/P/metrics.jsonl
  data/cache/value_p2/best.pt
  data/cache/value_p2/metrics.jsonl
  data/cache/aux_p2/aux.pt
  data/cache/aux_p2/metrics.json
  data/cards/emb/qwen3-0.6b-d512.npz
  data/cards/emb/strings-qwen3-0.6b-d512.npz
  data/cards/lua_desc.npz
)
# v0.2+: the Forge base release r1 (contributors fine-tune from it; docs/contributing-packs.md).
[ "$TAG" = v0.1 ] || FILES+=(data/cache/forge_r1/best.pt data/cache/forge_r1/metrics.jsonl)
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "pack_models: missing $f" >&2
    exit 3
  fi
done

OUT="data/cache/release/decklight-models-$TAG.tar.gz"
mkdir -p data/cache/release release
tar --sort=name --owner=0 --group=0 --mtime=@0 -cf - -C . "${FILES[@]}" | gzip -n > "$OUT"
(cd data/cache/release && sha256sum "decklight-models-$TAG.tar.gz" > SHA256SUMS)
SHA="$(sha256sum "$OUT" | awk '{print $1}')"
printf '%s  %s\n' "$SHA" "decklight-models-$TAG.tar.gz" > "release/MODELS-$TAG.sha256"
echo "packed $OUT ($SHA)"
