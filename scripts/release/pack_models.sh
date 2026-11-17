#!/usr/bin/env bash
# Pack the release model bundle: the 9 untracked product files (+ the product policy r1 from v0.2.0) into one
# deterministic tarball, plus checksums.
#
# Usage: bash scripts/release/pack_models.sh [TAG]   (default TAG v0.3.0)
#   Writes data/cache/release/decklight-models-$TAG.tar.gz,
#   data/cache/release/SHA256SUMS and the tracked release/MODELS-$TAG.sha256.
#   Exits 3 naming the first missing file. Works from any cwd.
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${1:-v0.3.0}"
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
# v0.2+: the Forge release r1 (the product policy since v0.2.0; contributors fine-tune from it).
[ "$TAG" = v0.1 ] || FILES+=(data/cache/forge_r1/best.pt data/cache/forge_r1/metrics.jsonl)
# v0.3+: the product policy Y (scripts/product.sh) and what the app embeds that git does not hold (build_app_assets.py):
# Y's ONNX export and the studied deck primers (their run log stays out).
case "$TAG" in v0.1|v0.2*) ;; *)
  FILES+=(data/cache/y_league/best.pt data/cache/y_league/metrics.jsonl)
  while IFS= read -r f; do FILES+=("$f"); done < <(find data/cache/onnx/y data/primers/studied -type f ! -name '*.log' | sort)
  [ -f data/cache/onnx/y/policy.onnx ] && [ -f data/primers/studied/primers.json ] ||
    { echo "pack_models: no app data (data/cache/onnx/y, data/primers/studied): export Y to ONNX first" >&2; exit 3; } ;;
esac
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
