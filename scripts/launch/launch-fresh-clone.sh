#!/usr/bin/env bash
# Release step 4 (docs/plans/2026-10-21-github-release.md): the public README, followed on a clean box from the
# generated public export only (scripts/release/fresh_clone_test.sh). Regenerates the export, bundles it with the
# model tarball into data/cache/release-test/, ships only that (setup replaced: no private build on the box).
# Small GPU box (>= 16 cores for the parallel unit tests), 1.5 h cap.
# Env: HOURS (1.5), MAX_DPH (0.15), MIN_CPUS (16).
set -euo pipefail
cd "$(dirname "$0")/../.."
TAR=data/cache/release/decklight-models-v0.1.tar.gz
[ -f "$TAR" ] || { echo "missing $TAR (scripts/release/pack_models.sh)"; exit 3; }
EXP=/tmp/decklight-public-test
bash scripts/release/make_public.sh --force HEAD "$EXP" > data/cache/release-export.log 2>&1 \
  || { echo "export failed: data/cache/release-export.log"; exit 3; }
mkdir -p data/cache/release-test
git -C "$EXP" bundle create "$PWD/data/cache/release-test/decklight.bundle" --all
cp "$TAR" data/cache/release-test/
git bundle verify data/cache/release-test/decklight.bundle > /dev/null
N=fresh-clone-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "NP=${MIN_CPUS:-16} bash scripts/release/fresh_clone_test.sh $N" --setup "true" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.15}" --min-cpus "${MIN_CPUS:-16}" --min-gpu-ram 6 \
  --data data/cache/release-test \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
