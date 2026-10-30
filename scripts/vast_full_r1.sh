#!/usr/bin/env bash
# Full-r1 product card (pre-registered NOTEBOOK 2026-10-30 "Full-r1 product card"; user: "Can you test r1 to become the
# product?"). Only the policy changes: r1 (data/cache/forge_r1) + value_p2 + the product search (scripts/product.sh).
#   full_r1    : vs ygo-agent on the stored product arm's 1,488 deals (seeds 9000+, GA 6, both seats), paired with
#                runs/full-p-ho-20261021-1204/full_p_ho (Full-P + ho, 0.623).
#   fz_r1/fz_p : the 11 Forge archetypes (data/forge/f3/pairs, 324 pairs x 1 seed x 2 seats = 648 games per arm),
#                searched r1 vs searched P, both against raw r0 (x4b_D), same deals (seed-per-pair, 51000+).
#   full_p_rep : Full-P + ho again on the first 248 gauntlet deals (GA 1): drift check of today's code.
# Order = fail-fast: full_r1 first (milestone checked from the laptop at ~250 games).
# Usage (on the box, via scripts/launch/launch-full-r1.sh): bash scripts/vast_full_r1.sh <name>
set -uo pipefail
NAME=${1:?name}
cd "$(dirname "$0")/.."
. scripts/product.sh
OUT=runs/$NAME
mkdir -p "$OUT/decks"
alert() { echo "ALERT $*" | tee -a "$OUT/ALERT"; }
R1=data/cache/forge_r1/best.pt; R0=data/cache/x4b_D/best.pt
for f in "$R1" "$R0" "$PRODUCT_CKPT" "$PRODUCT_VALUE"; do [ -f "$f" ] || { alert "missing $f"; exit 3; }; done
find decks/x4b decks/forge -name '*.ydk' -exec cp {} "$OUT/decks/" \;
cat data/forge/f3/pairs/*.pairs | grep -v '^\s*$' > "$OUT/forge.pairs"
NP=$(wc -l < "$OUT/forge.pairs")
[ "$NP" -ge 300 ] || { alert "forge.pairs has $NP pairs, want 324"; exit 3; }

H="$PRODUCT_SEARCH --claim 1 --value $PRODUCT_VALUE"
FZ="--opp ckpt --opp-ckpt $R0 --pairs @$OUT/forge.pairs --seeds 1 --seed-per-pair 1 --deal-seed0 51000 --deck-dir $OUT/decks"
GS=${GA_SEEDS:-6}; LIM=""
[ "${SMOKE:-0}" = 1 ] && { GS=1; LIM="--games-limit ${SMOKE_GAMES:-8}"; }
V="full_r1|--ckpt $R1 --opp ygoagent --ga-seeds $GS --deal-seed0 9000 $LIM"
V="$V;fz_r1|--ckpt $R1 $FZ $LIM"
V="$V;fz_p|--ckpt $PRODUCT_CKPT $FZ $LIM"
V="$V;full_p_rep|--ckpt $PRODUCT_CKPT --opp ygoagent --ga-seeds 1 --deal-seed0 9000 $LIM"
echo "variants: $V" >> "$OUT/PROGRESS"
# The probe sizes procs on the forge config (new: searched vs a ckpt opponent on pairs), so a broken config fails here.
CKPT=$PRODUCT_CKPT SM_ARGS="$H" VARIANTS="$V" PROBE_ARGS="--ckpt $R1 $FZ" \
  PROBE_NPROC=${PROBE_NPROC:-40} VAR_CAP=${VAR_CAP:-260m} GPU_MARGIN=${GPU_MARGIN:-1.3} \
  bash scripts/vast_search_match.sh "$NAME"
rc=$?
for t in full_r1 fz_r1 fz_p full_p_rep; do
  n=$(cat "$OUT/$t"/games-*.jsonl 2>/dev/null | wc -l)
  echo "$t games $n" >> "$OUT/PROGRESS"
  [ "$n" -gt 0 ] || alert "$t has 0 games"
done
exit $rc
