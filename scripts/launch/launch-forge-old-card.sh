#!/usr/bin/env bash
# Forge F2 confirmation card (user 2026-10-30: "Confirmation card"; pre-registered NOTEBOOK 2026-10-30 09:20): the round's
# old-archetype gate was underpowered (768 games each). Independent card, fresh deal seeds 47000: r0 (D) vs r1 on the 5
# X4b archetypes' pairs, 24 seeds per pair = 1,536 games per archetype per arm (7,680 pooled), raw, opponent r0 raw.
# Gate (unchanged margin): pooled r1 - r0 lower 95% > -1.5 pp -> r1 is release 1.
# Analysis: scripts/audit/forge_release_card.py --old (it also reports per archetype; exit 2 is expected: no res/ga arms).
# Env: HOURS (1.5), MAX_DPH (0.30), SEEDS (24), SMOKE=1 (--games-limit 256).
cd "$(dirname "$0")/../.."
ARCHS="branded-light-and-darkness-ritual invoked radiant-typhoon sacred-beasts atlantean-mermail"
R0=data/cache/x4b_D/best.pt; R1=data/cache/forge_r1/best.pt
for f in $R0 $R1 data/cache/forge_r1/metrics.jsonl; do [ -f "$f" ] || { echo "missing $f"; exit 3; }; done
cmp -s $R0 $R1 && { echo "r1 is r0"; exit 3; }
N=forge-oldcard-$(date +%Y%m%d-%H%M)${SMOKE:+-smoke}; echo $N
V=""
for a in $ARCHS; do
  pf="--pairs @decks/x4b/pairs/arch-$a.pairs --seeds ${SEEDS:-24}"
  V="$V;R-$a|$pf;N-$a|--ckpt $R1 $pf"
done
V="${V#;}"
SM="--raw 1 --opp ckpt --opp-ckpt $R0 --seed-per-pair 1 --deal-seed0 47000 --deck-dir data/x4b_flat"
[ "${SMOKE:-0}" = 1 ] && SM="$SM --games-limit 256"
echo '{}' > data/cache/forge_r1/empty_verify.json
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "mkdir -p data/x4b_flat && find decks/x4b -name '*.ydk' -exec cp {} data/x4b_flat/ \; && PROBE_NPROC=${PROBE_NPROC:-40} CKPT=$R0 SM_ARGS='$SM' VARIANTS='$V' VAR_CAP=${VAR_CAP:-30m} bash scripts/vast_search_match.sh $N/card; uv run --no-sync python scripts/audit/forge_release_card.py runs/$N/card --verify data/cache/forge_r1/empty_verify.json --old $ARCHS --name r1 > runs/$N/old_card.txt 2>&1; grep -q '^GATE old' runs/$N/old_card.txt || echo \"ALERT old card: \$(tail -1 runs/$N/old_card.txt)\" >> runs/$N/ALERT" \
  --hours "${HOURS:-1.5}" --max-dph "${MAX_DPH:-0.30}" --min-cpus "${MIN_CPUS:-24}" --min-gpu-ram "${MIN_GPU_RAM:-16}" \
  --reserve "${RESERVE:-0.10}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-48} gpu_name!=CMP_170HX ${EXTRA_Q:-}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/x4b_D data/cache/forge_r1 \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
