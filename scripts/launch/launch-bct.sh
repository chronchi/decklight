#!/usr/bin/env bash
# M1 teacher BC (ADR 0006): fine-tune rl_init (ft-mix) with ygo-agent teacher games; report card for all 3 variants.
cd "$(dirname "$0")/../.."
N=bct-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/bct.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/rl_init/best.pt VARIANTS='zero ft-teach ft-teach-h' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent' bash scripts/vast_bc_gold.sh $N" \
  --hours 3 --max-dph 0.15 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=48" \
  --data data/demos data/cards/emb decks data/models data/cache/rl_init > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -4 data/cache/briefs/$N.launcher.log; cat runs/$N/timing.txt runs/$N/summary.txt 2>/dev/null
for f in runs/$N/playtest-*.txt; do echo "== $f"; grep -E "^combo mean|^vs " "$f"; done
