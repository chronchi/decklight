#!/usr/bin/env bash
# M3 mix (ADR 0006/0007): from ft-teach-h, + DAgger + window search (ft-dw) and + own-turn/turn-1 search (ft-all);
# report card for all three (zero = ft-teach-h itself).
cd "$(dirname "$0")/../.."
N=mix-$(date +%Y%m%d-%H%M); echo $N; echo $N > data/cache/briefs/mix.tag
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_teach_h/best.pt VARIANTS='zero ft-dw ft-all' PLAY_ALL=1 SKIP_SCRATCH=1 PT_ZERO=0 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards' bash scripts/vast_bc_gold.sh $N" \
  --hours 3.5 --max-dph 0.15 --min-cpus 16 --min-gpu-ram 11 --extra-query "compute_cap>=800 cpu_ram>=48" \
  --data data/demos data/cards/emb decks data/models data/cache/base_teach_h > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"; tail -4 data/cache/briefs/$N.launcher.log; cat runs/$N/timing.txt runs/$N/summary.txt 2>/dev/null
for f in runs/$N/playtest-*.txt; do echo "== $f"; grep -E "^combo mean|^vs " "$f"; done
