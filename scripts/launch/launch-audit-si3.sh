#!/usr/bin/env bash
# Game audit of selfimit r3 (ft-si-kl03 of selfimit-ft-r3-20261007-2012: vs ygo-agent 33.9% vs base_r3 42.3% on the
# same 248 deals, 35 W->L vs 14 L->W, NOTEBOOK 2026-10-08 03:00): the checkpoint through the exact audit-r3 card
# (BASE base_r2, same decks/seeds, games saved + rendered) so each ygo-agent game diffs decision by decision against
# base_r3's (runs/audit-r3-20261006-1641) with scripts/audit/diff_games.py.
cd "$(dirname "$0")/../.."
N=audit-si3-$(date +%Y%m%d-%H%M); echo $N
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "BASE=data/cache/base_r2/best.pt CKPTS='data/cache/audit_si3/kl03' RENDER=1 PT_ARGS='--decks 8 --hands 3 --seeds 2 --narrate 0 --samples 4 --ygoagent --newcards decks/newcards --save-games 1000' bash scripts/vast_card.sh $N" \
  --hours 1 --max-dph 0.25 --min-cpus 24 --min-gpu-ram 11 --extra-query "cpu_ram>=48" \
  --data data/demos data/cards/emb decks data/models data/cache/base_r2 data/cache/audit_si3 > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
