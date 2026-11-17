#!/usr/bin/env bash
# Cycle-1 league segment (docs/lab/prereg/2026-11-10-master-cycle1.md, amendments 6-7; scripts/vast_c1_league.sh).
# Stages data/cache/c1_league: X2 (init), X (card opponent), X's value head with X2's arch file (value init: the
# export's invariant/load checks build the value net from the metrics.jsonl beside it, and the trunk is X2's), the 25
# accepted specialists (policy + train lists only), X's and X2's G4 re-plays, the c1 round's verify map, verify pairs,
# card decks and R rows (games + summaries), and the mirror pairs (the reserve lists, each against itself).
# Env: Y_FROM (an export dir: evaluate it, no training), HOURS (8), MAX_DPH (0.26), MIN_CPUS (32), MIN_GPU_RAM (16), RESERVE (0.25), plus the job's env (ITERS, ...).
cd "$(dirname "$0")/../.."
G=data/cache/c1_league; X=runs/window-ppo-20261108-0940/train/export-X; R=runs/c1-round-20261110-1927
G4=runs/c1-g4-20261111-0235
rm -rf $G; mkdir -p $G/X $G/X2 $G/Xv $G/packs $G/vpairs $G/cardR $G/card-decks
cp $X/policy/{best.pt,metrics.jsonl} $G/X/ && cp $R/X2/{best.pt,metrics.jsonl} $G/X2/ \
  && cp $X/value/best.pt $G/Xv/ \
  && python3 -c "import json,sys; r=json.loads(open(sys.argv[1]).readline()); r['value']=r['args']['value']=True; open(sys.argv[2],'w').write(json.dumps(r)+'\\n')" \
       $R/X2/metrics.jsonl $G/Xv/metrics.jsonl \
  && cp $G4/eval-X.jsonl $G4/eval-X2.jsonl $R/verify.json $G/ && cp $R/verify/*.pairs $G/vpairs/ \
  && cp $R/decks/*.ydk $G/card-decks/ || { echo "staging $G failed"; exit 3; }
for d in data/forge/c1/packs/*/; do
  a=$(basename "$d"); mkdir -p $G/packs/$a/policy $G/packs/$a/decks
  cp "$d"pack.json $G/packs/$a/ && cp "$d"policy/{weights.safetensors,metrics.jsonl} $G/packs/$a/policy/ \
    && cp -r "$d"decks/train $G/packs/$a/decks/ || { echo "staging pack $a failed"; exit 3; }
done
for d in $R/card/R-*/; do
  t=$(basename "$d"); mkdir -p $G/cardR/$t; cp "$d"games-*.jsonl "$d"summary.json $G/cardR/$t/ || { echo "staging $t failed"; exit 3; }
done
python3 -c "import json; print(' '.join(json.load(open('$R/release_card.json'))['old']))" > $G/old_archs.txt
if [ -n "${Y_FROM:-}" ]; then  # resume: evaluate an exported Y (E and C only)
  mkdir -p $G/Y && cp "$Y_FROM"/policy/{best.pt,metrics.jsonl} $G/Y/ || { echo "staging Y failed"; exit 3; }
fi
tr ':' '\n' < decks/forge/f2/pairs/reserve.pairs | sort -u | awk '{print $1 ":" $1}' > $G/mirror.pairs
python3 scripts/tsoj/deck_opps.py --packs $G/packs --drop sky-striker --want 25 --out $G/opps || exit 3
for f in $G/X/best.pt $G/X2/best.pt $G/Xv/best.pt $G/eval-X.jsonl $G/eval-X2.jsonl $G/verify.json $G/old_archs.txt \
         data/cards/emb/qwen3-0.6b-d512.npz data/cache/x_window/best.pt data/cache/value_p2/best.pt; do
  [ -s "$f" ] || { echo "missing $f"; exit 3; }
done
[ "$(wc -w < $G/old_archs.txt)" = 14 ] && [ "$(ls -d $G/cardR/R-* | wc -l)" = 41 ] && [ "$(wc -l < $G/mirror.pairs)" = 24 ] \
  || { echo "staging counts wrong: old $(wc -w < $G/old_archs.txt), R $(ls -d $G/cardR/R-* | wc -l), mirror $(wc -l < $G/mirror.pairs)"; exit 3; }
du -sh $G
N=c1-league-$(date +%Y%m%d-%H%M); echo $N
ENVS="ITERS=${ITERS:-80} KL=${KL:-0.03} EXT_SHARE=${EXT_SHARE:-0.2}"
[ -n "${Y_FROM:-}" ] && ENVS="$ENVS Y_IN=$G/Y/best.pt"
taskset -c 0-3 uv run --no-sync python scripts/vast_job.py --name $N \
  --cmd "$ENVS bash scripts/vast_c1_league.sh $N" \
  --hours "${HOURS:-8}" --reserve "${RESERVE:-0.25}" --max-dph "${MAX_DPH:-0.26}" --min-cpus "${MIN_CPUS:-32}" \
  --min-gpu-ram "${MIN_GPU_RAM:-16}" --max-inet-cost "${MAX_INET:-0.005}" \
  --extra-query "compute_cap>=800 cpu_ram>=${MIN_RAM:-64}" \
  --data decks data/cards/emb data/cards/lua_desc.npz data/models data/cache/P data/cache/x_window data/cache/value_p2 $G \
  > data/cache/briefs/$N.launcher.log 2>&1
echo "exit $?"
