# Usable cores on a vast box: the cgroup quota (v2 cpu.max or v1 cfs_quota/period) if set, else every core.
# `nproc` alone is wrong: GNU nproc honours OMP_NUM_THREADS, which the job scripts set to 1 (stage-l-20261012-0156 ran
# 1 proc on a 128-core box, NOTEBOOK 2026-10-12 02:45). Usage: . scripts/cores.sh; J=$(box_cores)
box_cores() {
  local q p n
  n=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
  if read -r q p 2>/dev/null < /sys/fs/cgroup/cpu.max && [ "$q" != max ]; then n=$(( q / p ))
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$q" -gt 0 ] && n=$(( q / p ))
  fi
  [ "$n" -lt 1 ] && n=1
  echo "$n"
}
