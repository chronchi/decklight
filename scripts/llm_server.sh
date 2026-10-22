#!/bin/sh
# Long-running opencode server for fast muse calls (decklight.demos.llm, YGO_OPENCODE_SERVER=http://127.0.0.1:4097).
# Laptop rule: pinned to cores 0-3, nice 19. The OpenCode key stays on the laptop (opencode's own auth).
#   scripts/llm_server.sh start | stop | status
cd "$(dirname "$0")/.." || exit 1
PID=data/cache/llm_server.pid
case "$1" in
  start)
    if [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null; then echo "running (pid $(cat $PID))"; exit 0; fi
    OPENCODE_CONFIG="$PWD/config/opencode-llm.json" nohup taskset -c 0-3 nice -n 19 opencode serve --hostname 127.0.0.1 --port 4097 > data/cache/llm_server.log 2>&1 &
    echo $! > $PID; echo "started (pid $!)";;
  stop) [ -f $PID ] && kill "$(cat $PID)" && rm -f $PID && echo stopped;;
  status) [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null && echo "running (pid $(cat $PID))" || echo "not running";;
  *) echo "usage: $0 start|stop|status"; exit 2;;
esac
