#!/usr/bin/env bash
# Build and run a local Multirole (the server behind EDOPro's public lobbies) for client tests.
#
#   scripts/multirole_local.sh [WORKDIR]      # default WORKDIR: data/cache/multirole
#   uv run python scripts/audit/multirole_check.py --games 3 --bot raw-blueeyes
#
# Build tools (meson, ninja) come from uv. Multirole's C++ libraries (Boost >= 1.78, libgit2,
# OpenSSL, SQLite, fmt) are not Python packages; without root (no apt) they come from a conda
# env inside WORKDIR. With podman or docker, Multirole's own Dockerfile is the alternative.
# The server clones its card repos from local git repos (our third_party/CardScripts and
# BabelCDB, an empty banlist, our built libocgcore.so), so its card set is ours: nothing big
# is downloaded. Ports: 7911 rooms, 7922 room listing (JSON). Stop it with `kill $(cat WORKDIR/run/multirole.pid)`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mkdir -p "${1:-$ROOT/data/cache/multirole}" && cd "${1:-$ROOT/data/cache/multirole}" && pwd)"
CONDA="${CONDA:-conda}"
cd "$W"

[[ -d Multirole ]] || git clone --depth 1 https://github.com/ProjectIgnis/Multirole.git
[[ -d env ]] || "$CONDA" create -y -q -p ./env -c conda-forge "libboost-devel>=1.78" libgit2 openssl sqlite "fmt>=8" \
    "gxx_linux-64=13" pkg-config
[[ -d tools ]] || { uv venv -q tools && uv pip install -q -p tools/bin/python meson ninja; }

E="$W/env"
if [[ ! -x Multirole/build/multirole ]]; then
  (cd Multirole && PATH="$W/tools/bin:$PATH" PKG_CONFIG_PATH="$E/lib/pkgconfig:$E/share/pkgconfig" \
     BOOST_INCLUDEDIR="$E/include" BOOST_LIBRARYDIR="$E/lib" \
     CC="$E/bin/x86_64-conda-linux-gnu-gcc" CXX="$E/bin/x86_64-conda-linux-gnu-g++" \
     LDFLAGS="-Wl,-rpath,$E/lib -L$E/lib" CPPFLAGS="-I$E/include" \
     bash -c 'meson setup build -Doptimization=2 -Duse_tcmalloc=disabled && taskset -c 0-3 meson compile -C build -j 3')
fi

[[ -f "$ROOT/build/core/libocgcore.so" ]] || { echo "build the core first (build/core/libocgcore.so)" >&2; exit 1; }
mkdir -p repos run
for n in banlists bin; do
  rm -rf "repos/$n"; mkdir "repos/$n"; git -C "repos/$n" init -q
done
printf '#[No limit]\n!No limit\n' > repos/banlists/nolimit.lflist.conf
cp "$ROOT/build/core/libocgcore.so" repos/bin/
for n in banlists bin; do
  git -C "repos/$n" add -A && git -C "repos/$n" -c user.name=local -c user.email=local@localhost commit -qm init
done

python3 - "$ROOT" "$W" <<'PY'
import json, sys
root, w = sys.argv[1:]
c = json.load(open(f"{w}/Multirole/etc/config.json"))
src = {"scripts": f"{root}/third_party/CardScripts", "databases": f"{root}/third_party/BabelCDB",
       "banlists": f"{w}/repos/banlists", "bin": f"{w}/repos/bin"}
for r in c["repos"]:
    r["remote"] = "file://" + src[r["name"]]
    r["webhookToken"] = "local-test"
c["lobbyMaxConnections"] = 16
json.dump(c, open(f"{w}/run/config.json", "w"), indent=1)
PY

cd run
rm -rf sync tmp
cp ../Multirole/build/multirole ../Multirole/build/hornet .
taskset -c 0-3 ./multirole > multirole.log 2>&1 &
echo $! > multirole.pid
for _ in $(seq 1 90); do
  if ss -ltn | grep -q ":7911 "; then echo "multirole up (pid $(cat multirole.pid)): rooms :7911, listing :7922"; exit 0; fi
  sleep 2
done
echo "multirole did not open :7911 in 180 s; see $W/run/multirole.log" >&2
exit 1
