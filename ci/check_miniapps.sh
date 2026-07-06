#!/usr/bin/env bash

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ARCH="${POLYREGION_ARCH:-$(uname -m)}"
DIST="$ROOT/native/out/polyregion-Release-${ARCH}-dylib-dist"
SRC="$HERE/check_miniapps"
BUILD="$SRC/_ctest-${ARCH}"

[ -x "$DIST/bin/polycpp" ] || { echo "no dist at $DIST - run 'just build-dist' first" >&2; exit 1; }

# emulator bundle: env, then in-repo (just build-emulators), else cpu-only.
EMU="${POLYREGION_EMULATORS_HOME:-$ROOT/emulators/out}"
[ -f "$EMU/env.sh" ] || EMU=""
[ -n "$EMU" ] && echo "emulators: $EMU" || echo "emulators: none (cpu-only)"

# XXX cap both ctest -j and cmake --parallel; else nproc^2 compilers OOMs tight hosts
if [ -z "${MINIAPP_JOBS:-}" ]; then
  N=$(nproc 2>/dev/null || echo 2)
  MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
  [ "$MEM_KB" -gt 0 ] && M=$(( MEM_KB / 4 / 1024 / 1024 )) && [ "$M" -lt "$N" ] && N="$M"
  [ "$N" -lt 1 ] && N=1
  MINIAPP_JOBS="$N"
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$MINIAPP_JOBS"
echo "concurrency: MINIAPP_JOBS=$MINIAPP_JOBS (override via env)"

cmake -S "$SRC" -B "$BUILD" -DPOLYCPP_DIST="$DIST" -DMINIAPP_TARGET_ARCH="$ARCH" \
  ${EMU:+"-DMINIAPP_EMULATORS=$EMU"} \
  ${MINIAPP_BACKENDS:+"-DMINIAPP_BACKENDS=$MINIAPP_BACKENDS"}
ctest --test-dir "$BUILD" --output-on-failure -j "$MINIAPP_JOBS"
