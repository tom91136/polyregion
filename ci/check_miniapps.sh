#!/usr/bin/env bash

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DIST="$ROOT/native/out/polyregion-Release-$(uname -m)-dylib-dist"
SRC="$HERE/check_miniapps"
BUILD="$SRC/_ctest"

[ -x "$DIST/bin/polycpp" ] || { echo "no dist at $DIST - run 'just build-dist' first" >&2; exit 1; }

# emulator bundle: explicit env, else the in-repo bundle (just build-emulators), else none -> cpu-only
EMU="${POLYREGION_EMULATORS_HOME:-$ROOT/emulators/out}"
[ -f "$EMU/env.sh" ] || EMU=""
[ -n "$EMU" ] && echo "emulators: $EMU" || echo "emulators: none (cpu-only)"

cmake -S "$SRC" -B "$BUILD" -DPOLYCPP_DIST="$DIST" \
  ${EMU:+"-DMINIAPP_EMULATORS=$EMU"} \
  ${MINIAPP_BACKENDS:+"-DMINIAPP_BACKENDS=$MINIAPP_BACKENDS"}
ctest --test-dir "$BUILD" --output-on-failure -j "$(nproc 2>/dev/null || echo 2)"
