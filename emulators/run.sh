#!/usr/bin/env bash
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$(cd "${1:-$DIR/out}" 2>/dev/null && pwd || true)"
[ -n "$BUNDLE" ] && [ -f "$BUNDLE/env.sh" ] || { echo "no bundle env.sh (run 'just build-emulators' first)"; exit 2; }
# shellcheck source=/dev/null
source "$BUNDLE/env.sh"
B="$(mktemp -d)"; trap 'rm -rf "$B"' EXIT
CXX="${CXX:-g++}"
rc=0

echo "== OpenCL (rusticl + pocl) =="
if "$CXX" -O2 -std=c++17 "$DIR/vecadd.cpp" -o "$B/cl" \
      -I"$BUNDLE/ocl/include" -L"$BUNDLE/ocl/lib" -lOpenCL -Wl,-rpath,"$BUNDLE/ocl/lib"; then
  "$B/cl" || rc=1
else echo "  build FAIL"; rc=1; fi

echo "== CUDA (gpuocelot) =="
# XXX sm_35 + -O2 as gpuocelot is validated on Fermi, don't push our luck
if ! command -v clang >/dev/null; then
  echo "  no clang with NVPTX on PATH, skipping"
elif clang -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_35 -O2 -S "$DIR/vecadd.cu" -o "$B/vecadd.ptx" 2>/dev/null \
     && "$CXX" -O2 -std=c++17 -x c++ "$DIR/vecadd.cu" -o "$B/cu" -L"$BUNDLE/ocelot" -lcuda -Wl,-rpath,"$BUNDLE/ocelot"; then
  "$B/cu" "$B" || rc=1
else echo "  build FAIL"; rc=1; fi

[ $rc -eq 0 ] && echo "SMOKE OK" || echo "SMOKE FAILED"
exit $rc
