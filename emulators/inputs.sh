#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-$(uname -s)}" in
  linux|Linux)               files=(Dockerfile collect.sh spirv-tools.ref gpuocelot.patch mesa.patch) ;;
  macos|Darwin)              files=(macos.sh spirv-tools.ref mesa.patch) ;;
  windows|MINGW*|MSYS*|CYGWIN*) files=(windows.bat spirv-tools.ref gpuocelot.patch gpuocelot.windows.patch mesa.patch) ;;
  *) echo "inputs.sh: unsupported OS $1" >&2; exit 1 ;;
esac
sum() { sha256sum 2>/dev/null || shasum -a 256; }
cd "$DIR"
for f in "${files[@]}"; do
  printf '%s\n' "$f"
  tr -d '\r' < "$f"
done | sum | cut -d' ' -f1
