#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="${1:-$DIR/out}"
expected="$(bash "$DIR/inputs.sh")"
actual="$(cat "$BUNDLE/.inputs.sha256" 2>/dev/null || true)"
if [ "$actual" != "$expected" ]; then
  echo "emulator bundle $BUNDLE is stale: inputs ${actual:-unstamped}, expected $expected; run 'just build-emulators'" >&2
  exit 1
fi
