#!/usr/bin/env bash
# usage: run.sh <bundle-dir|-> <polytest> <audit-selector|-> <images-manifest> <cmd...>
set -euo pipefail
bundle=$1
polytest=$2
selector=$3
images=$4
shift 4
[ "$bundle" != - ] && . "$bundle/env.sh" >/dev/null
[ "$selector" = - ] && exec "$@"
exec "$polytest" --audit-skip "$selector" "$images" -- "$@"
