#!/usr/bin/env bash
# usage: run.sh <bundle-dir|-> <cmd...>
set -euo pipefail
bundle=$1
shift
[ "$bundle" != - ] && . "$bundle/env.sh" >/dev/null
exec "$@"
