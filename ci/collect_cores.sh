#!/usr/bin/env bash
set -euo pipefail

cores_dir="${1:-/var/lib/polyregion-cores}"
out_dir="${2:-crash-reports}"
per_exe="${CORES_PER_EXE:-3}"

mkdir -p "$out_dir"

# core_pattern is core.%E.%p (executable path with / as !): rank each core newest-first within its executable, then
# order by rank so the newest core of every executable comes before any second core
mapfile -t cores < <(
  find "$cores_dir" -maxdepth 1 -type f -name 'core.*' -printf '%T@\t%f\n' \
    | sort -t$'\t' -k1,1rn \
    | awk -F'\t' -v k="$per_exe" '
        { exe = $2; sub(/^core\./, "", exe); sub(/\.[0-9]+$/, "", exe)
          rank = ++seen[exe]
          if (rank <= k) print rank "\t" $2 }' \
    | sort -t$'\t' -k1,1n -s \
    | cut -f2
)

if [ ${#cores[@]} -eq 0 ]; then
  echo "no cores in $cores_dir"
else
  command -v gdb >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq gdb || true; }
  for c in "${cores[@]}"; do
    exe=${c#core.}
    exe=${exe%.*}
    exe=${exe//!//}
    [ -f "$exe" ] || exe=
    gdb -batch -ex 'thread apply all bt' ${exe:+"$exe"} -c "$cores_dir/$c" > "$out_dir/${c##*!}.txt" 2>&1 || true
  done
fi
ls -la "$out_dir" || true
