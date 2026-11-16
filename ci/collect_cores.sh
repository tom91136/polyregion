#!/usr/bin/env bash
set -euo pipefail

cores_dir="${1:-/var/lib/polyregion-cores}"
out_dir="${2:-crash-reports}"
per_exe="${CORES_PER_EXE:-3}"
max_cores="${CORES_MAX:-24}"
max_raw="${CORES_UPLOAD_MAX:-2}"
max_raw_bytes="${CORES_UPLOAD_MAX_BYTES:-419430400}"

mkdir -p "$out_dir"

# core_pattern is core.%E.%s.%p (path with / as !, signal, pid): crash signals rank first, then newest-first per exe and signal
mapfile -t cores < <(
  find "$cores_dir" -maxdepth 1 -type f -name 'core.*' -printf '%T@\t%f\n' \
    | sort -t$'\t' -k1,1rn \
    | awk -F'\t' -v k="$per_exe" '
        { name = $2; sub(/^core\./, "", name); sub(/\.[0-9]+$/, "", name)
          sig = name; sub(/^.*\./, "", sig)
          rank = ++seen[name]
          if (rank <= k) print ((sig ~ /^(4|5|7|8|11)$/) ? 0 : 1) "\t" rank "\t" $2 }' \
    | sort -t$'\t' -k1,1n -k2,2n -s \
    | cut -f3 \
    | head -n "$max_cores"
)

if [ ${#cores[@]} -eq 0 ]; then
  echo "no cores in $cores_dir"
else
  command -v gdb >/dev/null || { sudo apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update -qq && sudo apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y -qq gdb || true; }
  raw=0
  for c in "${cores[@]}"; do
    exe=${c#core.}
    exe=${exe%.*}
    sig=${exe##*.}
    exe=${exe%.*}
    exe=${exe//!//}
    [ -f "$exe" ] || exe=
    name=${c##*!}
    qemu=
    case "${exe##*/}" in qemu-*) qemu=1 ;; esac
    (
      [ -z "$qemu" ] || export DEBUGINFOD_URLS="${DEBUGINFOD_URLS:-https://debuginfod.ubuntu.com}" DEBUGINFOD_TIMEOUT=60
      timeout 600 gdb -batch \
        -ex 'set pagination off' \
        -ex 'set debuginfod enabled on' \
        -ex 'info threads' \
        -ex 'p $_siginfo' \
        -ex 'bt full' \
        -ex 'info registers' \
        -ex 'x/8i $pc' \
        -ex 'thread apply all bt full' \
        ${exe:+"$exe"} -c "$cores_dir/$c"
    ) > "$out_dir/$name.txt" 2>&1 || true
    if [ -n "$qemu" ] && [[ "$sig" =~ ^(4|5|7|8|11)$ ]] && [ "$raw" -lt "$max_raw" ]; then
      dest="$out_dir/raw/$name"
      mkdir -p "$dest"
      if command -v zstd >/dev/null; then zstd -q -T0 -3 -o "$dest/core.zst" "$cores_dir/$c" || true
      else gzip -3 -c "$cores_dir/$c" > "$dest/core.gz" || true; fi
      if [ "$(cat "$dest"/core.* 2>/dev/null | wc -c)" -gt "$max_raw_bytes" ]; then rm -rf "$dest"
      else cp -L "$exe" "$dest/"; raw=$((raw + 1)); fi
    fi
  done
fi
ls -laR "$out_dir" || true
