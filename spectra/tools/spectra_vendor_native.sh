#!/usr/bin/env bash
set -euo pipefail

backend=${1:?usage: spectra_vendor_native.sh <cuda|hip|onedpl> <compile|run>}
mode=${2:-compile}
jobs=${SPECTRA_NATIVE_JOBS:-4}
source_root=${SPECTRA_SOURCE_ROOT:-/spectra}
vendor_root=${SPECTRA_VENDOR_ROOT:-/vendor}
output_root=${SPECTRA_OUTPUT_ROOT:-/out}
binary_dir="$output_root/native-$backend"

case "$mode" in
  compile|run) ;;
  *) echo "unknown native implementation mode: $mode" >&2; exit 2 ;;
esac
case "$jobs" in
  ''|*[!0-9]*|0) echo "SPECTRA_NATIVE_JOBS must be a positive integer" >&2; exit 2 ;;
esac

mkdir -p "$binary_dir"
mapfile -d '' sources < <(find "$source_root/test/generated" -name '*.cpp' -print0 | sort -z)
if [[ ${#sources[@]} -eq 0 ]]; then
  echo "Spectra vendor-native validation found no generated cases" >&2
  exit 2
fi

compile_one() {
  local source=$1 name log
  name=${source##*/}
  name=${name%.cpp}
  log="$binary_dir/$name.log"
  case "$backend" in
    cuda)
      nvcc --allow-unsupported-compiler --extended-lambda -x cu -std=c++17 \
        -Xcompiler=-Wno-subobject-linkage \
        -DSPECTRA_NATIVE_THRUST=1 \
        -I"$vendor_root/cuda_cccl/include" \
        -I"$source_root/include" -I"$source_root/generated/cpp/include" -I"$source_root/test/generated" \
        "$source" -o "$binary_dir/$name" >"$log" 2>&1
      ;;
    hip)
      hipcc --offload-arch="${SPECTRA_HIP_ARCH:-gfx90a}" -x hip -std=c++17 \
        -DSPECTRA_NATIVE_THRUST=1 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP \
        -I"$vendor_root/rocm/rocThrust" -I"$vendor_root/rocm/rocPRIM/rocprim/include" \
        -I"$source_root/include" -I"$source_root/generated/cpp/include" -I"$source_root/test/generated" \
        "$source" -o "$binary_dir/$name" >"$log" 2>&1
      ;;
    onedpl)
      icpx -fsycl -std=c++17 -DSPECTRA_NATIVE_ONEDPL=1 \
        -I"$vendor_root/onedpl/include" \
        -I"$source_root/include" -I"$source_root/generated/cpp/include" -I"$source_root/test/generated" \
        "$source" -o "$binary_dir/$name" >"$log" 2>&1
      ;;
    *) echo "unknown native implementation backend: $backend" >&2; return 2 ;;
  esac
}
export backend binary_dir source_root vendor_root
export -f compile_one

status=0
printf '%s\0' "${sources[@]}" | xargs -0 -n1 -P"$jobs" bash -c '
  if ! compile_one "$1"; then
    name=${1##*/}; name=${name%.cpp}
    echo "FAILED $name" >&2
    tail -80 "$binary_dir/$name.log" >&2
    exit 1
  fi
' _ || status=$?
if [[ $status -ne 0 ]]; then
  echo "Spectra $backend vendor-native validation compile failed" >&2
  exit "$status"
fi
echo "Spectra $backend vendor-native validation: ${#sources[@]}/${#sources[@]} compiled and linked"

if [[ $mode == run ]]; then
  passed=0
  for source in "${sources[@]}"; do
    name=${source##*/}
    name=${name%.cpp}
    if ! "$binary_dir/$name" >"$binary_dir/$name.run.log" 2>&1; then
      echo "FAILED $name" >&2
      tail -80 "$binary_dir/$name.run.log" >&2
      exit 1
    fi
    ((passed += 1))
  done
  echo "Spectra $backend vendor-native validation: $passed/${#sources[@]} ran successfully"
fi
