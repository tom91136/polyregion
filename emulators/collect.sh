#!/usr/bin/env bash
set -euo pipefail

OUT=/out
mkdir -p "$OUT"/lib "$OUT"/icd

COMPONENTS=(mesa ocl pocl vkloader ocelot swiftshader)
for c in "${COMPONENTS[@]}"; do cp -a "/opt/$c" "$OUT/$c"; done
OUTDIRS=("${COMPONENTS[@]/#/$OUT/}")
ln -sf libgpuocelot.so "$OUT"/ocelot/libcuda.so.1
ln -sf libgpuocelot.so "$OUT"/ocelot/libcuda.so

find1()  { find "$@" 2>/dev/null | head -1; }
reldir() { [ -n "$1" ] || return 0; realpath --relative-to="$OUT" "$(dirname "$1")"; }
soname() { local b; b=$(basename "$1"); echo "${b%%.so*}.so"; }

RUSTICL=$(find1   "$OUT"/mesa -name 'libRusticlOpenCL.so.*' -type f)
LVPLIB=$(find1    "$OUT"/mesa -name 'libvulkan_lvp.so'      -type f)
LVPJSON=$(find1   "$OUT"/mesa -name 'lvp_icd*.json'         -type f)
OCLLOADER=$(find1 "$OUT"/ocl  -name 'libOpenCL.so.1*'       -type f)
POCLLIB=$(find1   "$OUT"/pocl -name 'libpocl.so*'           -type f)
SSLIB=$(find1     "$OUT"/swiftshader -name 'libvk_swiftshader.so' -type f)
SSJSON=$(find1    "$OUT"/swiftshader -name 'vk_swiftshader_icd.json')

[ -n "$OCLLOADER" ] || { echo "ERROR: ocl-icd libOpenCL.so.1 not found"; exit 1; }
[ -n "$RUSTICL" ]   || { echo "ERROR: libRusticlOpenCL not found"; exit 1; }

# rewrite ICD entries to bare sonames, resolved via LD_LIBRARY_PATH
sed -i "s|\"library_path\":[^,]*|\"library_path\": \"$(soname "$LVPLIB")\"|" "$LVPJSON"
[ -n "$SSLIB" ] && sed -i "s|\"library_path\":[^,]*|\"library_path\": \"$(soname "$SSLIB")\"|" "$SSJSON"
soname "$RUSTICL" > "$OUT"/icd/rusticl.icd
[ -n "$POCLLIB" ] && soname "$POCLLIB" > "$OUT"/icd/pocl.icd || echo "WARN: libpocl.so not found, pocl not exposed"

hostlibs='ld-linux|libc\.so|libm\.so|libdl\.so|libpthread|librt\.so|libresolv|libnsl|ld64|libstdc\+\+|libgcc_s'
collect_deps() {
  ldd "$1" 2>/dev/null | awk '/=>/ {print $3}' | while read -r so; do
    [ -e "$so" ] || continue
    basename "$so" | grep -qE "$hostlibs" && continue
    case "$so" in "$OUT"/*) continue;; esac
    cp -aL "$so" "$OUT"/lib/ 2>/dev/null || true
  done
}
find "${OUTDIRS[@]}" -name '*.so*' -type f | while read -r obj; do
  collect_deps "$obj"
done

# $ORIGIN rpath so deps self-resolve from lib/
find "$OUT" -name '*.so*' -type f | while read -r so; do
  # shellcheck disable=SC2016
  patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib' "$so" 2>/dev/null || true
done

if [ -n "$(find "$OUT"/lib -maxdepth 1 \( -name 'libstdc++*' -o -name 'libgcc_s*' \) -print -quit)" ]; then
  echo "ERROR: libstdc++/libgcc_s bundled (must resolve from host)"; exit 1
fi

# literal $HERE (env.sh expands it when sourced), order-preserving dedup
LIBPATH=""; declare -A seen
for L in "$OCLLOADER" "$RUSTICL" "$LVPLIB" "$POCLLIB" "$SSLIB"; do
  d=$(reldir "$L"); [ -z "$d" ] && continue
  [ -n "${seen[$d]:-}" ] && continue; seen[$d]=1
  LIBPATH="${LIBPATH:+$LIBPATH:}\$HERE/$d"
done
for d in ocelot lib; do LIBPATH="${LIBPATH:+$LIBPATH:}\$HERE/$d"; done

# both Vulkan ICDs enumerate (lavapipe + swiftshader); polyregion selects by device name
VKDRIVERS="\$HERE/${LVPJSON#"$OUT"/}"
[ -n "$SSJSON" ] && VKDRIVERS="${VKDRIVERS}:\$HERE/${SSJSON#"$OUT"/}"

cat > "$OUT"/env.sh <<EOS
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]:-\$0}")" && pwd)"
export LD_LIBRARY_PATH="${LIBPATH}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export VK_DRIVER_FILES="${VKDRIVERS}"
export VK_ICD_FILENAMES="\$VK_DRIVER_FILES"
export OCL_ICD_VENDORS="\$HERE/icd"
export RUSTICL_ENABLE="\${RUSTICL_ENABLE:-llvmpipe}"
# rusticl/pocl are CPU OpenCL devices; polyinvoke skips those unless this is set
export POLYINVOKE_OPENCL_CPU="\${POLYINVOKE_OPENCL_CPU:-1}"
echo "emulators active: ocelot(CUDA) lavapipe+swiftshader(Vulkan) rusticl+pocl(OpenCL) via \$HERE"
EOS

echo "=== /out staged ==="
du -sh "$OUT"/lib "${OUTDIRS[@]}" 2>/dev/null
echo "lib deps bundled: $(find "$OUT"/lib -mindepth 1 | wc -l)"
echo "rusticl=$RUSTICL  icd=$(cat "$OUT"/icd/rusticl.icd)"
echo "pocl=$POCLLIB"
echo "loader=$OCLLOADER"
echo "Done"
