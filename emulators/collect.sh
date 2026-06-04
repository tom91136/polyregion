#!/usr/bin/env bash
set -euo pipefail

OUT=/out
mkdir -p "$OUT"/lib

cp -a /opt/lavapipe "$OUT"/lavapipe
cp -a /opt/pocl     "$OUT"/pocl
cp -a /opt/vkloader "$OUT"/vkloader
cp -a /opt/ocelot   "$OUT"/ocelot
ln -sf libgpuocelot.so "$OUT"/ocelot/libcuda.so.1
ln -sf libgpuocelot.so "$OUT"/ocelot/libcuda.so

LVPJSON=$(find "$OUT"/lavapipe -name 'lvp_icd*.json' | head -1)
# XXX PoCL's embedded libOpenCL is .so.2; create libOpenCL.so.1
POCLDIR=$(dirname "$(find "$OUT"/pocl -name 'libOpenCL.so*' | head -1)")
[ -n "$POCLDIR" ] && ln -sf libOpenCL.so.2 "$POCLDIR/libOpenCL.so.1"

# XXX ICD library_path -> bare soname, resolved via LD_LIBRARY_PATH
sed -i 's|"library_path":[^,]*|"library_path": "libvulkan_lvp.so"|' "$LVPJSON"

hostlibs='ld-linux|libc\.so|libm\.so|libdl\.so|libpthread|librt\.so|libresolv|libnsl|ld64|libstdc\+\+|libgcc_s'
collect_deps() {
  ldd "$1" 2>/dev/null | awk '/=>/ {print $3}' | while read -r so; do
    [ -e "$so" ] || continue
    basename "$so" | grep -qE "$hostlibs" && continue
    case "$so" in "$OUT"/*) continue;; esac
    cp -aL "$so" "$OUT"/lib/ 2>/dev/null || true
  done
}
find "$OUT"/lavapipe "$OUT"/pocl "$OUT"/vkloader "$OUT"/ocelot -name '*.so*' -type f | while read -r obj; do
  collect_deps "$obj"
done

# XXX $ORIGIN RPATH so deps self-resolve from lib/ without putting common libs on LD_LIBRARY_PATH
find "$OUT" -name '*.so*' -type f | while read -r so; do
  patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib' "$so" 2>/dev/null || true
done

if ls "$OUT"/lib | grep -qE 'libstdc\+\+|libgcc_s'; then
  echo "ERROR: libstdc++/libgcc_s bundled (must resolve from host)"; exit 1
fi

# ICD path resolved at build time; bake it relative so env.sh doesn't re-glob on every source
cat > "$OUT"/env.sh <<EOS
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]:-\$0}")" && pwd)"
export LD_LIBRARY_PATH="\$HERE/ocelot:\$HERE/pocl/lib64:\$HERE/lavapipe/lib64:\$HERE/vkloader/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export VK_DRIVER_FILES="\$HERE/${LVPJSON#"$OUT"/}"
export VK_ICD_FILENAMES="\$VK_DRIVER_FILES"
echo "emulators active: ocelot(CUDA) lavapipe(Vulkan) pocl(OpenCL) via \$HERE"
EOS

echo "=== /out staged ==="
du -sh "$OUT"/lib "$OUT"/lavapipe "$OUT"/pocl "$OUT"/vkloader "$OUT"/ocelot 2>/dev/null
echo "lib deps bundled: $(ls "$OUT"/lib | wc -l)"
echo "Done"
