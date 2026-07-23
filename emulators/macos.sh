#!/usr/bin/env bash
set -euo pipefail

# macos.sh [out]  build+stage  |  --collect re-stage+smoke  |  --check smoke

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${EMU_WORK:-$HOME/emu-build}"
MESA_REF="${MESA_REF:-mesa-26.1.1}"
POCL_REF="${POCL_REF:-v7.1}"
XLAT_REF="${XLAT_REF:-llvm_release_210}"
SWIFTSHADER_REF="${SWIFTSHADER_REF:-2843cbcc714fe111e1083127c048a18002bc10ed}"
BREW="$(command -v brew || echo /opt/homebrew/bin/brew)"
PREFIX="$("$BREW" --prefix)"
LLVM21="$PREFIX/opt/llvm@21"
export HOMEBREW_NO_AUTO_UPDATE=1
# keg-only bison/flex/llvm + the venv python (>=3.10, carries mako for mesa codegen)
export PATH="$PREFIX/bin:$PREFIX/opt/bison/bin:$PREFIX/opt/flex/bin:$PREFIX/opt/llvm/bin:$WORK/venv/bin:$PATH"

deps() {
  "$BREW" install meson ninja bison flex glslang vulkan-loader vulkan-headers \
    opencl-headers llvm llvm@21 hwloc spirv-tools expat zstd
  mkdir -p "$WORK"
  [ -x "$WORK/venv/bin/python" ] || "$PREFIX/bin/python3" -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q --upgrade pip
  "$WORK/venv/bin/pip" install -q meson mako pyyaml packaging
}

# llvm-spirv matching llvm@21, for pocl's SPIR-V
xlat() {
  [ -d "$WORK/spirv-xlat/.git" ] || git clone -b "$XLAT_REF" --depth 1 \
    https://github.com/KhronosGroup/SPIRV-LLVM-Translator "$WORK/spirv-xlat"
  rm -rf "$WORK/build-xlat" "$WORK/xlat-prefix"
  cmake -S "$WORK/spirv-xlat" -B "$WORK/build-xlat" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$LLVM21/lib/cmake/llvm" -DCMAKE_INSTALL_PREFIX="$WORK/xlat-prefix"
  ninja -C "$WORK/build-xlat" install
}

pocl() {
  [ -d "$WORK/pocl-src/.git" ] || git clone -b "$POCL_REF" --depth 1 \
    https://github.com/pocl/pocl "$WORK/pocl-src"
  rm -rf "$WORK/build-pocl" "$WORK/pocl-prefix"
  # LLC_HOST_CPU pinned: pocl can't auto-detect the CPU on some hosts (GHA runners). arm64 macOS is always
  # Apple Silicon so apple-m1 is the safe floor; x86_64 takes the generic x86-64 baseline
  local llc_cpu; llc_cpu="$([ "$(uname -m)" = arm64 ] && echo apple-m1 || echo x86-64)"
  # static+hidden LLVM keeps pocl's copy from colliding with libLLVMpolyregion; cmake-scoped LDFLAGS keeps -L/-lzstd out of pocl's runtime kernel-link
  printf '%s\n' '*4llvm*' '*LLVM*' '_llvm_*' > "$WORK/llvm-hide.sym"
  LDFLAGS="-L$PREFIX/lib -Wl,-unexported_symbols_list,$WORK/llvm-hide.sym${LDFLAGS:+ $LDFLAGS}" \
  cmake -S "$WORK/pocl-src" -B "$WORK/build-pocl" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DWITH_LLVM_CONFIG="$LLVM21/bin/llvm-config" -DLLC_HOST_CPU="$llc_cpu" -DSTATIC_LLVM=ON \
    -DHAVE_WARN_INCOMPATIBLE_POINTER_TYPES=OFF \
    -DENABLE_ICD=ON -DENABLE_HOST_CPU_DEVICES=ON -DENABLE_LOADABLE_DRIVERS=OFF \
    -DENABLE_SPIRV=ON -DLLVM_SPIRV="$WORK/xlat-prefix/bin/llvm-spirv" \
    -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$WORK/pocl-prefix"
  ninja -C "$WORK/build-pocl" install
}

mesa() {
  [ -d "$WORK/mesa/.git" ] || git clone -b "$MESA_REF" --depth 1 \
    https://gitlab.freedesktop.org/mesa/mesa "$WORK/mesa"
  export LIBRARY_PATH="$PREFIX/lib"
  export PKG_CONFIG_PATH="$PREFIX/opt/expat/lib/pkgconfig:$PREFIX/opt/zlib/lib/pkgconfig"
  rm -rf "$WORK/build-mesa" "$WORK/mesa-prefix"
  # static+hidden LLVM keeps lavapipe's copy from colliding with libLLVMpolyregion; LDFLAGS scoped to meson so it can't leak into the swiftshader build
  printf '%s\n' '*4llvm*' '*LLVM*' '_llvm_*' > "$WORK/llvm-hide.sym"
  LDFLAGS="-Wl,-unexported_symbols_list,$WORK/llvm-hide.sym${LDFLAGS:+ $LDFLAGS}" \
  meson setup "$WORK/mesa" "$WORK/build-mesa" --prefix="$WORK/mesa-prefix" --prefer-static \
    -Dbuildtype=release -Dvulkan-drivers=swrast -Dgallium-drivers=llvmpipe \
    -Dllvm=enabled -Dshared-llvm=disabled \
    -Dopengl=false -Dgles1=disabled -Dgles2=disabled -Degl=disabled \
    -Dgbm=disabled -Dglx=disabled -Dplatforms= -Dvideo-codecs= -Dgallium-va=disabled
  ninja -C "$WORK/build-mesa" install
}

swiftshader() {
  [ -d "$WORK/swiftshader/.git" ] || git clone --filter=blob:none https://github.com/google/swiftshader "$WORK/swiftshader"
  git -C "$WORK/swiftshader" checkout -q "$SWIFTSHADER_REF"
  rm -rf "$WORK/swiftshader/build-ss"
  cmake -S "$WORK/swiftshader" -B "$WORK/swiftshader/build-ss" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSWIFTSHADER_BUILD_TESTS=OFF -DSWIFTSHADER_WARNINGS_AS_ERRORS=OFF
  ninja -C "$WORK/swiftshader/build-ss" vk_swiftshader
}

# otool dep ref -> source file (abs brew lib, or @rpath via the referrer's rpaths)
dep_source() {
  local ref="$1" referrer="$2" rp name
  case "$ref" in
    /usr/lib/*|/System/*) return 1;;
    /*) [ -e "$ref" ] && { echo "$ref"; return 0; } || return 1;;
    @rpath/*) name="${ref#@rpath/}"
      while IFS= read -r rp; do
        case "$rp" in @*) continue;; esac
        [ -e "$rp/$name" ] && { echo "$rp/$name"; return 0; }
      done < <(otool -l "$referrer" | awk '/LC_RPATH/{f=1;next} f&&/ path /{print $2;f=0}')
      return 1;;
    *) return 1;;
  esac
}

# copy non-system deps into BUNDLE/lib, rewrite refs to @loader_path/../lib, recurse
relocate() {
  local f="$1" ref base src
  while IFS= read -r ref; do
    case "$ref" in /usr/lib/*|/System/*|@loader_path/*|@executable_path/*) continue;; esac
    base="$(basename "$ref")"
    if [ ! -e "$BUNDLE/lib/$base" ]; then
      src="$(dep_source "$ref" "$f")" || continue
      cp -f "$src" "$BUNDLE/lib/$base"; chmod u+w "$BUNDLE/lib/$base"
      install_name_tool -id "@loader_path/../lib/$base" "$BUNDLE/lib/$base" 2>/dev/null
      relocate "$BUNDLE/lib/$base"
    fi
    install_name_tool -change "$ref" "@loader_path/../lib/$base" "$f" 2>/dev/null
  done < <(otool -L "$f" | tail -n +2 | awk '{print $1}')
}

collect() {
  local OUT="$1"; BUNDLE="$OUT"
  rm -rf "$OUT"; mkdir -p "$OUT/loader" "$OUT/lib" "$OUT/share/vulkan/icd.d"
  # loader/ on DYLD (dlopened by name); lib/ off it via @loader_path so bundled libLLVM can't shadow host clang
  cp "$WORK/mesa-prefix/lib/libvulkan_lvp.dylib" "$OUT/lib/"
  cp "$WORK/mesa-prefix/share/vulkan/icd.d/"lvp_icd.*.json "$OUT/share/vulkan/icd.d/"
  cp -RP "$("$BREW" --prefix vulkan-loader)/lib/"libvulkan.*dylib "$OUT/loader/"
  cp -RP "$WORK/pocl-prefix/lib/"libOpenCL.*dylib "$OUT/loader/"
  cp -R "$WORK/pocl-prefix/share/pocl" "$OUT/share/pocl"  # pocl finds bitcode/headers at ../share/pocl
  cp "$WORK/swiftshader/build-ss/Darwin/libvk_swiftshader.dylib" "$OUT/lib/"
  cp "$WORK/swiftshader/build-ss/Darwin/vk_swiftshader_icd.json" "$OUT/share/vulkan/icd.d/"
  chmod u+w "$OUT"/loader/*.dylib "$OUT"/lib/*.dylib
  local root real
  for root in "$OUT"/loader/libvulkan.dylib "$OUT"/loader/libOpenCL.dylib \
              "$OUT"/lib/libvulkan_lvp.dylib "$OUT"/lib/libvk_swiftshader.dylib; do
    real="$(python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$root")"
    install_name_tool -id "@loader_path/$(basename "$real")" "$real" 2>/dev/null
    relocate "$real"
  done
  # install_name_tool breaks the signature; arm64 SIGKILLs unsigned mach-o
  find "$OUT/loader" "$OUT/lib" -type f -name '*.dylib' -exec codesign -f -s - {} \; 2>/dev/null
  # point each ICD manifest at its driver in ../../../lib
  python3 - "$OUT/share/vulkan/icd.d" <<'PY'
import json,sys,glob,os
for p in glob.glob(sys.argv[1]+"/*.json"):
    d=json.load(open(p)); d["ICD"]["library_path"]="../../../lib/"+os.path.basename(d["ICD"]["library_path"])
    json.dump(d,open(p,"w"),indent=4)
PY
  cat > "$OUT/env.sh" <<'EOS'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export DYLD_LIBRARY_PATH="$HERE/loader${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export VK_DRIVER_FILES="$(ls "$HERE"/share/vulkan/icd.d/*.json | paste -sd: -)"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export POLYINVOKE_OPENCL_LIB="$(ls "$HERE"/loader/libOpenCL*.dylib 2>/dev/null | head -1)"
[ -d "$HERE/share/OpenCL/vendors" ] && export OCL_ICD_VENDORS="$HERE/share/OpenCL/vendors"
export POLYINVOKE_OPENCL_CPU="${POLYINVOKE_OPENCL_CPU:-1}"
export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null)}"  # pocl links kernels with clang -> needs the SDK
export LIBRARY_PATH="$SDKROOT/usr/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"  # pocl kernel link (ld -lSystem) on x86_64 mac
echo "emulators active: lavapipe+swiftshader(Vulkan) pocl(OpenCL+SPIR-V) via $HERE"
EOS
  echo "=== bundle staged at $OUT ==="
  du -sh "$OUT" "$OUT/loader" "$OUT/lib"
  echo "loader entries: $(find "$OUT/loader" -type f | wc -l)  lib entries: $(find "$OUT/lib" -type f | wc -l)"
}

check() {
  local OUT="$1"
  [ -f "$OUT/env.sh" ] || { echo "no bundle at $OUT (run emulators/macos.sh first)"; exit 2; }
  local B; B="$(mktemp -d)"; trap 'rm -rf "$B"' RETURN
  # compile with the bundle inactive (else it shadows the compiler's LLVM); llvm@21 clang emits spirv64 for vecadd
  clang++ -O2 -std=c++17 "$DIR/vecadd.vk.cpp" -o "$B/vk" -I"$("$BREW" --prefix vulkan-headers)/include" \
    -L"$OUT/loader" -lvulkan -Wl,-rpath,"$OUT/loader"
  clang++ -O2 -std=c++17 "$DIR/vecadd.cl.cpp" -o "$B/cl" \
    -I"$("$BREW" --prefix opencl-headers)/include" -L"$OUT/loader" -lOpenCL -Wl,-rpath,"$OUT/loader"
  ( # shellcheck disable=SC1090
    source "$OUT/env.sh"
    export PATH="$LLVM21/bin:$PATH"; rc=0
    echo "== Vulkan (lavapipe + swiftshader) =="; "$B/vk" || rc=1
    echo "== OpenCL (pocl) =="; "$B/cl" || rc=1
    [ $rc -eq 0 ] && echo "SMOKE OK" || echo "SMOKE FAILED"
    exit $rc )
}

case "${1:-}" in
  --check)   check "${2:-$DIR/out}";;
  --collect) collect "${2:-$DIR/out}"; check "${2:-$DIR/out}";;
  *) OUT="${1:-$DIR/out}"; deps; xlat; pocl; mesa; swiftshader; collect "$OUT"; check "$OUT";;
esac
