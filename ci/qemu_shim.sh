#!/usr/bin/env bash
# Install clang/flang wrappers injecting --sysroot / -fuse-ld=lld under POLYREGION_SYSROOT
set -euo pipefail

arch="${1:?arch (e.g. riscv64) required}"
dist="${2:?dist dir required}"
sysroot="${3:?sysroot dir required}"

case "$arch" in
  riscv64) triple=riscv64-linux-gnu ;;
  aarch64) triple=aarch64-linux-gnu ;;
  ppc64le) triple=powerpc64le-linux-gnu ;;
  arm)     triple=arm-linux-gnueabihf ;;
  *) echo "qemu_shim: unsupported arch: $arch (add a case)" >&2; exit 1 ;;
esac

llvm_major=$(basename "$(ls "$dist"/bin/clang-[0-9]* 2>/dev/null | head -1)" | sed 's/^clang-//')
[ -n "$llvm_major" ] || { echo "qemu_shim: no clang-N in $dist/bin" >&2; exit 1; }
# XXX aarch32 dist omits flang
has_flang=1
[ -x "$dist/bin/flang-$llvm_major" ] || [ -x "$dist/bin/flang-$llvm_major.real" ] || has_flang=0

# AL8: gcc-toolset-N (C++17 libstdc++); Ubuntu: stock gcc path only
gcc_dir=$(ls -d "$sysroot"/opt/rh/gcc-toolset-*/root/usr/lib/gcc/*-redhat-linux/[0-9]*/ 2>/dev/null | sort -Vr | head -1 || true)
if [ -z "$gcc_dir" ]; then
  gcc_dir=$(ls -d "$sysroot/usr/lib/gcc/$triple"/*/ 2>/dev/null | sort -Vr | head -1 || true)
fi
gcc_dir="${gcc_dir%/}"
[ -n "$gcc_dir" ] || { echo "qemu_shim: no gcc install dir under $sysroot" >&2; exit 1; }

# Stash flang-N as .real; leave flang-N as a symlink so FlangTargets.cmake still resolves it.
if [ "$has_flang" = 1 ] && [ -x "$dist/bin/flang-$llvm_major" ] && [ ! -e "$dist/bin/flang-$llvm_major.real" ]; then
  mv "$dist/bin/flang-$llvm_major" "$dist/bin/flang-$llvm_major.real"
  ln -sfn "flang-$llvm_major.real" "$dist/bin/flang-$llvm_major"
fi
rm -f "$dist/bin/clang++" "$dist/bin/flang-new" "$dist/bin/flang"

# XXX binfmt-misc F flag drops exec -a; symlink argv[0] preserves clang++/flang-new mode
mkdir -p "$dist/bin/.shim"
ln -sfn "../clang-$llvm_major"       "$dist/bin/.shim/clang++"
ln -sfn "../clang-$llvm_major"       "$dist/bin/.shim/clang"
if [ "$has_flang" = 1 ]; then
  ln -sfn "../flang-$llvm_major.real"  "$dist/bin/.shim/flang-new"
fi

# XXX cmake testCCompiler.c uses clang not clang++; also needs -fuse-ld=lld
for _drv in clang clang++; do
  cat > "$dist/bin/$_drv" <<SHIM
#!/usr/bin/env bash
DIR="\$(dirname "\$0")"
inject=1
for a in "\$@"; do [ "\$a" = "-cc1" ] && inject=0 && break; done
if [ -n "\${POLYREGION_SYSROOT-}" ] && [ "\$inject" = 1 ]; then
  set -- --sysroot="\$POLYREGION_SYSROOT" --gcc-install-dir="$gcc_dir" \\
    -B"\$DIR" -fuse-ld=lld "\$@"
fi
exec "\$DIR/.shim/$_drv" "\$@"
SHIM
  chmod +x "$dist/bin/$_drv"
done

# XXX flang self-invokes bin/flang -fc1; skip inject, rejects --gcc-install-dir / -B
if [ "$has_flang" = 1 ]; then
  cat > "$dist/bin/flang-new" <<SHIM
#!/usr/bin/env bash
DIR="\$(dirname "\$0")"
inject=1
for a in "\$@"; do [ "\$a" = "-fc1" ] && inject=0 && break; done
if [ -n "\${POLYREGION_SYSROOT-}" ] && [ "\$inject" = 1 ]; then
  set -- --sysroot="\$POLYREGION_SYSROOT" -fuse-ld=lld "\$@"
fi
exec "\$DIR/.shim/flang-new" "\$@"
SHIM
  chmod +x "$dist/bin/flang-new"
  ln -sf flang-new "$dist/bin/flang"
fi

echo "qemu_shim: installed clang++$([ "$has_flang" = 1 ] && echo /flang-new) wrappers for $arch in $dist/bin (gcc-install-dir=$gcc_dir)"
