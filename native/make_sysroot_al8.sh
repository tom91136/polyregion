#!/usr/bin/env bash
# Build an AlmaLinux 8 sysroot tarball for CMAKE_SYSROOT.
# Usage: ./make_sysroot_al8.sh [x86_64|aarch64 ...]

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

OUT_DIR="${OUT_DIR:-out/sysroot-build/al8}"
mkdir -p "$OUT_DIR"

if [ -z "${OCI:-}" ]; then
  if   command -v docker >/dev/null 2>&1; then OCI=docker
  elif command -v podman >/dev/null 2>&1; then OCI=podman
  else echo "need docker or podman on PATH" >&2; exit 1
  fi
fi

archs=("${@:-x86_64}")

for arch in "${archs[@]}"; do
  case "$arch" in
    x86_64|aarch64) ;;
    *) echo "unsupported arch: $arch (x86_64/aarch64 only)" >&2; exit 1 ;;
  esac

  echo "==> building al8 sysroot for $arch"
  image="polyregion-sysroot-al8:$arch"

  "$OCI" build --pull=always --arch="$arch" -t "$image" -f - <<'EOF'
ARG SYSBASE=quay.io/almalinuxorg/almalinux:8
FROM ${SYSBASE} AS staging

RUN dnf install -y dnf-plugins-core && dnf config-manager --set-enabled powertools

RUN mkdir -p /mnt/sys-root \
 && dnf --installroot /mnt/sys-root --releasever 8 --setopt install_weak_deps=false \
        --nodocs -y install \
        glibc glibc-devel glibc-headers glibc-static \
        libgcc libstdc++ libstdc++-devel libstdc++-static \
        libatomic libatomic-static \
        libgfortran libquadmath \
        kernel-headers \
        gcc gcc-c++ \
        gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ \
        gcc-toolset-12-libstdc++-devel gcc-toolset-12-libatomic-devel \
 && dnf --installroot /mnt/sys-root clean all

RUN rm -rf \
    /mnt/sys-root/usr/share/{locale,doc,man,info} \
    /mnt/sys-root/boot \
    /mnt/sys-root/var/{cache,log,lib}/{dnf,rpm,yum.*}

RUN cd /mnt/sys-root/usr/lib64 \
 && ln -sf libstdc++.so.6   libstdc++.so   \
 && ln -sf libgcc_s.so.1    libgcc_s.so    \
 && ln -sf libquadmath.so.0 libquadmath.so \
 && ln -sf libgfortran.so.5 libgfortran.so \
 && ln -sf libatomic.so.1   libatomic.so   \
 && cd /mnt/sys-root/usr/lib/gcc/x86_64-redhat-linux/8 \
 && ln -sf ../../../../lib64/libstdc++.so.6 libstdc++.so

RUN ln -sfn x86_64-redhat-linux /mnt/sys-root/usr/lib/gcc/x86_64-linux-gnu \
 && ln -sfn x86_64-redhat-linux /mnt/sys-root/usr/include/c++/8/x86_64-linux-gnu

FROM scratch
COPY --from=staging /mnt/sys-root/ /
EOF

  cid="$("$OCI" create "$image")"
  tarball="$OUT_DIR/sysroot-al8-${arch}.tar.xz"
  echo "==> exporting -> $tarball"
  "$OCI" export "$cid" | xz -T 0 > "$tarball"
  "$OCI" rm "$cid" >/dev/null
  echo "==> done: $(du -h "$tarball" | cut -f1) $tarball"
done
