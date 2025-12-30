#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="sysroot"

cd "$(dirname "${BASH_SOURCE[0]}")"

for arch in x86_64 aarch64; do
  echo "Building sysroot for ${arch}..."
  podman build --pull=always --arch="$arch" -t "${IMAGE_NAME}:$arch" -f - <<'EOF'
# Based on https://github.com/AlmaLinux/container-images/blob/main/Containerfiles/8/Containerfile.micro

ARG SYSBASE=quay.io/almalinuxorg/almalinux:8
FROM ${SYSBASE} as system-build

RUN dnf install -y dnf-plugins-core && dnf config-manager --set-enabled powertools
RUN mkdir -p /mnt/sys-root; \
    dnf install --installroot /mnt/sys-root \
    gcc gcc-c++ \
    glibc-devel glibc-headers glibc-static \
    libstdc++-static libstdc++-devel libatomic-static \
    coreutils-single glibc-minimal-langpack \
    util-linux bash grep sed gawk findutils procps-ng diffutils \
    gzip zlib-devel bzip2-devel xz-devel libzstd-devel lz4-devel \
    --releasever 8 --setopt install_weak_deps=false --nodocs -y; \
    dnf --installroot /mnt/sys-root clean all;

RUN rm -rf \
    /mnt/sys-root/usr/share/locale/en* \
    /mnt/sys-root/boot /mnt/sys-root/dev/null \
    /mnt/sys-root/var/log/hawkey.log \
    /mnt/sys-root/var/cache/dnf \
    /mnt/sys-root/var/log/dnf* \
    /mnt/sys-root/var/lib/dnf \
    /mnt/sys-root/var/log/yum.* \
    /mnt/sys-root/var/lib/rpm/*

FROM scratch
COPY --from=system-build /mnt/sys-root/ /
EOF

  cid="$(podman create "${IMAGE_NAME}:$arch")"
  archive="$IMAGE_NAME-$arch.tar.gz"
  rm -rf "$archive"
  echo "Exporting to $archive..."
  podman export "$cid" | gzip > "$archive"
  podman rm "$cid" >/dev/null
done

echo "Done"
