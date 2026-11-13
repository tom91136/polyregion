#!/usr/bin/env bash
# usage: build.sh <arch> <podman|docker>; writes out/sysroot-<base>-<arch>.tar.xz
set -euo pipefail
arch=$1
container=$2
cd "$(dirname -- "${BASH_SOURCE[0]}")"
case "$arch" in
  x86_64)  base=al8;      dockerfile=Dockerfile;        plat=linux/amd64;   args=(--build-arg RHEL_TRIPLE=x86_64-redhat-linux  --build-arg GNU_TRIPLE=x86_64-linux-gnu       --build-arg "EXTRA_PKGS=libquadmath zlib-devel zlib-static") ;;
  aarch64) base=al8;      dockerfile=Dockerfile;        plat=linux/arm64;   args=(--build-arg RHEL_TRIPLE=aarch64-redhat-linux --build-arg GNU_TRIPLE=aarch64-linux-gnu       --build-arg "EXTRA_PKGS=zlib-devel zlib-static") ;;
  ppc64le) base=al8;      dockerfile=Dockerfile;        plat=linux/ppc64le; args=(--build-arg RHEL_TRIPLE=ppc64le-redhat-linux  --build-arg GNU_TRIPLE=powerpc64le-linux-gnu --build-arg "EXTRA_PKGS=zlib-devel zlib-static") ;;
  riscv64) base=ubuntu20; dockerfile=Dockerfile.ubuntu; plat=linux/riscv64; args=(--build-arg SYSBASE=docker.io/riscv64/ubuntu:20.04 --build-arg GCC_MAJOR=10 --build-arg EXTRA_PKGS=zlib1g-dev) ;;
  arm)     base=ubuntu20; dockerfile=Dockerfile.ubuntu; plat=linux/arm/v7;  args=(--build-arg SYSBASE=docker.io/arm32v7/ubuntu:20.04 --build-arg GCC_MAJOR=10 --build-arg EXTRA_PKGS=zlib1g-dev) ;;
  *) echo "unsupported arch: $arch (x86_64/aarch64/ppc64le/riscv64/arm)" >&2; exit 1 ;;
esac
flags=()  # cross-arch flag only when target != host
if [ "$arch" != "$(uname -m)" ]; then
  case "$container" in podman) flags=(--arch="$arch");; docker) flags=(--platform="$plat");; esac
fi
mkdir -p out
img=polyregion-sysroot-${base}:$arch
$container build --pull ${flags[@]+"${flags[@]}"} -f "$dockerfile" --build-context ci=../ci ${args[@]+"${args[@]}"} -t "$img" .
cid=$($container create "$img")
$container export "$cid" | xz -T0 > "out/sysroot-${base}-$arch.tar.xz"
$container rm "$cid" >/dev/null
