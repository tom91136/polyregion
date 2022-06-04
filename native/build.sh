#!/usr/bin/env bash

set -eu

export CCACHE_DISABLE=1

if [ "${GITHUB_ACTIONS:-false}" = true ]; then
  echo "Running in GitHub Actions, setting TERM..."
  TERM=xterm
  export TERM=xterm
fi

ACTION=${1:?}

OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH="${OS}-$(uname -m)"
BUILD="build-$ARCH"
echo "Using build name \"$BUILD\""

case "$ACTION" in
configure)
  CXX="$(which clang++)"
  CC="$(which clang)"
  LINKER="$(which ld.lld)"
  cmake "-B$BUILD" -H. \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DUSE_LINKER="${LINKER}" \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja
  ;;
build)
  TARGET=${2:?}
  cmake --build "$BUILD" --target "$TARGET" -j "$(nproc)"
  ;;
*)
  echo "Unknown action $ACTION"
  ;;
esac
