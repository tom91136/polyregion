#!/usr/bin/env bash

set -eu

export CCACHE_DISABLE=1

if [ "${GITHUB_ACTIONS:-false}" = true ]; then
  echo "Running in GitHub Actions, setting TERM..."
  TERM=xterm
  export TERM=xterm
fi

ACTION=${1:?}

CMAKE_BIN="cmake"
if ! command -v cmake &>/dev/null; then
  echo "cmake not found, trying cmake3..."
  if ! command -v cmake3 &>/dev/null; then
    echo "cmake3 not found, cmake not available for build"
    exit 1
  else
    CMAKE_BIN="cmake3"
  fi
fi

OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH="${OS}-$(uname -m)"
BUILD="build-$ARCH"
echo "Using build name \"$BUILD\""

case "$ACTION" in
configure)
  SOURCE=${2:?}
  CXX="$(which clang++)"
  CC="$(which clang)"

  case "$OSTYPE" in
  darwin*) LINKER="$(which ld)" ;; # no LLD on macOS for now
  *) LINKER="$(which ld.lld)" ;;
  esac

  "$CMAKE_BIN" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DUSE_LINKER="${LINKER}" \
    -DBUILD_SHARED_LIBS=OFF \
    -P build_llvm.cmake

  "$CMAKE_BIN" "-B$BUILD" -S "$SOURCE" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DUSE_LINKER="${LINKER}" \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja
  ;;
build)
  TARGET=${2:?}
  "$CMAKE_BIN" --build "$BUILD" --target "$TARGET"
  ;;
*)
  echo "Unknown action $ACTION"
  ;;
esac
