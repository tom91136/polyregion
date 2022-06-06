#!/usr/bin/env bash

set -eu

LLVM_VERSION="14.0.4"

export_var() {
  export "$1"="$2"
  # see
  # https://docs.github.com/en/actions/reference/workflow-commands-for-github-actions#setting-an-environment-variable
  if [ "${GITHUB_ACTIONS:-false}" = true ]; then
    echo "$1=$2" >>"$GITHUB_ENV"
  fi
}

{
  yum -y install centos-release-scl epel-release
  yum -y install devtoolset-10 ninja-build git java-1.8.0-openjdk-devel
} &
{
  curl -LO "https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-project-$LLVM_VERSION.src.tar.xz"
  tar -xf "llvm-project-$LLVM_VERSION.src.tar.xz"
} & {
  curl -L "https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6-Linux-x86_64.sh" -o "cmake-3.18.sh"
  chmod +x "./cmake-3.18.sh"
  "./cmake-3.18.sh" --skip-license --include-subdir

} & wait

export_var PATH "$PWD/cmake-3.18.6-Linux-x86_64/bin/:$PATH"

echo "Prepare for SCL enable"
set +u # scl_source has unbound vars, disable check
source scl_source enable devtoolset-10 || true
set -u

cd "llvm-project-$LLVM_VERSION.src"

echo "Preparing LLVM build"

cmake3 -S llvm -B build \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja

cmake3 --build build

echo "LLVM build complete"

export_var JAVA_HOME "/usr/lib/jvm/java-1.8.0"
export_var PATH "$PWD/build/bin/:$PATH"

echo "Setup complete!"

set +eu