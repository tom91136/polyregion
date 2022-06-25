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

CMAKE_VERSION="3.22.4"
CMAKE_NAME="cmake-$CMAKE_VERSION-linux-$(uname -m)"

setup_env() {

  export_var PATH "$PWD/$CMAKE_NAME/bin/:$PATH"
  export_var PATH "$PWD/llvm-project-$LLVM_VERSION.src/build/bin/:$PATH"
  export_var JAVA_HOME "/usr/lib/jvm/java-1.8.0"
  echo "Prepare for SCL enable"

  set +u # scl_source has unbound vars, disable check
  source scl_source enable devtoolset-10 || true
  set -u
}

setup_build() {
  yum -y install centos-release-scl epel-release
  yum -y install devtoolset-10 ninja-build git java-1.8.0-openjdk-devel python3 libatomic-static file
}

ACTION=${1:?}
case "$ACTION" in
setup-build)
  setup_build
  echo "Build setup complete!"
  ;;
setup-llvm)
  setup_env

  echo "Preparing CMake"
  curl -L "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/$CMAKE_NAME.sh" -o "cmake-$CMAKE_VERSION.sh"

  chmod +x "./cmake-$CMAKE_VERSION.sh"
  "./cmake-$CMAKE_VERSION.sh" --skip-license --include-subdir
  rm -rf "./cmake-$CMAKE_VERSION.sh"

  echo "Preparing LLVM build"
  curl -LO "https://github.com/llvm/llvm-project/releases/download/llvmorg-$LLVM_VERSION/llvm-project-$LLVM_VERSION.src.tar.xz"
  tar -xf "llvm-project-$LLVM_VERSION.src.tar.xz"

  (
    cd "llvm-project-$LLVM_VERSION.src"
    cmake -S llvm -B build \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
      -DCMAKE_BUILD_TYPE=Release \
      -GNinja
  )

  (
    cd "llvm-project-$LLVM_VERSION.src"
    cmake --build build
  )

  rm -rf "llvm-project-$LLVM_VERSION.src.tar.xz"

  echo "LLVM build complete"
  ;;
env)
  setup_env
  echo "PATH=$PATH"
  echo "JAVA_HOME=$JAVA_HOME"
  echo "ENV setup complete!"
  ;;
*)
  echo "Unknown action $ACTION"
  ;;
esac

set +eu
