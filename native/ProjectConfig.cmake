
set(LLVM_SRC_VERSION 22.1.5)

set(LLVM_SOURCE_URL https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_SRC_VERSION}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz)
set(LLVM_SOURCE_SHA256 7972b87b705a003ce70ab55f9f0fb495d156887cba0eb296d284731139118e2c)

set(LLVM_PATCH_DIR ${CMAKE_CURRENT_LIST_DIR}/llvm-patches-${LLVM_SRC_VERSION})


