
set(LLVM_SRC_VERSION 22.1.5)

set(LLVM_SOURCE_URL https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_SRC_VERSION}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz)
set(LLVM_SOURCE_SHA256 7972b87b705a003ce70ab55f9f0fb495d156887cba0eb296d284731139118e2c)

set(LLVM_PATCH_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/llvm-patches
        ${CMAKE_CURRENT_LIST_DIR}/llvm-patches-${LLVM_SRC_VERSION})

# CUDA version list and min-arch list are parallel; polyc/CMakeLists.txt asserts.
set(POLYREGION_CUDA_LIBDEVICE_VERSIONS "11.8.0;12.9.2;13.2.0")
set(POLYREGION_CUDA_LIBDEVICE_MIN_ARCH "35;50;75")
set(POLYREGION_ROCM_TAG "rocm-7.2.3")
set(POLYREGION_DEVICE_LIBS_DIR "${CMAKE_CURRENT_LIST_DIR}/thirdparty/device_libs")


