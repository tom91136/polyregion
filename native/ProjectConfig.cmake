
set(LLVM_SRC_VERSION 22.1.8)
string(REGEX MATCH "^[0-9]+" LLVM_MAJOR "${LLVM_SRC_VERSION}")

set(LLVM_SOURCE_URL https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_SRC_VERSION}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz)
set(LLVM_SOURCE_SHA256 922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888)

set(LLVM_PATCH_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/llvm-patches
        ${CMAKE_CURRENT_LIST_DIR}/llvm-patches-${LLVM_SRC_VERSION})

# CUDA libdevice entries as version:min-arch pairs, split in polyc/CMakeLists.txt.
set(POLYREGION_CUDA_LIBDEVICE "11.8.0:35;12.9.2:50;13.2.0:75")
set(POLYREGION_ROCM_TAG "rocm-7.2.3")
set(POLYREGION_DEVICE_LIBS_DIR "${CMAKE_CURRENT_LIST_DIR}/thirdparty/device_libs")

set(POLYREGION_DIST_CHECK_CUDA_ARCH "sm_70")
set(POLYREGION_DIST_CHECK_ROCM_ARCH "gfx900")


