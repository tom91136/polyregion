# XXX Enable CMAKE_SYSTEM_NAME only if cross compiling — flang-rt (LLVM 21.1.4) is not yet a
# cross compiler so enabling that will fail. We always set the compiler target triple though,
# so clang resolves the multiarch sysroot paths (usr/lib/aarch64-linux-gnu/Scrt1.o etc.) when
# building on a runner with a different default triple than the bullseye sysroot expects.
#set(CMAKE_SYSTEM_NAME Linux)
#set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_ASM_COMPILER_TARGET aarch64-linux-gnu)

set(CMAKE_C_COMPILER clang)
set(CMAKE_ASM_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_ASM_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif ()

set(CMAKE_C_FLAGS_INIT "-fPIC")
set(CMAKE_CXX_FLAGS_INIT "-fPIC")
set(CMAKE_ASM_FLAGS_INIT "-fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_SHARED_LINKER_FLAGS -fuse-ld=lld)
set(CMAKE_MODULE_LINKER_FLAGS -fuse-ld=lld)
set(CMAKE_EXE_LINKER_FLAGS -fuse-ld=lld)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)


set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_MAKE_BUILD_TRIPLET "--host=aarch64-linux-gnu")
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_FILE}")
if (DEFINED ENV{CMAKE_SYSROOT})
    set(CMAKE_SYSROOT "$ENV{CMAKE_SYSROOT}")
endif ()
