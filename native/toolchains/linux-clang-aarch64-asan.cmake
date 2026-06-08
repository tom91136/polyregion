# XXX Enable CMAKE_SYSTEM_NAME only if cross compiling — flang-rt (LLVM 21.1.4) is not yet a
# cross compiler so enabling that will fail. We always set the compiler target triple though,
# so clang resolves the multiarch sysroot paths (usr/lib/aarch64-linux-gnu/Scrt1.o etc.) when
# building on a runner with a different default triple than the bullseye sysroot expects.
#set(CMAKE_SYSTEM_NAME Linux)
#set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_ASM_COMPILER_TARGET aarch64-linux-gnu)

# XXX Use the project's bundled clang dist so the asan runtime matches what we shipped.
# Run `cmake -P build.cmake -DACTION=DIST` first to populate the dist dir.
set(_POLYREGION_BUILT_CLANG_DIR "${CMAKE_CURRENT_LIST_DIR}/../out/polyregion-Release-aarch64-dylib-dist/bin")
if (NOT EXISTS "${_POLYREGION_BUILT_CLANG_DIR}/clang")
    message(FATAL_ERROR "asan toolchain expects a built polyregion clang at ${_POLYREGION_BUILT_CLANG_DIR}; build the dylib dist first")
endif ()

set(CMAKE_C_COMPILER "${_POLYREGION_BUILT_CLANG_DIR}/clang")
set(CMAKE_ASM_COMPILER "${_POLYREGION_BUILT_CLANG_DIR}/clang")
set(CMAKE_CXX_COMPILER "${_POLYREGION_BUILT_CLANG_DIR}/clang++")

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

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/sysroot_gcc_dir.cmake")
if (POLYREGION_SYSROOT_GCC_INSTALL_DIR)
    string(APPEND CMAKE_C_FLAGS_INIT   " --gcc-install-dir=${POLYREGION_SYSROOT_GCC_INSTALL_DIR}")
    string(APPEND CMAKE_CXX_FLAGS_INIT " --gcc-install-dir=${POLYREGION_SYSROOT_GCC_INSTALL_DIR}")
    string(APPEND CMAKE_ASM_FLAGS_INIT " --gcc-install-dir=${POLYREGION_SYSROOT_GCC_INSTALL_DIR}")
endif ()

set(CMAKE_SHARED_LINKER_FLAGS -fuse-ld=lld)
set(CMAKE_MODULE_LINKER_FLAGS -fuse-ld=lld)
set(CMAKE_EXE_LINKER_FLAGS -fuse-ld=lld)

# XXX the dist's own ld.lld is built without zlib (the AL8 sysroot has no zlib-devel) so it
# cannot read the dist's -gz compressed archives; link with the system lld instead.
# Plain set shadows the -DUSE_LINKER=lld cache entry build.cmake passes.
find_program(_SYSTEM_LLD ld.lld REQUIRED)
set(USE_LINKER "${_SYSTEM_LLD}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)


set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_BUILD_TYPE release)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_MAKE_BUILD_TRIPLET "--host=aarch64-linux-gnu")
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_FILE}")
if (DEFINED ENV{CMAKE_SYSROOT})
    set(CMAKE_SYSROOT "$ENV{CMAKE_SYSROOT}")
endif ()

# ASan+UBSan for offload-runtime debugging; vptr needs RTTI (project builds -fno-rtti) and
# -shared-libsan keeps one dynamic runtime so uninstrumented hosts can LD_PRELOAD it
set(_SAN_BASE "-fsanitize=address,undefined -fno-sanitize=vptr")
set(_SAN_FLAGS "${_SAN_BASE} -fno-omit-frame-pointer -g")
set(_SAN_LINK_FLAGS "${_SAN_BASE} -shared-libsan -frtlib-add-rpath")
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} ${_SAN_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} ${_SAN_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${CMAKE_ASM_FLAGS_INIT} ${_SAN_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_SAN_LINK_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_SAN_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_SAN_LINK_FLAGS}")
