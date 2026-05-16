# XXX Enable CMAKE_SYSTEM_NAME only if cross compiling - flang-rt (LLVM 21.1.5) is not yet a
# cross compiler so enabling that will fail. The compiler target is set unconditionally so
# clang finds the multiarch sysroot paths (usr/lib/x86_64-linux-gnu/Scrt1.o etc.).
#set(CMAKE_SYSTEM_NAME Linux)
#set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER_TARGET x86_64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET x86_64-linux-gnu)
set(CMAKE_ASM_COMPILER_TARGET x86_64-linux-gnu)

set(CMAKE_C_COMPILER clang)
set(CMAKE_ASM_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_ASM_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif ()

set(CMAKE_C_FLAGS_INIT "-march=westmere -mtune=skylake -fPIC")
set(CMAKE_CXX_FLAGS_INIT "-march=westmere -mtune=skylake -fPIC")
set(CMAKE_ASM_FLAGS_INIT "-march=westmere -mtune=skylake -fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_SHARED_LINKER_FLAGS -fuse-ld=lld)
set(CMAKE_MODULE_LINKER_FLAGS -fuse-ld=lld)
set(CMAKE_EXE_LINKER_FLAGS -fuse-ld=lld)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# ONLY restricts find_* to the sysroot — correct when cross-building, but with no sysroot ONLY
# resolves to "nowhere" and find_package(ZLIB) etc. fails. Setting CMAKE_SYSROOT=/ as a workaround
# also strips the leading `/` from absolute rpath entries (CMake's sysroot relative-path logic),
# which then breaks the linker runpath that points at libLLVM.so at runtime.
if (DEFINED ENV{CMAKE_SYSROOT} OR CMAKE_SYSROOT)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
else ()
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif ()


set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_MAKE_BUILD_TRIPLET "--host=x86_64-linux-gnu")
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_FILE}")
if (DEFINED ENV{CMAKE_SYSROOT})
    set(CMAKE_SYSROOT "$ENV{CMAKE_SYSROOT}")
endif ()
