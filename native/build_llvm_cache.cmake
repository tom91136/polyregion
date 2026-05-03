
if ("$ENV{POLYREGION_32BIT}" STREQUAL "ON")
    set(POLYREGION_32BIT ON)
else ()
    set(POLYREGION_32BIT OFF)
endif ()

if (UNIX)
    list(APPEND RUNTIME_COMPONENTS compiler-rt flang-rt libcxx libcxxabi libunwind openmp)
    #    list(APPEND RUNTIME_TARGETS cxx-headers)
elseif (WIN32)
    # needs a bootstrapping build for libcxx because cl.exe isn't support
    # see https://libcxx.llvm.org/BuildingLibcxx.html#support-for-windows
    list(APPEND RUNTIME_COMPONENTS compiler-rt flang-rt openmp)
    # nothing for RUNTIME_TARGETS
else ()
    message(FATAL_ERROR "Unsupported platform, cannot determine runtimes to build")
endif ()

if (POLYREGION_32BIT)
    list(REMOVE_ITEM RUNTIME_COMPONENTS flang-rt)
endif ()


if (UNIX)
    set(CMAKE_C_FLAGS_RELEASE "-O3 -gline-tables-only -gz -DNDEBUG" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -gline-tables-only -gz -DNDEBUG" CACHE STRING "")

    set(CMAKE_C_FLAGS_DEBUG "-O2 -g3" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g3" CACHE STRING "")
endif ()

set(LLVM_TARGETS_TO_BUILD
        AArch64
        X86
        NVPTX
        AMDGPU
        CACHE STRING "")

set(LLVM_PROJECTS
        clang
        clang-tools-extra
        lld)
if (NOT POLYREGION_32BIT)
    list(APPEND LLVM_PROJECTS flang)
endif ()
set(LLVM_ENABLE_PROJECTS ${LLVM_PROJECTS} CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES ${RUNTIME_COMPONENTS} CACHE STRING "")

if (DEFINED ENV{POLYREGION_LLVM_DYLIB} AND "$ENV{POLYREGION_LLVM_DYLIB}" STREQUAL "OFF")
    set(POLYREGION_LLVM_DYLIB OFF)
else ()
    set(POLYREGION_LLVM_DYLIB ON)
endif ()
message(STATUS "POLYREGION_LLVM_DYLIB = ${POLYREGION_LLVM_DYLIB}")

set(LLVM_DYLIB_COMPONENTS all CACHE STRING "")
set(LLVM_BUILD_LLVM_DYLIB ${POLYREGION_LLVM_DYLIB} CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB ${POLYREGION_LLVM_DYLIB} CACHE BOOL "")
set(LLVM_ENABLE_PLUGINS ON CACHE BOOL "")

set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBPFM OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_DIA_SDK OFF CACHE BOOL "")

set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_BUILD_RUNTIME ON CACHE BOOL "")
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLVM_BUILD_TESTS OFF CACHE BOOL "")

set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ABI_BREAKING_CHECKS "FORCE_ON" CACHE STRING "")

#set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR OFF CACHE BOOL "")
#
set(LIBCXX_INSTALL_MODULES ON CACHE BOOL "")
set(LIBCXX_INCLUDE_TESTS OFF CACHE BOOL "")

set(FLANG_INCLUDE_TESTS OFF CACHE BOOL "")
set(FLANG_INCLUDE_DOCS OFF CACHE BOOL "")

set(LIBOMP_INSTALL_ALIASES OFF CACHE BOOL "")
set(LIBOMP_OMPD_GDB_SUPPORT OFF CACHE BOOL "")
#
if (APPLE)
    set(LLVM_ENABLE_LIBCXX ON CACHE BOOL "")
    set(COMPILER_RT_ENABLE_IOS OFF CACHE BOOL "")
    set(COMPILER_RT_ENABLE_TVOS OFF CACHE BOOL "")
    set(COMPILER_RT_ENABLE_WATCHOS OFF CACHE BOOL "")
elseif (UNIX)
    set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
    # XXX breaks with RTTI errors when building one of the sanitisers
    # set(SANITIZER_CXX_ABI libstd++ CACHE STRING "")

    set(LIBCXX_ENABLE_STATIC_ABI_LIBRARY ON CACHE BOOL "")
    set(LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY OFF CACHE BOOL "")
    set(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY ON CACHE BOOL "")
    set(LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
    set(LIBCXX_HAS_ATOMIC_LIB OFF CACHE BOOL "")

    set(LIBCXXABI_ENABLE_STATIC_UNWINDER ON CACHE BOOL "")
    set(LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_SHARED_LIBRARY OFF CACHE BOOL "")
    set(LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY ON CACHE BOOL "")
    set(LIBCXXABI_USE_COMPILER_RT ON CACHE BOOL "")
    set(LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")
    set(LIBCXXABI_ADDITIONAL_LIBRARIES "clang_rt.builtins" CACHE STRING "") # fixes missing builtins symbols on aarch64

    set(COMPILER_RT_USE_LLVM_UNWINDER ON CACHE BOOL "")
    set(COMPILER_RT_USE_BUILTINS_LIBRARY ON CACHE BOOL "")
    set(LIBUNWIND_USE_COMPILER_RT ON CACHE BOOL "")
endif ()

set(LLVM_STATIC_LINK_CXX_STDLIB ON CACHE BOOL "")
set(COMPILER_RT_DEFAULT_TARGET_ONLY ON CACHE BOOL "")
set(COMPILER_RT_BUILD_LIBFUZZER OFF CACHE BOOL "")
set(COMPILER_RT_SANITIZERS_TO_BUILD "asan;dfsan;msan;hwasan;tsan;cfi" CACHE STRING "")

if (APPLE)
    # see https://github.com/Homebrew/homebrew-core/blob/1ffeb486e05622d8eeea0f1bffa3b7d85e6809b1/Formula/l/llvm.rb#L159
    # otherwise the default clang binary will not work unless --sysroot=$(xcrun --show-sdk-path) is added
    execute_process(
            COMMAND xcrun --show-sdk-path
            OUTPUT_VARIABLE DARWIN_SYSROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(DEFAULT_SYSROOT "${DARWIN_SYSROOT}" CACHE STRING "")
    message(STATUS "Apple: DEFAULT_SYSROOT is set to: ${DEFAULT_SYSROOT}")
    set(DEFAULT_SYSROOT "${DARWIN_SYSROOT}" CACHE STRING "")

    execute_process(
            COMMAND brew --prefix zlib
            OUTPUT_VARIABLE BREW_ZLIB_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ZLIB_ROOT "${BREW_ZLIB_PREFIX}" CACHE STRING "")

endif ()

if (POLYREGION_LLVM_DYLIB)
    set(LLVM_DISTRIBUTION_SHARED_LIBS LLVM MLIR clang-cpp)
else ()
    set(LLVM_DISTRIBUTION_SHARED_LIBS "")
endif ()

set(LLVM_DISTRIBUTION_COMPONENTS_BASE
        # Shared libraries (when dylib mode is on)
        ${LLVM_DISTRIBUTION_SHARED_LIBS}

        # Linker
        lld

        # Clang tools
        clang
        clang-tidy
        clang-resource-headers
        clang-format
        clang-offload-bundler
        clang-tblgen
        modularize

        # Clang shared
        builtins
        runtimes
        ${RUNTIME_TARGETS}

        # LLVM tools
        llc
        lli
        opt
        llvm-tblgen
        llvm-cov
        llvm-link
        llvm-dis
        llvm-profdata
        llvm-profgen
        llvm-diff
        llvm-extract

        # binutil replacements
        llvm-addr2line
        llvm-dwarfdump
        llvm-ar
        llvm-as
        llvm-cxxfilt
        llvm-install-name-tool
        llvm-nm
        llvm-objcopy
        llvm-objdump
        llvm-ranlib
        llvm-readelf
        llvm-size
        llvm-strings
        llvm-strip
        llvm-config
)

if (POLYREGION_32BIT)
    set(LLVM_DISTRIBUTION_COMPONENTS ${LLVM_DISTRIBUTION_COMPONENTS_BASE} CACHE STRING "")
else ()
    set(LLVM_DISTRIBUTION_COMPONENTS ${LLVM_DISTRIBUTION_COMPONENTS_BASE} flang CACHE STRING "")
endif ()

