
if (UNIX)
    # flang-rt's CMakeLists calls enable_language(Fortran), so a Fortran compiler must be on
    # PATH at LLVM-runtimes-configure time. Linux runners get gfortran via apt, macOS via brew
    # (gcc formula). Windows MSVC has no native gfortran path, so flang-rt is omitted there.
    list(APPEND RUNTIME_COMPONENTS compiler-rt flang-rt libcxx libcxxabi libunwind openmp)
    #    list(APPEND RUNTIME_TARGETS cxx-headers)
elseif (WIN32)
    # libcxx/libcxxabi/libunwind need a bootstrapping build because cl.exe is unsupported
    # (https://libcxx.llvm.org/BuildingLibcxx.html#support-for-windows). flang-rt is omitted
    # because Windows MSVC has no native Fortran path. compiler-rt is built using the
    # workflow-installed clang-cl (see windows-shared.yaml + build.cmake LLVM_NATIVE_BUILD).
    list(APPEND RUNTIME_COMPONENTS compiler-rt openmp)
    # nothing for RUNTIME_TARGETS
else ()
    message(FATAL_ERROR "Unsupported platform, cannot determine runtimes to build")
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
        SPIRV
        CACHE STRING "")

set(LLVM_ENABLE_PROJECTS
        clang
        clang-tools-extra
        lld
        flang
        CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES ${RUNTIME_COMPONENTS} CACHE STRING "")

if (DEFINED ENV{POLYREGION_LLVM_DYLIB} AND "$ENV{POLYREGION_LLVM_DYLIB}" STREQUAL "OFF")
    set(POLYREGION_LLVM_DYLIB OFF)
else ()
    set(POLYREGION_LLVM_DYLIB ON)
endif ()
message(STATUS "POLYREGION_LLVM_DYLIB = ${POLYREGION_LLVM_DYLIB}")

# CI runners (esp. macos-26 ~7 GB / 3 cores, and Linux x86_64 with thin LTO) crash mid-build
# when LLVM saturates parallelism. Allow these to be capped from the environment.
if (DEFINED ENV{POLYREGION_LLVM_PARALLEL_LINK_JOBS})
    set(LLVM_PARALLEL_LINK_JOBS "$ENV{POLYREGION_LLVM_PARALLEL_LINK_JOBS}" CACHE STRING "")
endif ()
if (DEFINED ENV{POLYREGION_LLVM_PARALLEL_COMPILE_JOBS})
    set(LLVM_PARALLEL_COMPILE_JOBS "$ENV{POLYREGION_LLVM_PARALLEL_COMPILE_JOBS}" CACHE STRING "")
endif ()
if (DEFINED ENV{POLYREGION_LLVM_PARALLEL_TABLEGEN_JOBS})
    set(LLVM_PARALLEL_TABLEGEN_JOBS "$ENV{POLYREGION_LLVM_PARALLEL_TABLEGEN_JOBS}" CACHE STRING "")
endif ()
# Flang TUs eat ~5-6GB each, so on memory-tight runners ninja's default $(nproc) parallelism
# OOMs and thrashes. Cap flang separately so the rest of LLVM still compiles at full width.
if (DEFINED ENV{POLYREGION_FLANG_PARALLEL_COMPILE_JOBS})
    set(FLANG_PARALLEL_COMPILE_JOBS "$ENV{POLYREGION_FLANG_PARALLEL_COMPILE_JOBS}" CACHE STRING "")
endif ()

set(LLVM_DYLIB_COMPONENTS all CACHE STRING "")
set(LLVM_BUILD_LLVM_DYLIB ${POLYREGION_LLVM_DYLIB} CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB ${POLYREGION_LLVM_DYLIB} CACHE BOOL "")
if (WIN32)
    # MSVC cannot use the normal dylib plugin mode, but LLVM supports static Windows tools
    # loading plugins that name their owning executable with PLUGIN_TOOL.
    set(LLVM_ENABLE_PLUGINS OFF CACHE BOOL "")
    set(LLVM_EXPORT_SYMBOLS_FOR_PLUGINS ON CACHE BOOL "")
else ()
    set(LLVM_ENABLE_PLUGINS ${POLYREGION_LLVM_DYLIB} CACHE BOOL "")
endif ()

set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")
set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "")
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

set(LLVM_DISTRIBUTION_COMPONENTS ${LLVM_DISTRIBUTION_COMPONENTS_BASE} flang CACHE STRING "")
