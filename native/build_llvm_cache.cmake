
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

    include(${CMAKE_CURRENT_LIST_DIR}/cmake/sysroot_gcc_dir.cmake)
    if (POLYREGION_SYSROOT_GCC_INSTALL_DIR)
        set(_extra " --gcc-install-dir=${POLYREGION_SYSROOT_GCC_INSTALL_DIR}")
        string(APPEND CMAKE_C_FLAGS_RELEASE   "${_extra}")
        string(APPEND CMAKE_CXX_FLAGS_RELEASE "${_extra}")
        string(APPEND CMAKE_C_FLAGS_DEBUG     "${_extra}")
        string(APPEND CMAKE_CXX_FLAGS_DEBUG   "${_extra}")
        set(CMAKE_C_FLAGS   "${_extra}" CACHE STRING "")
        set(CMAKE_CXX_FLAGS "${_extra}" CACHE STRING "")
        set(CMAKE_ASM_FLAGS "${_extra}" CACHE STRING "")
    endif ()

    # XXX shm_open is in librt on glibc <2.34; thin-LTO drops the implicit dependency LLVM's OrcJIT introduces,
    # leaving libLLVM.so unlinked against -lrt. --as-needed: librt is only retained when its symbols are referenced;
    # cmake try-compile (which doesn't use rt) skips it cleanly so its pthread@GLIBC_PRIVATE back-refs don't trip the link.
    if (NOT APPLE)
        set(_polyregion_rt "-Wl,--push-state,--as-needed,-lrt,--pop-state")
        set(CMAKE_SHARED_LINKER_FLAGS "${_polyregion_rt}" CACHE STRING "")
        set(CMAKE_EXE_LINKER_FLAGS    "${_polyregion_rt}" CACHE STRING "")
        set(CMAKE_MODULE_LINKER_FLAGS "${_polyregion_rt}" CACHE STRING "")
    endif ()
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
# Namespace the bundled libLLVM so it can never collide with a system libLLVM that a dlopen'd driver
# (e.g. a Mesa Vulkan ICD) pulls in-process: the distro's RTTI-bearing `@LLVM_<ver>` symbols vs our
# no-RTTI bundle under the same soname means load failures or a shared cl::opt registry clash. A
# unique symbol-version node here, paired with the unique soname from 0004-libllvm-unique-soname.patch,
# makes the two fully independent: ours resolves against `libLLVMpolyregion.so`, the system's against `libLLVM.so`.
set(LLVM_SHLIB_SYMBOL_VERSION "POLYREGION_LLVM" CACHE STRING "")
# XXX Force /MT to match polyregion and vcpkg static-CRT deps; LLVM defaults to /MD and
# the resulting MD_DynamicRelease markers conflict with /MT consumers.
set(LLVM_USE_CRT_RELEASE MT CACHE STRING "")
set(LLVM_USE_CRT_RELWITHDEBINFO MT CACHE STRING "")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "")
# Plugins are independent of the dylib axis on Unix (host tools get -rdynamic via
# LLVM_EXPORT_SYMBOLS_FOR_PLUGINS). On Windows the generated .def overflows the PE 65535
# export limit, so plugin sources are folded into polycpp/polyfc instead (POLYREGION_FUSED_DRIVER).
if (WIN32)
    set(LLVM_ENABLE_PLUGINS OFF CACHE BOOL "")
else ()
    set(LLVM_ENABLE_PLUGINS ON CACHE BOOL "")
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

# The dylib variant must link libLLVM.so against the system shared libstdc++: -static-libstdc++
# bakes libstdc++ into libLLVM.so and re-exports it through the LLVM_22.1 version script, so
# a later dlopen of libhsa-runtime64 resolves codecvt/basic_filebuf vtable slots against the
# wrong copy and SEGVs inside do_unshift. The static variant has no such consumer.
if (POLYREGION_LLVM_DYLIB)
    set(LLVM_STATIC_LINK_CXX_STDLIB OFF CACHE BOOL "")
else ()
    set(LLVM_STATIC_LINK_CXX_STDLIB ON CACHE BOOL "")
endif ()
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

        # For ${PROJECT}Exports.cmake so find_package(...) works
        llvm-libraries
        clang-libraries
        mlir-libraries
        flang-libraries
        lld-libraries

        # Headers
        llvm-headers
        clang-headers
        mlir-headers
        flang-headers
        lld-headers

        # CMake configs (LLVMConfig.cmake + LLVMExports.cmake etc., per project)
        cmake-exports
        clang-cmake-exports
        lld-cmake-exports
        mlir-cmake-exports
        flang-cmake-exports

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
