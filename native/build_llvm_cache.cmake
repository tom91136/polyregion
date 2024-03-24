

if (UNIX)
    list(APPEND RUNTIME_COMPONENTS compiler-rt libcxx libcxxabi)
#    list(APPEND RUNTIME_TARGETS cxx-headers)
elseif (WIN32)
    # needs a bootstrapping build for libcxx because cl.exe isn't support
    # see https://libcxx.llvm.org/BuildingLibcxx.html#support-for-windows
    list(APPEND RUNTIME_COMPONENTS compiler-rt)
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
        ARM
        X86
        ARM
        NVPTX
        AMDGPU
        CACHE STRING "")

# Enable the LLVM projects and runtimes.
set(LLVM_ENABLE_PROJECTS
        clang
        clang-tools-extra
        lld
        CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES ${RUNTIME_COMPONENTS} CACHE STRING "")

#set(LLVM_DYLIB_COMPONENTS all CACHE STRING "")
set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB OFF CACHE BOOL "")
#set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")

set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBPFM OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")

set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_BUILD_RUNTIME ON CACHE BOOL "")
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLVM_BUILD_TESTS OFF CACHE BOOL "")

set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")

#set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR OFF CACHE BOOL "")

set(LLVM_STATIC_LINK_CXX_STDLIB ON CACHE BOOL "")
set(COMPILER_RT_DEFAULT_TARGET_ONLY ON CACHE BOOL "")
set(COMPILER_RT_BUILD_LIBFUZZER OFF CACHE BOOL "")
set(COMPILER_RT_SANITIZERS_TO_BUILD "asan;dfsan;msan;hwasan;tsan;cfi" CACHE STRING "")



# You likely want more tools; this is just an example :) Note that we need to
# include cxx-headers explicitly here (in addition to it being added to
# LLVM_RUNTIME_DISTRIBUTION_COMPONENTS above).
set(LLVM_DISTRIBUTION_COMPONENTS
        # LLVM

        # Linker
        lld

        # Clang tools
        clang
        clang-tidy
        clang-rename
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
        llvm-tblgen
        llvm-cov
        llvm-link
        llvm-profdata
        llvm-profgen

        # binutil replacements
        llvm-addr2line
        llvm-ar
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


        CACHE STRING "")

