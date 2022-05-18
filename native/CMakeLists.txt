cmake_minimum_required(VERSION 3.18)
include(FetchContent)

project(polyregion-native)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)

set(LIBFFI_LIBRARY_PATH "" CACHE INTERNAL "libffi library path")
set(LIBFFI_HEADER_PATH "" CACHE INTERNAL "libffi header path")
set(LINKER "" CACHE INTERNAL "Path to custom linker")
set(POLYREGION_SHARED_LIB OFF CACHE BOOL "")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "LTO disabled in Debug mode")
    set(DEBUG ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
else ()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    set(DEBUG OFF)
    set(CMAKE_CXX_COMPILE_OPTIONS_IPO -flto=thin)
endif ()

set(LLVM_SRC_VERSION 14.0.3)
set(LLVM_SRC_DIR_IN_RELEASE llvm-${LLVM_SRC_VERSION}.src)
set(LLD_SRC_DIR_IN_RELEASE lld-${LLVM_SRC_VERSION}.src)

FetchContent_Declare(
        llvm
        URL https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_SRC_VERSION}/llvm-${LLVM_SRC_VERSION}.src.tar.xz
        URL_HASH SHA256=1e09e8c26e1b67bc94a128b62e9b9c24b70c697a2436a479c9e5eedc4ae29654
)

FetchContent_Declare(
        lld
        URL https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_SRC_VERSION}/lld-${LLVM_SRC_VERSION}.src.tar.xz
        URL_HASH SHA256=a91716f59f8e78018246a7ab6cad1bfdddec270c4db6d3d2a394844ee8799c8f
)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.hpp
        URL_HASH SHA256=e832d339d9e0c042e7dff807754769d778cf5d6ae9730ce21eed56de99cb5e86
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't anything else
)

FetchContent_Declare(
        catch2
        URL https://github.com/catchorg/Catch2/releases/download/v2.13.9/catch.hpp
        URL_HASH SHA256=27da57c7a06d09be8dd81fab7246b79e7892b6ae7e4e49ba8631f1d5a955e3fc
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't anything else
)


# use new behaviour for IPO here (e.g enable where possible)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)


set(USE_EXTERNAL_LLVM OFF)

if (USE_EXTERNAL_LLVM)
    set(LLVM_DIR /lib/llvm-12/cmake)
    set(LLD_DIR /lib/llvm-12/lib/cmake/lld)
    find_package(LLVM REQUIRED CONFIG)
    find_package(LLD REQUIRED CONFIG)

else ()
    # X86;AArch64;ARM;NVPTX;AMDGPU;WebAssembly
    set(LLVM_TARGETS_TO_BUILD "X86;AArch64;ARM;NVPTX;AMDGPU" CACHE INTERNAL "")
    #    set(LLVM_TARGETS_TO_BUILD "X86" CACHE INTERNAL "")
    set(LLVM_ENABLE_IDE ON CACHE BOOL "")
    set(LLVM_ENABLE_THREADS ON CACHE BOOL "")
    if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        set(LLVM_ENABLE_LTO "Thin" CACHE INTERNAL "")
    endif ()

    set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
    set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
    set(LLVM_ENABLE_LIBPFM OFF CACHE BOOL "")
    set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")

    set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")
    set(LLVM_STATIC_LINK_CXX_STDLIB ON CACHE BOOL "")
    set(BUILD_SHARED_LIBS ${POLYREGION_SHARED_LIB} CACHE BOOL "")

    #    set(LLVM_ENABLE_LLD ON CACHE BOOL "")

    #    if (LINKER)
    #        set(LLVM_USE_LINKER "${LINKER}" CACHE INTERNAL "")
    #    endif ()
    #    set(CMAKE_CXX_FLAGS "-fvisibility=hidden  -ffunction-sections -fdata-sections" CACHE INTERNAL "")


    set(LLVM_ENABLE_RTTI ON CACHE BOOL "")

    set(LLVM_BUILD_RUNTIME OFF CACHE BOOL "")

    set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
    set(LLVM_INCLUDE_TOOLS OFF CACHE BOOL "")

    set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")

    set(LLVM_BUILD_TESTS OFF CACHE BOOL "")
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")

    set(LLVM_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")

    set(LLVM_BUILD_DOCS OFF CACHE BOOL "")
    set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")

    set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
    set(LLVM_INSTALL_UTILS OFF CACHE BOOL "")

    set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
    set(LLVM_USE_HOST_TOOLS OFF CACHE BOOL "")
    # XXX there appears to be a bug in llvm_create_cross_target where LLVM_INCLUDE_BENCHMARKS isn't honoured
    # so we set LLVM_USE_HOST_TOOLS to off

    cmake_policy(SET CMP0116 OLD)
    set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)


    # XXX all LLVM specific cmake variables must be set BEFORE FetchContent_MakeAvailable otherwise they have no effect
    # FetchContent_MakeAvailable was introduced in 3.14 but there isn't a version that accepts extra args at the end
    # for add_subdirectory hence we write one ourselves for llvm only
    FetchContent_GetProperties(llvm)
    if (NOT llvm_POPULATED)
        FetchContent_Populate(llvm)
        # XXX as we are doing our custom FetchContent_MakeAvailable, SOURCE_SUBDIR isn't available so we do it manually here
        set(llvm_SOURCE_DIR "${llvm_SOURCE_DIR}/${LLVM_SRC_DIR_IN_RELEASE}")
        set(llvm_BINARY_DIR "${llvm_BINARY_DIR}/${LLVM_SRC_DIR_IN_RELEASE}")
        add_subdirectory(${llvm_SOURCE_DIR} ${llvm_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()


    #list (APPEND CMAKE_MODULE_PATH "${llvm_SOURCE_DIR}/cmake/modules")
    message(STATUS "${llvm_BINARY_DIR}/lib/cmake/llvm")
    find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH
            PATHS "${llvm_BINARY_DIR}/lib/cmake/llvm")
    include_directories("${llvm_BINARY_DIR}/include" "${llvm_SOURCE_DIR}/include")
    message(STATUS "Using LLVM: ${LLVM_PACKAGE_VERSION} (${llvm_SOURCE_DIR})")


    FetchContent_GetProperties(lld)
    if (NOT lld_POPULATED)
        FetchContent_Populate(lld)
        # XXX as we are doing our custom FetchContent_MakeAvailable, SOURCE_SUBDIR isn't available so we do it manually here
        set(lld_SOURCE_DIR "${lld_SOURCE_DIR}/${LLD_SRC_DIR_IN_RELEASE}")
        set(lld_BINARY_DIR "${lld_BINARY_DIR}/${LLD_SRC_DIR_IN_RELEASE}")

        set(LLVM_TABLEGEN_EXE ${llvm_BINARY_DIR}/bin/llvm-tblgen)
        set(LLD_VERSION ${LLVM_SRC_VERSION})

        #        configure_file(
        #                ${lld_SOURCE_DIR}/include/lld/Common/Version.inc.in
        #                ${lld_BINARY_DIR}/include/lld/Common/Version.inc)


        macro(add_lld_library name)
            cmake_parse_arguments(ARG
                    "SHARED"
                    ""
                    ""
                    ${ARGN})
            if (ARG_SHARED)
                set(ARG_ENABLE_SHARED SHARED)
            endif ()
            llvm_add_library(${name} ${ARG_ENABLE_SHARED} ${ARG_UNPARSED_ARGUMENTS})
            set_target_properties(${name} PROPERTIES FOLDER "lld libraries")
        endmacro(add_lld_library)

        add_subdirectory(${lld_SOURCE_DIR}/Common ${lld_BINARY_DIR}/Common)
        target_include_directories(lldCommon
                PUBLIC ${lld_SOURCE_DIR}/include ${lld_BINARY_DIR}/include
                PUBLIC ${lld_BINARY_DIR}/Common)

        add_subdirectory(${lld_SOURCE_DIR}/ELF ${lld_BINARY_DIR}/ELF)

        target_include_directories(lldELF
                PUBLIC ${lld_SOURCE_DIR}/include ${lld_BINARY_DIR}/include
                PUBLIC ${lld_BINARY_DIR}/ELF
                PUBLIC ${lld_SOURCE_DIR}/ELF)
        #        add_subdirectory(COFF)
        #        add_subdirectory(MachO)
        #        add_subdirectory(MinGW)
        #        add_subdirectory(wasm)

    endif ()


endif ()


FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(catch2)

include_directories("${json_SOURCE_DIR}")
include_directories("${catch2_SOURCE_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})

# set CXX stuff here in case LLVM overwrites it
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# FIXME adjust for CUDA stuff for now
set(COMMON_DEBUG_OPTIONS -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -static-libsan)
#set(COMMON_DEBUG_OPTIONS  )

set(COMMON_COMPILE_OPTIONS
        -ffunction-sections       # keep functions in its own section for better --gc-sections results
        -fdata-sections           # same as above
        -fno-rtti                 # don't use RTTI for space
        -pedantic
        -Wall
        -Wextra
        # -Wshadow         # too prevalent in LLVM
        # -Wold-style-cast # too prevalent in LLVM
        -Wcast-align
        -Werror=return-type
        -Wno-unused-parameter
        -Wno-unused-function
        -Wno-unused-variable
        )

if (DEBUG)
    # --exclude-libs=ALL kills -fsanitize=*
    # keep sections for debug!
    set(COMMON_COMPILE_OPTIONS ${COMMON_COMPILE_OPTIONS})
else ()
    set(COMMON_LINKER_OPTIONS
            LINKER:--no-allow-shlib-undefined
            LINKER:--no-undefined-version
            LINKER:--no-undefined
            LINKER:--exclude-libs=ALL # ld/lld exports all object symbols by default, don't do that
            LINKER:--gc-sections      # remove unused sections
            LINKER:-s                 # strip symbols not used for DSO
            LINKER:-O3                # O3 for linker means optimise for space
            )
    set(COMMON_COMPILE_OPTIONS ${COMMON_COMPILE_OPTIONS}
            -fvisibility=hidden       # don't export anything by default
            -fvisibility-inlines-hidden
            )
endif ()

set(COMMON_LINK_OPTIONS
        -static-libgcc
        -static-libstdc++
        ${COMMON_COMPILE_OPTIONS}
        ${COMMON_LINKER_OPTIONS})

if (LINKER)
    set(COMMON_LINK_OPTIONS --ld-path=${LINKER} ${COMMON_LINK_OPTIONS})
endif ()

add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(bindings)
