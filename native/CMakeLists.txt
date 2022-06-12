cmake_minimum_required(VERSION 3.18) # XXX FetchContent_Declare DOWNLOAD_NO_EXTRACT is >= 3.18
include(FetchContent)
include(ProcessorCount)

project(polyregion-native)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)

set(USE_LINKER "" CACHE INTERNAL "Path to custom linker")
set(POLYREGION_SHARED_LIB OFF CACHE BOOL "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

ProcessorCount(NPROC)

if (USE_LINKER)
    message(STATUS "Using custom linker: ${USE_LINKER}")
    if (MSVC)
        set(CMAKE_LINKER "${USE_LINKER}")
    else ()
        set(COMMON_LINK_OPTIONS --ld-path=${USE_LINKER} ${COMMON_LINK_OPTIONS})
    endif ()
endif ()


if (MSVC)
    # TODO Enable Asan on Windows if possible
    set(COMMON_DEBUG_OPTIONS -fno-omit-frame-pointer)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    #    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Release>:Release>")
else ()
    # FIXME adjust for CUDA stuff for now
#    set(COMMON_DEBUG_OPTIONS -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -static-libsan)
    set(COMMON_DEBUG_OPTIONS  )
endif ()

if (MSVC)
    set(COMMON_COMPILE_OPTIONS
            /W4
            /nologo
            )
else ()

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
            -Werror=switch
            -Wno-unused-parameter
            -Wno-unused-function
            -Wno-unused-variable
            )
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # --exclude-libs=ALL kills -fsanitize=*
    # keep sections for debug!
    if (WIN32)
        set(POLYREGION_SHARED_LIB OFF)
    else ()
        set(POLYREGION_SHARED_LIB ON)
    endif ()
    set(DEBUG ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    message(STATUS "LTO disabled in Debug mode")
    set(COMMON_COMPILE_OPTIONS ${COMMON_COMPILE_OPTIONS})
else ()
    set(POLYREGION_SHARED_LIB OFF)
    set(DEBUG OFF)

    if (NOT MSVC)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        set(CMAKE_CXX_COMPILE_OPTIONS_IPO -flto=thin)
    endif ()

    set(COMMON_COMPILE_OPTIONS
            ${COMMON_COMPILE_OPTIONS}
            -fvisibility=hidden       # don't export anything by default
            -fvisibility-inlines-hidden
            )
    if (APPLE)
        # LLD support for macOS is not there yet, these flags are for ld64.
        set(COMMON_LINKER_OPTIONS
                LINKER:-dead_strip
                LINKER:-dead_strip_dylibs
                LINKER:-S # No debug symbols
                LINKER:-x # No local symbols
                )
    else ()
        set(COMMON_LINKER_OPTIONS
                LINKER:--lto-O3
                LINKER:--icf=all
                LINKER:--threads=${NPROC}
                LINKER:--no-allow-shlib-undefined
                LINKER:--no-undefined-version
                LINKER:--no-undefined
                LINKER:--exclude-libs=ALL # ld/lld exports all object symbols by default, don't do that
                LINKER:--gc-sections      # remove unused sections
                LINKER:--strip-all        # strip symbols not used for DSO
                LINKER:-O3                # O3 for linker means optimise for space
                )
    endif ()


endif ()


set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS} ${COMMON_LINKER_OPTIONS})

if (UNIX AND NOT APPLE)
    set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS}
            -static-libgcc -static-libstdc++)
endif ()

set(LLVM_SRC_VERSION 14.0.3)
set(LLVM_SRC_DIR_IN_RELEASE llvm-${LLVM_SRC_VERSION}.src)
set(LLD_SRC_DIR_IN_RELEASE lld-${LLVM_SRC_VERSION}.src)

if(WIN32)
    set(DL_LIB Version) # we need Version.lib for stuff like GetFileVersionInfoSizeA
else()
    set(DL_LIB dl)
endif()


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
        libffi
        URL https://github.com/libffi/libffi/releases/download/v3.4.2/libffi-3.4.2.tar.gz
        URL_HASH SHA256=540fb721619a6aba3bdeef7d940d8e9e0e6d2c193595bc243241b77ff9e93620
)

FetchContent_Declare(
        cpuinfo
        GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
        GIT_TAG 9fa621933fc6080b96fa0f037cdc7cd2c69ab272 # head as of 01/06/2022 (DD/MM/YYYY)
)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.hpp
        URL_HASH SHA256=e832d339d9e0c042e7dff807754769d778cf5d6ae9730ce21eed56de99cb5e86
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't do anything else
)

FetchContent_Declare(
        catch2
        URL https://github.com/catchorg/Catch2/releases/download/v2.13.9/catch.hpp
        URL_HASH SHA256=27da57c7a06d09be8dd81fab7246b79e7892b6ae7e4e49ba8631f1d5a955e3fc
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't do anything else
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
    set(BUILD_SHARED_LIBS ${POLYREGION_SHARED_LIB} CACHE BOOL "")

    # X86;AArch64;ARM;NVPTX;AMDGPU;WebAssembly
    set(LLVM_TARGETS_TO_BUILD "X86;AArch64;ARM;NVPTX;AMDGPU" CACHE INTERNAL "")
    #    set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "SPIR" CACHE INTERNAL "")
    #    set(LLVM_TARGETS_TO_BUILD "X86" CACHE INTERNAL "")
    #    set(LLVM_TARGETS_TO_BUILD "" CACHE INTERNAL "")
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
    if (WIN32)
        #        set(LLVM_USE_CRT_RELEASE MT CACHE STRING "")
    else ()
        set(LLVM_STATIC_LINK_CXX_STDLIB ON CACHE BOOL "")
    endif ()

    #    set(LLVM_ENABLE_LLD ON CACHE BOOL "")

    if (USE_LINKER)
        set(LLVM_USE_LINKER "${USE_LINKER}" CACHE INTERNAL "")
    endif ()
    #    set(CMAKE_CXX_FLAGS "-fvisibility=hidden  -ffunction-sections -fdata-sections" CACHE INTERNAL "")


    set(LLVM_ENABLE_RTTI OFF CACHE BOOL "")

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

    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
        cmake_policy(SET CMP0116 OLD)
        set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)
    endif ()


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
    message(STATUS "Using LLVM: ${LLVM_PACKAGE_VERSION} (${llvm_SOURCE_DIR})")


    FetchContent_GetProperties(lld)
    if (NOT lld_POPULATED)
        FetchContent_Populate(lld)
        # XXX as we are doing our custom FetchContent_MakeAvailable, SOURCE_SUBDIR isn't available so we do it manually here
        set(lld_SOURCE_DIR "${lld_SOURCE_DIR}/${LLD_SRC_DIR_IN_RELEASE}")
        set(lld_BINARY_DIR "${lld_BINARY_DIR}/${LLD_SRC_DIR_IN_RELEASE}")

        set(LLVM_TABLEGEN_EXE ${llvm_BINARY_DIR}/bin/llvm-tblgen)
        set(LLD_VERSION ${LLVM_SRC_VERSION})


        configure_file(
                ${lld_SOURCE_DIR}/include/lld/Common/Version.inc.in
                ${lld_BINARY_DIR}/include/lld/Common/Version.inc)


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

            target_compile_options(${name} PRIVATE ${COMMON_COMPILE_OPTIONS})

        endmacro(add_lld_library)

        if (MSVC)
            set(HAVE_CXX_ATOMICS64_WITHOUT_LIB ON)
            set(HAVE_CXX_ATOMICS_WITHOUT_LIB ON)
        endif ()


        set(LLVM_TABLEGEN_FLAGS "${LLVM_INCLUDE_DIRS}")
        list(TRANSFORM LLVM_TABLEGEN_FLAGS PREPEND -I)

        add_subdirectory(${lld_SOURCE_DIR}/Common ${lld_BINARY_DIR}/Common)
        target_include_directories(lldCommon PUBLIC
                ${lld_SOURCE_DIR}/include ${lld_BINARY_DIR}/include
                ${lld_BINARY_DIR}/Common
                "${LLVM_INCLUDE_DIRS}"
                )

        add_subdirectory(${lld_SOURCE_DIR}/ELF ${lld_BINARY_DIR}/ELF)
        target_include_directories(lldELF BEFORE PUBLIC
                ${lld_SOURCE_DIR}/include ${lld_BINARY_DIR}/include
                ${lld_BINARY_DIR}/ELF ${lld_SOURCE_DIR}/ELF
                "${LLVM_INCLUDE_DIRS}"
                )
        #        add_subdirectory(COFF)
        #        add_subdirectory(MachO)
        #        add_subdirectory(MinGW)
        #        add_subdirectory(wasm)

        unset(LLVM_TABLEGEN_FLAGS)
    endif ()
endif ()


FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(catch2)


set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(CPUINFO_BUILD_PKG_CONFIG OFF CACHE BOOL "")

FetchContent_MakeAvailable(cpuinfo)

# libffi does not use CMake so we inject one manually from VCPKG
# See https://github.com/microsoft/vcpkg/blob/2f0872b7d592635e8e4919d178650d549c211464/ports/libffi/CMakeLists.txt
# and related configurations in <ROOT>/libffi
FetchContent_GetProperties(libffi)
if (NOT libffi_POPULATED)
    FetchContent_Populate(libffi)
    file(COPY ${CMAKE_SOURCE_DIR}/libffi/CMakeLists.txt DESTINATION "${libffi_SOURCE_DIR}")
    file(COPY ${CMAKE_SOURCE_DIR}/libffi/libffiConfig.cmake.in DESTINATION "${libffi_SOURCE_DIR}")
    file(COPY ${CMAKE_SOURCE_DIR}/libffi/fficonfig.h DESTINATION "${libffi_SOURCE_DIR}/include")
    if (WIN32)
        set(VCPKG_TARGET_ARCHITECTURE x64)
    endif ()
    set(libffi_DEFINITIONS FFI_BUILDING NO_JAVA_RAW_API)

    add_subdirectory(${libffi_SOURCE_DIR} ${libffi_BINARY_DIR} EXCLUDE_FROM_ALL)
    set(libffi_INCLUDE_DIRS "${libffi_BINARY_DIR}/include")

endif ()

# set CXX stuff here in case LLVM overwrites it
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)


add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(bindings)
