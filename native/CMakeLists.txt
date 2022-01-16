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


FetchContent_Declare(
        llvm
        URL https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/llvm-13.0.0.src.tar.xz
        URL_HASH MD5=8c24626dce3ee9d87d1079ebf0897db5
)

FetchContent_Declare(
        capstone
        URL https://github.com/capstone-engine/capstone/archive/refs/tags/4.0.2.tar.gz
        URL_HASH MD5=8894344c966a948f1248e66c91b53e2c
)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.10.4/json.hpp
        URL_HASH MD5=c55ff53aa6aa8b87b8313bc2fdcd032a
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't anything else
)

FetchContent_Declare(
        catch2
        URL https://github.com/catchorg/Catch2/releases/download/v2.13.8/catch.hpp
        URL_HASH MD5=e45f70693208dca2b6813a078e3e64eb
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't anything else
)


# FetchContent_MakeAvailable was introduced in 3.14
# there also isn't a version that accepts extra args at the end for add_subdirectory
# hence we write one ourselves
macro(FetchContent_MakeAvailable_Args NAME ARGS)
    FetchContent_GetProperties(${NAME})
    if (NOT ${NAME}_POPULATED)
        FetchContent_Populate(${NAME})
        add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR} ${ARGS})
    endif ()
endmacro()


# use new behaviour for IPO here (e.g enable where possible)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)


#### capstone ###
#set(CAPSTONE_BUILD_TESTS OFF CACHE BOOL "Disable Capstone tests")
#set(CAPSTONE_BUILD_SHARED OFF CACHE BOOL "Disable Capstone shared library")
#set(CAPSTONE_BUILD_CSTOOL OFF CACHE BOOL "Disable cstool build")
#
#set(CAPSTONE_MIPS_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_PPC_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_SPARC_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_SYSZ_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_XCORE_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_M68K_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_TMS320C64X_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_M680X_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_EVM_SUPPORT OFF CACHE BOOL "")
#set(CAPSTONE_OSXKERNEL_SUPPORT OFF CACHE BOOL "")
#
#set(CAPSTONE_ARM_SUPPORT ON CACHE BOOL "")
#set(CAPSTONE_ARM64_SUPPORT ON CACHE BOOL "")
#set(CAPSTONE_X86_SUPPORT ON CACHE BOOL "")
#FetchContent_MakeAvailable_Args(capstone EXCLUDE_FROM_ALL)
#include_directories("${capstone_BINARY_DIR}/include" "${capstone_SOURCE_DIR}/include")
#### capstone ###


set(USE_EXTERNAL_LLVM OFF)

if (USE_EXTERNAL_LLVM)
    set(LLVM_DIR /lib/llvm-12/cmake)
    set(LLD_DIR /lib/llvm-12/lib/cmake/lld)
    find_package(LLVM REQUIRED CONFIG)
    find_package(LLD REQUIRED CONFIG)

else ()
    set(LLVM_TARGETS_TO_BUILD "X86;AArch64;ARM;NVPTX;AMDGPU;WebAssembly" CACHE INTERNAL "")
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


    set(LLVM_ENABLE_RTTI OFF CACHE BOOL "") # so we get typeinfo

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
    set(LLVM_USE_HOST_TOOLS ON CACHE BOOL "")


    cmake_policy(SET CMP0116 OLD)
    set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)


    # XXX all LLVM specific cmake variables must be set BEFORE FetchContent_MakeAvailable otherwise they have no effect
    FetchContent_MakeAvailable_Args(llvm EXCLUDE_FROM_ALL)
    #list (APPEND CMAKE_MODULE_PATH "${llvm_SOURCE_DIR}/cmake/modules")
    find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH
            PATHS "${llvm_BINARY_DIR}/lib/cmake/llvm")
    include_directories("${llvm_BINARY_DIR}/include" "${llvm_SOURCE_DIR}/include")
    message(STATUS "Using LLVM: ${LLVM_PACKAGE_VERSION} (${llvm_SOURCE_DIR})")

endif ()


FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(catch2)

include_directories("${json_SOURCE_DIR}")
include_directories("${catch2_SOURCE_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})

# set CXX stuff here in case LLVM overwrites it
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)


set(COMMON_DEBUG_OPTIONS -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -static-libsan)
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
    set(COMMON_LINKER_OPTIONS
            LINKER:--gc-sections      # remove unused sections
            LINKER:-s                 # strip symbols not used for DSO
            LINKER:-O3                # O3 for linker means optimise for space
            )
    set(COMMON_COMPILE_OPTIONS ${COMMON_COMPILE_OPTIONS})
else ()
    set(COMMON_LINKER_OPTIONS
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
        --ld-path=${CMAKE_LINKER}
        ${COMMON_COMPILE_OPTIONS}
        ${COMMON_LINKER_OPTIONS})

message("CMAKE_LINKER=${CMAKE_LINKER}")

add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(bindings)
