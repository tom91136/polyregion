cmake_minimum_required(VERSION 3.18) # XXX FetchContent_Declare DOWNLOAD_NO_EXTRACT is >= 3.18

include(FetchContent)
include(ProcessorCount)
include(ProjectConfig.cmake)


if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)
set(USE_LINKER "" CACHE INTERNAL "Path to custom linker")
set(POLYREGION_SHARED_LIB OFF CACHE BOOL "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
ProcessorCount(NPROC)


project(polyregion-native)

FetchContent_Declare(
        libffi
        URL https://github.com/libffi/libffi/releases/download/v3.4.4/libffi-3.4.4.tar.gz
        URL_HASH SHA256=d66c56ad259a82cf2a9dfc408b32bf5da52371500b84745f7fb8b645712df676
)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
        URL_HASH SHA256=665fa14b8af3837966949e8eb0052d583e2ac105d3438baba9951785512cf921
        DOWNLOAD_NO_EXTRACT YES # this is a header only lib, so don't do anything else
)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.2.1
)

FetchContent_Declare(
        Vulkan-Headers
        URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.3.231.zip
        URL_HASH SHA256=68fa5e19776d0e371cd86878db882a5aa9a4d6e3d91ed9363f8ea14e3c356cc6
)

#FetchContent_Declare(
#        volk
#        URL https://github.com/zeux/volk/archive/refs/tags/1.3.215.zip
#        URL_HASH SHA256=9fd7bb2a2e0fb0aaccf32847638b7f8f5e1ebea242f2e8e7a7ab741d6c302f1c
#)

#FetchContent_Declare(
#        fmt
#        URL https://github.com/fmtlib/fmt/releases/download/9.1.0/fmt-9.1.0.zip
#        URL_HASH SHA256=cceb4cb9366e18a5742128cb3524ce5f50e88b476f1e54737a47ffdf4df4c996
#)

#FetchContent_Declare(
#        Kompute
#        URL https://github.com/KomputeProject/kompute/archive/e7890bbb4d82ebfb855022e93a2de8b64091cd98.zip
#        URL_HASH SHA256=f06576e35769fa6789e5ce1f7d073e92b5cdd94d6c6b2b992cc0e274df84e4e6
#)

#FetchContent_Declare(
#        spdlog
#        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.11.0.zip
#        URL_HASH SHA256=33f83c6b86ec0fbbd0eb0f4e980da6767494dc0ad063900bcfae8bc3e9c75f21
#)


if (USE_LINKER)
    message(STATUS "Using custom linker: ${USE_LINKER}")
    if (MSVC)
        set(CMAKE_LINKER "${USE_LINKER}")
    else ()
        string(FIND "${USE_LINKER}" "ld.lld" USES_LLD_DRIVER REVERSE)
        if ("${USES_LLD_DRIVER}" STREQUAL "-1")
            string(REPLACE lld ld.lld USE_LINKER "${USE_LINKER}")
        endif ()
        set(COMMON_LINK_OPTIONS --ld-path=${USE_LINKER} ${COMMON_LINK_OPTIONS})
    endif ()
endif ()


if (MSVC)
    # TODO Enable Asan on Windows if possible
    set(COMMON_DEBUG_OPTIONS -fno-omit-frame-pointer)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Release>:>")
else ()
    # FIXME adjust for CUDA stuff for now
    set(COMMON_DEBUG_OPTIONS -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -static-libsan -g3)
#    set(COMMON_DEBUG_OPTIONS)
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
            -Wthread-safety
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

    # Setup link options
    if (APPLE)
        # LLD support for macOS is not there yet, flags are for ld64.
    else ()
        # For lld, we setup LTO, ICF, and threading
        set(COMMON_LINKER_OPTIONS
                LINKER:--lto-O3
                LINKER:--icf=all
                LINKER:--threads=${NPROC}
                LINKER:--no-allow-shlib-undefined
                LINKER:--no-undefined-version
                LINKER:--no-undefined
                LINKER:-O3                # O3 for linker means optimise for space
                )
    endif ()
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        # For Release, we also strip everything out
        if (APPLE)
            set(COMMON_LINKER_OPTIONS
                    ${COMMON_LINKER_OPTIONS}
                    LINKER:-dead_strip
                    LINKER:-dead_strip_dylibs
                    LINKER:-S # No debug symbols
                    LINKER:-x # No local symbols
                    )
        else ()
            set(COMMON_LINKER_OPTIONS
                    ${COMMON_LINKER_OPTIONS}
                    LINKER:--gc-sections      # remove unused sections
                    LINKER:--strip-all        # strip symbols not used for DSO
                    )
        endif ()
    endif ()
endif ()


set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS} ${COMMON_LINKER_OPTIONS})

if (UNIX AND NOT APPLE)
    set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS}
            -static-libgcc -static-libstdc++)
endif ()


if (WIN32)
    set(DL_LIB Version) # we need Version.lib for stuff like GetFileVersionInfoSizeA
else ()
    set(DL_LIB dl)
endif ()


# use new behaviour for IPO here (e.g enable where possible)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)


if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
    cmake_policy(SET CMP0116 OLD)
    set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)
endif ()

set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/llvm")
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/mlir")
set(LLD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}/lib/cmake/lld")

message(STATUS "Using LLVM at ${LLVM_DIR}")
message(STATUS "Using MLIR at ${MLIR_DIR}")
message(STATUS "Using LLD at ${LLD_DIR}")

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
find_package(MLIR REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
find_package(LLD REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

message(STATUS "Using LLVM: ${LLVM_PACKAGE_VERSION} tools")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")


FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(Catch2)
FetchContent_MakeAvailable(Vulkan-Headers)
#FetchContent_MakeAvailable(fmt)
#FetchContent_MakeAvailable(spdlog)

#FetchContent_MakeAvailable(volk)

#set(KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER OFF CACHE BOOL "")


#set(CMAKE_DISABLE_FIND_PACKAGE_Vulkan)
#set(Vulkan_LIBRARY "a")
#set(Vulkan_INCLUDE_DIR "${vulkan-headers_SOURCE_DIR}/include/vulkan")
#message(FATAL_ERROR ${vulkan-headers_SOURCE_DIR}/include/vulkan)

#FetchContent_GetProperties(kompute)
#if (NOT kompute_POPULATED)
#    FetchContent_Populate(kompute)
##    add_subdirectory(${kompute_SOURCE_DIR} ${kompute_BINARY_DIR} EXCLUDE_FROM_ALL)
#    set(kompute_INCLUDE_DIRS "${kompute_SOURCE_DIR}/src/include")
#    file(GLOB kompute_SOURCE_FILES "${kompute_SOURCE_DIR}/src/*.cpp")
#endif ()

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
