cmake_minimum_required(VERSION 3.23 FATAL_ERROR) # XXX FetchContent_Declare DOWNLOAD_NO_EXTRACT is >= 3.18

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20.0")
    cmake_policy(SET CMP0116 NEW)
endif ()

# use new behaviour for IPO here (e.g enable where possible)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS_RELEASE "-O3  -gline-tables-only -gz" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3  -gline-tables-only -gz" CACHE STRING "")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS_RELEASE "-O3  -g1 -gz" CACHE STRING "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3  -g1 -gz" CACHE STRING "")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # TODO
endif ()

include(FetchContent)
include(ProcessorCount)
include(ProjectConfig.cmake)
include(cmake/mergestaticlibs.cmake)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Tools land in bin/, runtime/plugin libs in lib/{,polycpp,polyfc}/lib — find libLLVM.so via $ORIGIN.
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../../../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)


# Bundled LLVM is built with LLVM_ENABLE_LTO=Thin, so its static archives contain bitcode
# and only lld can link them. Default to lld on Linux unless the caller picked something.
# Plain (non-CACHE) set so -DUSE_LINKER=... from the command line still wins.
if (NOT USE_LINKER AND UNIX AND NOT APPLE)
    set(USE_LINKER "lld")
endif ()
set(POLYREGION_SHARED_LIB OFF CACHE BOOL "")
ProcessorCount(NPROC)

project(polyregion-native)

FetchContent_Declare(
        fire
        URL https://raw.githubusercontent.com/kongaskristjan/fire-hpp/c31b9d6593ee4edfdf9df875d006e26d9a8df879/include/fire-hpp/fire.hpp
        URL_HASH SHA256=f9b82674d0c0a7322043fe837bce895d5798d18585f0e4b1ff105418a8a9e2f8
        DOWNLOAD_NO_EXTRACT YES
)

FetchContent_Declare(
        Aspartame
        #               SOURCE_DIR /home/tom/Aspartame
        GIT_REPOSITORY https://github.com/tom91136/Aspartame.git
        GIT_TAG e34db235
)

FetchContent_MakeAvailable(Aspartame)

find_package(fmt CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(unofficial-libffi CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(qjs CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

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
        set(LLVM_LINKER_DETECTED ON)
        set(LLVM_LINKER_IS_LLD ON)
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
    #        set(COMMON_DEBUG_OPTIONS)
endif ()

if (MSVC)
    set(COMMON_COMPILE_OPTIONS
            /W4

            /bigobj
            /nologo

            /Zc:preprocessor # XXX IMPORTANT: Clang forwards __VA_ARGS__ in macros and need this to work
            # Disabled warnings.
            -wd4141 # Suppress ''modifier' : used more than once' (because of __forceinline combined with inline)
            -wd4146 # Suppress 'unary minus operator applied to unsigned type, result still unsigned'
            -wd4244 # Suppress ''argument' : conversion from 'type1' to 'type2', possible loss of data'
            -wd4267 # Suppress ''var' : conversion from 'size_t' to 'type', possible loss of data'
            -wd4291 # Suppress ''declaration' : no matching operator delete found; memory will not be freed if initialization throws an exception'
            -wd4351 # Suppress 'new behavior: elements of array 'array' will be default initialized'
            -wd4456 # Suppress 'declaration of 'var' hides local variable'
            -wd4457 # Suppress 'declaration of 'var' hides function parameter'
            -wd4458 # Suppress 'declaration of 'var' hides class member'
            -wd4459 # Suppress 'declaration of 'var' hides global declaration'
            -wd4503 # Suppress ''identifier' : decorated name length exceeded, name was truncated'
            -wd4624 # Suppress ''derived class' : destructor could not be generated because a base class destructor is inaccessible'
            -wd4722 # Suppress 'function' : destructor never returns, potential memory leak
            -wd4100 # Suppress 'unreferenced formal parameter'
            -wd4127 # Suppress 'conditional expression is constant'
            -wd4512 # Suppress 'assignment operator could not be generated'
            -wd4505 # Suppress 'unreferenced local function has been removed'
            -wd4610 # Suppress '<class> can never be instantiated'
            -wd4510 # Suppress 'default constructor could not be generated'
            -wd4702 # Suppress 'unreachable code'
            -wd4245 # Suppress ''conversion' : conversion from 'type1' to 'type2', signed/unsigned mismatch'
            -wd4706 # Suppress 'assignment within conditional expression'
            -wd4310 # Suppress 'cast truncates constant value'
            -wd4701 # Suppress 'potentially uninitialized local variable'
            -wd4703 # Suppress 'potentially uninitialized local pointer variable'
            -wd4389 # Suppress 'signed/unsigned mismatch'
            -wd4611 # Suppress 'interaction between '_setjmp' and C++ object destruction is non-portable'
            -wd4805 # Suppress 'unsafe mix of type <type> and type <type> in operation'
            -wd4204 # Suppress 'nonstandard extension used : non-constant aggregate initializer'
            -wd4577 # Suppress 'noexcept used with no exception handling mode specified; termination on exception is not guaranteed'
            -wd4091 # Suppress 'typedef: ignored on left of '' when no variable is declared'
            # C4592 is disabled because of false positives in Visual Studio 2015
            # Update 1. Re-evaluate the usefulness of this diagnostic with Update 2.
            -wd4592 # Suppress ''var': symbol will be dynamically initialized (implementation limitation)
            -wd4319 # Suppress ''operator' : zero extending 'type' to 'type' of greater size'
            # C4709 is disabled because of a bug with Visual Studio 2017 as of
            # v15.8.8. Re-evaluate the usefulness of this diagnostic when the bug
            # is fixed.
            -wd4709 # Suppress comma operator within array index expression

            # We'd like this warning to be enabled, but it triggers from code in
            # WinBase.h that we don't have control over.
            -wd5105 # Suppress macro expansion producing 'defined' has undefined behavior

            # Ideally, we'd like this warning to be enabled, but even MSVC 2019 doesn't
            # support the 'aligned' attribute in the way that clang sources requires (for
            # any code that uses the LLVM_ALIGNAS macro), so this is must be disabled to
            # avoid unwanted alignment warnings.
            -wd4324 # Suppress 'structure was padded due to __declspec(align())'

            # Promoted warnings.
            -w14062 # Promote 'enumerator in switch of enum is not handled' to level 1 warning.

            # Promoted warnings to errors.
            -we4238 # Promote 'nonstandard extension used : class rvalue used as lvalue' to error.
    )
    add_compile_definitions(
            # For some reason MS wants to deprecate a bunch of standard functions...
            _CRT_SECURE_NO_DEPRECATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_DEPRECATE
            _CRT_NONSTDC_NO_WARNINGS
            _SCL_SECURE_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
    )
    add_compile_definitions(
            UNICODE
            _UNICODE
    )
else ()

    set(COMMON_RELEASE_OPTIONS)
    set(COMMON_COMPILE_OPTIONS
            -ffunction-sections       # keep functions in its own section for better --gc-sections results
            -fdata-sections           # same as above
            -fno-rtti                 # don't use RTTI for space
            -pedantic
            -Wall
            -Wextra
            # -Wshadow         # too prevalent in LLVM
            # -Wold-style-cast # too prevalent in LLVM
            # -Wcast-align      # too prevalent in LLVM
            -Werror=switch
            -Wno-unused-parameter
            -Wno-unused-function
            -Wno-unused-variable
            "$<$<CXX_COMPILER_ID:Clang>:-Wthread-safety>"
            "$<$<CXX_COMPILER_ID:Clang>:-Werror=return-type>"
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
        #        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        #        set(CMAKE_CXX_COMPILE_OPTIONS_IPO -flto=thin)
    endif ()

    if (UNIX)
        set(COMMON_COMPILE_OPTIONS
                ${COMMON_COMPILE_OPTIONS}
                -fvisibility=hidden       # don't export anything by default
                -fvisibility-inlines-hidden
        )
    endif ()

    # Setup link options
    if (APPLE)
        # LLD support for macOS is not there yet, flags are for ld64.
        set(STRICT_LINK_OPTIONS) # ld64 defaults to error on undefined on missing symbols, so nothing to do
        set(PLUGIN_LINK_OPTIONS LINKER:-undefined,dynamic_lookup)
    else ()
        set(STRICT_LINK_OPTIONS
                LINKER:--no-allow-shlib-undefined
                LINKER:--no-undefined-version
                LINKER:--no-undefined
        )
        set(PLUGIN_LINK_OPTIONS)
        if (USE_LINKER STREQUAL "lld") # For lld, we setup LTO, ICF, and threading
            set(COMMON_LINK_OPTIONS
                    ${COMMON_LINK_OPTIONS}
                    LINKER:--threads=${NPROC}
                    LINKER:-O3                # O3 for linker means optimise for space
                    # LINKER:--lto-O3
                    # LINKER:--icf=all
            )
        endif ()
    endif ()
    if (CMAKE_BUILD_TYPE MATCHES "Rel*")
        if (APPLE) # For Release, we also strip everything out
            set(COMMON_LINK_OPTIONS
                    ${COMMON_LINK_OPTIONS}
                    LINKER:-dead_strip
                    LINKER:-dead_strip_dylibs
                    LINKER:-S # No debug symbols
                    LINKER:-x # No local symbols
            )
        else ()
            set(COMMON_LINK_OPTIONS
                    ${COMMON_LINK_OPTIONS}
                    -static-libstdc++
                    LINKER:--gc-sections        # remove unused sections
                    LINKER:--compress-debug-sections=zlib
                    # LINKER:--strip-all        # strip symbols not used for DSO
            )
        endif ()
    endif ()
endif ()

if (UNIX AND NOT APPLE)
    set(COMMON_LINK_OPTIONS ${COMMON_LINK_OPTIONS}
            -static-libgcc -static-libstdc++
            # XXX Hide statically-embedded libstdc++ symbols from the dynsym table
            LINKER:--exclude-libs=libstdc++.a)
endif ()


if (WIN32)
    set(DL_LIB Version) # we need Version.lib for stuff like GetFileVersionInfoSizeA
else ()
    set(DL_LIB dl)
endif ()


if (DEFINED ENV{POLYREGION_LLVM_DYLIB} AND "$ENV{POLYREGION_LLVM_DYLIB}" STREQUAL "OFF")
    set(LLVM_VARIANT static)
else ()
    set(LLVM_VARIANT dylib)
endif ()
set(LLVM_BUILD_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-${LLVM_VARIANT}")
set(LLVM_DIR "${LLVM_BUILD_PREFIX}/lib/cmake/llvm")
set(LLD_DIR "${LLVM_BUILD_PREFIX}/lib/cmake/lld")
set(Clang_DIR "${LLVM_BUILD_PREFIX}/lib/cmake/clang")
set(LLVM_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/polyregion-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-${LLVM_VARIANT}-dist")

set(Flang_DIR "${LLVM_BUILD_PREFIX}/lib/cmake/flang")
set(MLIR_DIR "${LLVM_BUILD_PREFIX}/lib/cmake/mlir")

message(STATUS "Using LLVM at ${LLVM_DIR}")
message(STATUS "Using LLD at ${LLD_DIR}")
message(STATUS "Using Clang at ${Clang_DIR}")
message(STATUS "Using Flang at ${Flang_DIR}")
message(STATUS "Using MLIR at ${MLIR_DIR}")

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
find_package(Clang REQUIRED PATHS ${Clang_DIR} NO_DEFAULT_PATH)
find_package(LLD REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

# Defaults to whatever LLVM was built with; override via -DPOLYREGION_LLVM_DYLIB=OFF for static linking.
option(POLYREGION_LLVM_DYLIB "Link polyregion against libLLVM dylib" ${LLVM_LINK_LLVM_DYLIB})
message(STATUS "POLYREGION_LLVM_DYLIB = ${POLYREGION_LLVM_DYLIB}")

find_package(MLIR REQUIRED PATHS ${MLIR_DIR} NO_DEFAULT_PATH)
find_package(Flang REQUIRED PATHS ${Flang_DIR} NO_DEFAULT_PATH)

message(STATUS "Using LLVM: ${LLVM_PACKAGE_VERSION} tools")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

FetchContent_MakeAvailable(fire)


set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RelWithDebInfo>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:Release>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
        CACHE INTERNAL ""
)

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${STRICT_LINK_OPTIONS}
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
        CACHE INTERNAL ""
)

# Link options for binaries that dlopen vendor GPU runtimes (HIP, HSA, OpenCL ICDs).
# XXX AMD's  libamdhip64 / libamdocl64 / libamdrocclr_static call libstdc++'s codecvt::do_in inside
# `populateOCLDeviceConstants`; if the host process embeds its own static libstdc++, the locale
# facet registry diverges from libstdc++.so.6's and AMD SEGVs. Linking dynamic libstdc++ keeps
# one libstdc++ in the process.
# To support AMD, we basically cannot static link libstdc++
set(GPU_RUNTIME_LINK_OPTIONS ${LINK_OPTIONS} CACHE INTERNAL "")
if (UNIX AND NOT APPLE)
    list(REMOVE_ITEM GPU_RUNTIME_LINK_OPTIONS
            -static-libstdc++
            -static-libgcc
            "LINKER:--exclude-libs=libstdc++.a")
endif ()

set(PLUGIN_LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${PLUGIN_LINK_OPTIONS}
        "$<$<CONFIG:Debug>:${COMMON_DEBUG_OPTIONS}>"
        CACHE INTERNAL ""
)

enable_testing()
set(CMAKE_CTEST_ARGUMENTS
        "-j${NPROC}"
        "--output-on-failure"
        "--output-junit;${CMAKE_BINARY_DIR}/test-results.xml"
        CACHE INTERNAL "")

add_subdirectory(common)
add_subdirectory(polyast)
add_subdirectory(polyc)
add_subdirectory(polyinvoke)
add_subdirectory(polyfront)
add_subdirectory(polyreflect)
add_subdirectory(polytest)
add_subdirectory(polycpp)
add_subdirectory(polyfc)
add_subdirectory(bindings)

# ## test_all
# `cmake --build <build> --target test_all` => ${CMAKE_BINARY_DIR}/test-results.xml.
set(POLYREGION_TEST_BINARIES
        polycommon-tests
        polyc-tests
        polyinvoke-tests
        polyrt-tests
        polyreflect-rt-tests
        polycpp-tests
        polyfc-tests)


add_custom_target(test_all
        COMMAND ${CMAKE_CTEST_COMMAND} ${CMAKE_CTEST_ARGUMENTS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${POLYREGION_TEST_BINARIES}
        USES_TERMINAL
        VERBATIM
        COMMENT "ctest -j${NPROC} --output-junit test-results.xml")

# ## format/format-check
# `cmake --build <build> --target format`        => clang-format -i across the tree
# `cmake --build <build> --target format-check`  => clang-format --dry-run --Werror (CI gate)
file(GLOB_RECURSE POLYREGION_FORMAT_SOURCES
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/*.cpp"
        "${CMAKE_SOURCE_DIR}/*.cc"
        "${CMAKE_SOURCE_DIR}/*.h"
        "${CMAKE_SOURCE_DIR}/*.hpp")
list(FILTER POLYREGION_FORMAT_SOURCES EXCLUDE REGEX
        "/(_deps|llvm-[^/]+|polyregion-[^/]+-dist|vcpkg_installed|build-[^/]+|cmake-build-[^/]+|CMakeFiles)/|/polyinvoke/thirdparty/|/(debian-sysroot-image-creator|sysroot-[^/]+)/")

# Prefer the bundled LLVM dist's clang-format so output is reproducible across hosts.
file(GLOB POLYREGION_LLVM_DIST_BIN "${CMAKE_SOURCE_DIR}/llvm-*-dist/bin")
find_program(POLYREGION_CLANG_FORMAT NAMES clang-format HINTS ${POLYREGION_LLVM_DIST_BIN})

if (POLYREGION_CLANG_FORMAT)
    add_custom_target(format
            COMMAND ${POLYREGION_CLANG_FORMAT} --style=file -i ${POLYREGION_FORMAT_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            VERBATIM
            COMMENT "clang-format -i (${POLYREGION_CLANG_FORMAT})")
    add_custom_target(format-check
            COMMAND ${POLYREGION_CLANG_FORMAT} --style=file --dry-run --Werror ${POLYREGION_FORMAT_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            VERBATIM
            COMMENT "clang-format --dry-run (${POLYREGION_CLANG_FORMAT})")
else ()
    set(_no_cf_msg "clang-format not found; build the bundled LLVM (cmake -P build_llvm.cmake) or install one on PATH")
    add_custom_target(format
            COMMAND ${CMAKE_COMMAND} -E echo "${_no_cf_msg}"
            COMMAND ${CMAKE_COMMAND} -E false)
    add_custom_target(format-check
            COMMAND ${CMAKE_COMMAND} -E echo "${_no_cf_msg}"
            COMMAND ${CMAKE_COMMAND} -E false)
endif ()
