
if (UNIX)
    if (NOT ARCH)
        message(STATUS "ARCH not set, detecting host arch")
        execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH RESULT_VARIABLE SUCCESS OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT SUCCESS EQUAL "0")
            message(FATAL_ERROR "Cannot determine host arch, `uname -m` returned ${SUCCESS}")
        endif ()
    endif ()
elseif (WIN32)
    if (NOT ARCH)
        message(STATUS "ARCH not set, detecting host arch")
        if (DEFINED ENV{PROCESSOR_ARCHITECTURE})
            set(ARCH $ENV{PROCESSOR_ARCHITECTURE})
        else ()
            message(FATAL_ERROR "Cannot determine host arch, `PROCESSOR_ARCHITECTURE` variable is not set")
        endif ()
    endif ()
else ()
    message(FATAL_ERROR "Unknown platform (not Unix-like or Windows)")
endif ()


string(TOLOWER ${CMAKE_HOST_SYSTEM_NAME} CMAKE_HOST_SYSTEM_NAME)

# LLVM forces LLVM_BUILD_LLVM_DYLIB=OFF on MSVC/Windows, so the dylib variant has no
# install targets there. Pin to static on Windows.
if (WIN32)
    set(ENV{POLYREGION_LLVM_DYLIB} OFF)
endif ()

if (DEFINED ENV{POLYREGION_LLVM_DYLIB} AND "$ENV{POLYREGION_LLVM_DYLIB}" STREQUAL "OFF")
    set(LLVM_VARIANT static)
    set(POLYREGION_LLVM_DYLIB OFF)
else ()
    set(LLVM_VARIANT dylib)
    set(POLYREGION_LLVM_DYLIB ON)
endif ()

string(TOLOWER "build-${CMAKE_HOST_SYSTEM_NAME}-${ARCH}-${LLVM_VARIANT}" BUILD_NAME)
set(LLVM_BUILD_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/llvm-${CMAKE_BUILD_TYPE}-${ARCH}-${LLVM_VARIANT}")
set(LLVM_DIST_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/polyregion-${CMAKE_BUILD_TYPE}-${ARCH}-${LLVM_VARIANT}-dist")

message(STATUS "Architecture  = `${ARCH}`")
message(STATUS "LLVM variant  = `${LLVM_VARIANT}`")
message(STATUS "Build name    = `${BUILD_NAME}`")
message(STATUS "LLVM build dir= `${LLVM_BUILD_DIR}`")
message(STATUS "Dist dir      = `${LLVM_DIST_DIR}`")

if (UNIX)
    set(COMPILER_NAME "clang")
    if (NOT APPLE)
        list(APPEND BUILD_OPTIONS -DUSE_LINKER=lld)
    endif ()
else ()
    set(COMPILER_NAME "msvc")
endif ()


set(OVERLAY_ARCH ${ARCH})
if (UNIX AND ARCH STREQUAL "x86_64")
    set(OVERLAY_ARCH amd64)
endif ()

if (NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/toolchains/${CMAKE_HOST_SYSTEM_NAME}-${COMPILER_NAME}-${OVERLAY_ARCH}.cmake")
    if (NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
        unset(CMAKE_TOOLCHAIN_FILE)
        message(STATUS "Cannot find toolchain file ${CMAKE_TOOLCHAIN_FILE} for ${ARCH} (overlay=${OVERLAY_ARCH}), not using one for build...")
    endif ()
endif ()
message(STATUS "Toolchain    = `${CMAKE_TOOLCHAIN_FILE}`")

if (CMAKE_SYSROOT)
    if (NOT EXISTS "${CMAKE_SYSROOT}")
        message(FATAL_ERROR "Cannot find sysroot ${CMAKE_SYSROOT} for ${ARCH}")
    endif ()
    message(STATUS "Sysroot      = `${CMAKE_SYSROOT}`")
else ()
    message(STATUS "No sysroot specified, not cross building...")
endif ()

if (CMAKE_SYSROOT)
    set(ENV{CMAKE_SYSROOT} ${CMAKE_SYSROOT})
    list(APPEND BUILD_OPTIONS -DCMAKE_SYSROOT=${CMAKE_SYSROOT})
endif ()

macro(setup_vcpkg)
    if (DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_ROOT $ENV{VCPKG_ROOT})
        message(STATUS "vcpkg root   = `${VCPKG_ROOT}`")
    else ()
        message(FATAL_ERROR "Environment VCPKG_ROOT not defined")
    endif ()

    if (CMAKE_TOOLCHAIN_FILE)
        list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
        list(APPEND BUILD_OPTIONS -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    else ()
        list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
    endif ()
    if (WIN32)
        if (ARCH STREQUAL "amd64")
            set(VCPKG_TARGET_TRIPLET x64-${CMAKE_HOST_SYSTEM_NAME})
        elseif (ARCH STREQUAL "arm64")
            set(VCPKG_TARGET_TRIPLET arm64-${CMAKE_HOST_SYSTEM_NAME})
        else ()
            message(STATUS "Unknown Windows arch: ${ARCH}")
        endif ()
    else ()
        if (ARCH STREQUAL "x86_64")
            set(VCPKG_TARGET_TRIPLET ${CMAKE_HOST_SYSTEM_NAME}-${COMPILER_NAME}-amd64)
        else ()
            set(VCPKG_TARGET_TRIPLET ${CMAKE_HOST_SYSTEM_NAME}-${COMPILER_NAME}-${ARCH})
        endif ()
    endif ()
    message(STATUS "vcpkg triplet= `${VCPKG_TARGET_TRIPLET}`")
    list(APPEND BUILD_OPTIONS -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET})
endmacro()

set(LLVM_DIST_INSTALL_TARGETS
        install-distribution

        module_files
        tools/flang/tools/f18/install
        install-flang-cmake-exports
        install-flang-libraries
        install-flang-headers

        install-cmake-exports
        install-clang-cmake-exports
        install-lld-cmake-exports

        install-llvm-headers
        install-clang-headers)
if (POLYREGION_LLVM_DYLIB)
    list(APPEND LLVM_DIST_INSTALL_TARGETS install-MLIR)
endif ()

function(check_process_return VALUE NAME)
    if (NOT VALUE EQUAL "0")
        message(FATAL_ERROR "${NAME} failed with code ${VALUE}")
    else ()
        message(STATUS "${NAME} complete")
    endif ()
endfunction()

if (ACTION STREQUAL "LLVM")
    # CMake forces CMAKE_CROSSCOMPILING=TRUE whenever a toolchain file sets CMAKE_SYSTEM_NAME,
    # which gates off USE_TOOLCHAIN in LLVMExternalProjectUtils.cmake. That stops the runtimes
    # sub-build (flang-rt, openmp) from picking up the just-built flang and forces it to ask
    # for a host-side Fortran compiler. Drop the toolchain file when host == target so the
    # runtimes can chain onto the freshly built compilers.
    if (UNIX)
        execute_process(COMMAND uname -m OUTPUT_VARIABLE LLVM_HOST_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    elseif (WIN32 AND DEFINED ENV{PROCESSOR_ARCHITECTURE})
        set(LLVM_HOST_ARCH $ENV{PROCESSOR_ARCHITECTURE})
    endif ()
    if (LLVM_HOST_ARCH STREQUAL "AMD64")
        set(LLVM_HOST_ARCH x86_64)
    elseif (LLVM_HOST_ARCH STREQUAL "ARM64")
        set(LLVM_HOST_ARCH arm64)
    endif ()
    set(LLVM_NATIVE_BUILD OFF)
    if (ARCH STREQUAL LLVM_HOST_ARCH)
        set(LLVM_NATIVE_BUILD ON)
    elseif (LLVM_HOST_ARCH STREQUAL "x86_64" AND ARCH STREQUAL "amd64")
        set(LLVM_NATIVE_BUILD ON)
    elseif (LLVM_HOST_ARCH STREQUAL "arm64" AND ARCH STREQUAL "aarch64")
        set(LLVM_NATIVE_BUILD ON)
    endif ()
    message(STATUS "LLVM native build = `${LLVM_NATIVE_BUILD}` (host=${LLVM_HOST_ARCH}, target=${ARCH})")

    # Don't setup vcpkg here
    if (CMAKE_TOOLCHAIN_FILE AND NOT LLVM_NATIVE_BUILD)
        list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    endif ()
    if (LLVM_NATIVE_BUILD)
        # Without the toolchain file CMake autodetects the host compiler. On Linux that is
        # usually gcc; pin clang so the LLVM build matches what the toolchain would have set.
        # On Windows MSVC defaults pick cl.exe, but compiler-rt builtins use GCC-style
        # __attribute__ syntax cl.exe cannot parse, so pin clang-cl which understands both
        # MSVC CLI and GCC attributes. Workflow installs LLVM and exposes clang-cl on PATH.
        if (UNIX AND NOT APPLE)
            list(APPEND BUILD_OPTIONS -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_ASM_COMPILER=clang)
        elseif (WIN32)
            list(APPEND BUILD_OPTIONS -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_ASM_COMPILER=clang-cl)
        endif ()
        find_program(POLYREGION_CCACHE_PROGRAM ccache)
        if (POLYREGION_CCACHE_PROGRAM)
            list(APPEND BUILD_OPTIONS -DCMAKE_C_COMPILER_LAUNCHER=${POLYREGION_CCACHE_PROGRAM})
            list(APPEND BUILD_OPTIONS -DCMAKE_CXX_COMPILER_LAUNCHER=${POLYREGION_CCACHE_PROGRAM})
        endif ()
    endif ()
    message(STATUS "Starting LLVM build...")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            ${BUILD_OPTIONS}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_SYSTEM_PROCESSOR=${ARCH}
            -P build_llvm.cmake

            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "LLVM build")
elseif (ACTION STREQUAL "CONFIGURE")
    setup_vcpkg()
    if (APPLE)
        execute_process(
                COMMAND brew --prefix zlib
                OUTPUT_VARIABLE BREW_ZLIB_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        list(APPEND BUILD_OPTIONS -DZLIB_ROOT=${BREW_ZLIB_PREFIX})
    endif ()
    message(STATUS "Starting configuration...")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            ${BUILD_OPTIONS}
            -B "${BUILD_NAME}"
            -S .
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${LLVM_DIST_DIR}
            -DPOLYREGION_LLVM_DYLIB=${POLYREGION_LLVM_DYLIB}
            -DZLIB_USE_STATIC_LIBS=ON
            -GNinja

            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "Configuration")
elseif (ACTION STREQUAL "BUILD")
    setup_vcpkg()
    message(STATUS "Starting build for target ${TARGET} ...")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            --build ${BUILD_NAME}
            --target ${TARGET}
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "${TARGET} build")
elseif (ACTION STREQUAL "DIST")
    setup_vcpkg()
    message(STATUS "Refreshing LLVM dist into ${LLVM_DIST_DIR}")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            --build ${LLVM_BUILD_DIR}
            --target ${LLVM_DIST_INSTALL_TARGETS}
            -- -k 0
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "LLVM dist install")
    message(STATUS "Installing polyregion dist into ${LLVM_DIST_DIR}")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            --build ${BUILD_NAME}
            --target polyregion-dist
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "polyregion dist build")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            --build ${BUILD_NAME}
            --target install
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "polyregion dist install")
elseif (ACTION STREQUAL "CHECK")
    set(CHECK_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build-dist-check-${ARCH}-${LLVM_VARIANT}")
    message(STATUS "Configuring dist check against ${LLVM_DIST_DIR}")
    file(MAKE_DIRECTORY ${CHECK_BUILD_DIR})
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            -S ${CMAKE_CURRENT_SOURCE_DIR}/dist-check
            -B ${CHECK_BUILD_DIR}
            -DPOLYREGION_DIST=${LLVM_DIST_DIR}
            -DPOLYREGION_ARCH=${ARCH}
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "dist check configure")
    get_filename_component(CMAKE_BIN_DIR "${CMAKE_COMMAND}" DIRECTORY)
    find_program(CTEST_EXE ctest HINTS "${CMAKE_BIN_DIR}" REQUIRED)
    execute_process(
            COMMAND ${CTEST_EXE} -C ${CMAKE_BUILD_TYPE} --output-on-failure
            WORKING_DIRECTORY ${CHECK_BUILD_DIR}
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "dist check ctest")
else ()
    message(FATAL_ERROR "Unknown action: ${ACTION}")
endif ()
