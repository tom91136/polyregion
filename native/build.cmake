
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

function(check_process_return VALUE NAME)
    if (NOT VALUE EQUAL "0")
        message(FATAL_ERROR "${NAME} failed with code ${VALUE}")
    else ()
        message(STATUS "${NAME} complete")
    endif ()
endfunction()

if (ACTION STREQUAL "LLVM")
    # Don't setup vcpkg here
    if (CMAKE_TOOLCHAIN_FILE)
        list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
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
    message(STATUS "Starting configuration...")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            ${BUILD_OPTIONS}
            -B "${BUILD_NAME}"
            -S .
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${LLVM_DIST_DIR}
            -DPOLYREGION_LLVM_DYLIB=${POLYREGION_LLVM_DYLIB}
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
    message(STATUS "Installing polyregion dist into ${LLVM_DIST_DIR}")
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
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "dist check configure")
    get_filename_component(CMAKE_BIN_DIR "${CMAKE_COMMAND}" DIRECTORY)
    find_program(CTEST_EXE ctest HINTS "${CMAKE_BIN_DIR}" REQUIRED)
    execute_process(
            COMMAND ${CTEST_EXE} --output-on-failure
            WORKING_DIRECTORY ${CHECK_BUILD_DIR}
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "dist check ctest")
else ()
    message(FATAL_ERROR "Unknown action: ${ACTION}")
endif ()
