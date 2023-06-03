
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
string(TOLOWER "build-${CMAKE_HOST_SYSTEM_NAME}-${ARCH}" BUILD_NAME)

message(STATUS "Architecture = `${ARCH}`")
message(STATUS "Build name   = `${BUILD_NAME}`")

if (UNIX)
    set(COMPILER_NAME "clang")
else ()
    set(COMPILER_NAME "msvc")
endif ()

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/toolchains/${CMAKE_HOST_SYSTEM_NAME}-${COMPILER_NAME}-${ARCH}.cmake")
if (NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
    unset(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Cannot find toolchain file ${CMAKE_TOOLCHAIN_FILE} for ${ARCH}, not using one for build...")
elseif ()
    message(STATUS "Toolchain    = `${CMAKE_TOOLCHAIN_FILE}`")
endif ()

if (CMAKE_SYSROOT)
    if (NOT EXISTS "${CMAKE_SYSROOT}")
        message(FATAL_ERROR "Cannot find sysroot ${CMAKE_SYSROOT} for ${ARCH}")
    endif ()
    message(STATUS "Sysroot      = `${CMAKE_SYSROOT}`")
else ()
    message(STATUS "No sysroot specified, not cross building...")
endif ()

if (CMAKE_SYSROOT)
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
    set(VCPKG_TARGET_TRIPLET ${CMAKE_HOST_SYSTEM_NAME}-${COMPILER_NAME}-${ARCH})
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
else ()
    message(FATAL_ERROR "Unknown action: ${ACTION}")
endif ()
