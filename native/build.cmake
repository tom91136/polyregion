
if (UNIX)
    if (NOT ARCH)
        message(STATUS "ARCH not set, detecting host arch")
        execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH RESULT_VARIABLE SUCCESS OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT SUCCESS EQUAL "0")
            message(FATAL_ERROR "Cannot determine host arch, `uname -m` returned ${SUCCESS}")
        endif ()
    endif ()
elseif (WIN32)
    # FIXME W32
    # we only support 64 bit on windows
    if (CMAKE_SYSTEM_PROCESSOR)
        message(STATUS "Setting CMAKE_SYSTEM_PROCESSOR is not supported on Windows")
    endif ()
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
else ()
    message(FATAL_ERROR "Unknown platform (not Unix-like or Windows)")
endif ()


string(TOLOWER "build-${CMAKE_HOST_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}" BUILD_NAME)

message(STATUS "Architecture = `${ARCH}`")
message(STATUS "Build name   = `${BUILD_NAME}`")
if (CMAKE_SYSROOT)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/toolchain_${CMAKE_HOST_SYSTEM_NAME}_clang_${ARCH}.cmake")
    if (NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
        message(FATAL_ERROR "Cannot find toolchain ${CMAKE_TOOLCHAIN_FILE} for ${ARCH}")
    endif ()
    if (NOT EXISTS "${CMAKE_SYSROOT}")
        message(FATAL_ERROR "Cannot find sysroot ${CMAKE_SYSROOT} for ${ARCH}")
    endif ()
    message(STATUS "Toolchain    = `${CMAKE_TOOLCHAIN_FILE}`")
    message(STATUS "Sysroot      = `${CMAKE_SYSROOT}`")
else ()
    message(STATUS "No sysroot specified, not cross building...")
endif ()


function(check_process_return VALUE NAME)
    if (NOT VALUE EQUAL "0")
        message(FATAL_ERROR "${NAME} failed with code ${VALUE}")
    else ()
        message(STATUS "${NAME} complete")
    endif ()
endfunction()

if (CMAKE_SYSROOT)
    list(APPEND BUILD_OPTIONS -DCMAKE_SYSROOT=${CMAKE_SYSROOT})
endif ()
if (CMAKE_TOOLCHAIN_FILE)
    list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
endif ()


if (ACTION STREQUAL "LLVM")
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
