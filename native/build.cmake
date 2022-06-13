
if (UNIX)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH RESULT_VARIABLE SUCCESS OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT SUCCESS EQUAL "0")
        message(FATAL_ERROR "Cannot determine arch, " uname -m" returned ${SUCCESS}")
    endif ()
elseif (WIN32)
    # we only support 64 bit on windows
    set(ARCH x86_64)
else ()
    message(FATAL_ERROR "Unknown arch")
endif ()

string(TOLOWER "build-${CMAKE_HOST_SYSTEM_NAME}-${ARCH}" BUILD_NAME)
message(STATUS "Using build name `${BUILD_NAME}`")

function(check_process_return VALUE NAME)
    if (NOT VALUE EQUAL "0")
        message(FATAL_ERROR "${NAME} failed with code ${VALUE}")
    else ()
        message(STATUS "${NAME} complete")
    endif ()
endfunction()

if (ACTION STREQUAL "CONFIGURE")
    message(STATUS "Starting configuration...")


    message(STATUS "Starting LLVM build...")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            -DCMAKE_CXX_COMPILER=$ENV{CXX}
            -DCMAKE_C_COMPILER=$ENV{CC}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DUSE_LINKER=$ENV{LINKER}
            -DBUILD_SHARED_LIBS=OFF
            -P build_llvm.cmake
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "LLVM build")

    message(STATUS "Starting configuration...")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            -B "${BUILD_NAME}"
            -S .
            -DCMAKE_CXX_COMPILER=$ENV{CXX}
            -DCMAKE_C_COMPILER=$ENV{CC}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DUSE_LINKER=$ENV{LINKER}
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
    message(FATAL_ERROR "Unknown action ${ACTION}")
endif ()
