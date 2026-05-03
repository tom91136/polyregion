include(ProjectConfig.cmake)

if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(FATAL_ERROR "Expecting CMAKE_SYSTEM_PROCESSOR")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Expecting CMAKE_BUILD_TYPE")
endif ()

# LLVM_BUILD_LLVM_DYLIB is forced OFF on MSVC/Windows by LLVM's own cmake logic, so any
# distribution component referring to the LLVM/MLIR/clang-cpp shared libs would fail to
# resolve. Force the static variant on Windows.
if (WIN32)
    set(ENV{POLYREGION_LLVM_DYLIB} OFF)
endif ()

if (DEFINED ENV{POLYREGION_LLVM_DYLIB} AND "$ENV{POLYREGION_LLVM_DYLIB}" STREQUAL "OFF")
    set(LLVM_VARIANT static)
else ()
    set(LLVM_VARIANT dylib)
endif ()

set(LLVM_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-${LLVM_VARIANT})
set(LLVM_DIST_DIR ${CMAKE_CURRENT_BINARY_DIR}/polyregion-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-${LLVM_VARIANT}-dist)

set(DOWNLOAD_LLVM OFF)
if (EXISTS ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz)
    file(SHA256 ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz EXISTING_HASH)
    if (NOT "${EXISTING_HASH}" STREQUAL "${LLVM_SOURCE_SHA256}")
        message(STATUS "LLVM source hash did not match, downloading a fresh copy...")
        set(DOWNLOAD_LLVM ON)
    endif ()
else ()
    set(DOWNLOAD_LLVM ON)
endif ()

if (DOWNLOAD_LLVM)
    message(STATUS "Downloading LLVM source...")
    file(DOWNLOAD
            ${LLVM_SOURCE_URL}
            ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz
            EXPECTED_HASH SHA256=${LLVM_SOURCE_SHA256}
    )
    file(ARCHIVE_EXTRACT INPUT ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src.tar.xz DESTINATION "${LLVM_BUILD_DIR}")
endif ()

if (UNIX AND NOT APPLE)
    list(APPEND BUILD_OPTIONS -DLLVM_ENABLE_LTO=Thin)
endif ()

if (UNIX AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    list(APPEND BUILD_OPTIONS "-DLLVM_USE_SANITIZER=Address\\;Undefined")
    list(APPEND BUILD_OPTIONS "-DBUILD_SHARED_LIBS=ON")
    #    list(APPEND BUILD_OPTIONS "-DLLVM_DYLIB_COMPONENTS=all")
endif ()

if (CMAKE_CXX_COMPILER)
    list(APPEND BUILD_OPTIONS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
endif ()
if (CMAKE_C_COMPILER)
    list(APPEND BUILD_OPTIONS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
endif ()
if (CMAKE_SYSROOT)
    list(APPEND BUILD_OPTIONS -DCMAKE_SYSROOT=${CMAKE_SYSROOT})
endif ()
if (CMAKE_TOOLCHAIN_FILE)
    list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
endif ()
if (USE_LINKER)
    list(APPEND BUILD_OPTIONS -DLLVM_USE_LINKER=${USE_LINKER})
endif ()

file(GLOB_RECURSE CMAKE_CACHE_FILES
        "${LLVM_BUILD_DIR}/**/CMakeFiles/*"
        "${LLVM_BUILD_DIR}/**/CMakeCache.txt"
        "${LLVM_BUILD_DIR}/**/build.ninja"
)
list(LENGTH CMAKE_CACHE_FILES N_CACHE_FILES)

foreach (CACHE_FILE ${CMAKE_CACHE_FILES})
    file(REMOVE "${CACHE_FILE}")
    message(STATUS "Removed: ${CACHE_FILE}")
endforeach ()

message(STATUS "Removed ${N_CACHE_FILES} cache files")

file(REMOVE_RECURSE "${LLVM_BUILD_DIR}/runtimes")
file(REMOVE_RECURSE "${LLVM_BUILD_DIR}/projects")

if (UNIX AND NOT APPLE)
    # Set this explicitly for Linux otherwise compiler-rt builds using the host triplet
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
        list(APPEND BUILD_OPTIONS -DLLVM_HOST_TRIPLE=${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnueabihf)
    else ()
        list(APPEND BUILD_OPTIONS -DLLVM_HOST_TRIPLE=${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnu)
    endif ()
endif ()

execute_process(
        COMMAND ${CMAKE_COMMAND}
        -S ${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src/llvm
        -B ${LLVM_BUILD_DIR}
        -C ${CMAKE_CURRENT_BINARY_DIR}/build_llvm_cache.cmake
        --fresh
        ${BUILD_OPTIONS}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${LLVM_DIST_DIR}
        -DCMAKE_SKIP_RPATH=OFF # keep the rpath prefix otherwise our distribution may find the system libLLVM.so
        -DCMAKE_VERBOSE_MAKEFILE=ON
        -GNinja

        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        COMMAND_ECHO STDERR
        RESULT_VARIABLE SUCCESS)

if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM configure did not succeed")
else ()
    message(STATUS "LLVM configuration complete, starting build...")
endif ()

set(FLANG_BUILD_TARGETS
        module_files
        tools/flang/tools/f18/install
        install-flang-cmake-exports
        install-flang-libraries
        install-flang-headers)

# install-MLIR only exists when MLIR is built as a dylib component.
if (NOT DEFINED ENV{POLYREGION_LLVM_DYLIB} OR NOT "$ENV{POLYREGION_LLVM_DYLIB}" STREQUAL "OFF")
    set(MLIR_INSTALL_TARGETS install-MLIR)
endif ()

execute_process(
        COMMAND
        ${CMAKE_COMMAND} -E env ASAN_OPTIONS=detect_leaks=0 --
        ${CMAKE_COMMAND}
        --build ${LLVM_BUILD_DIR}
        --target
        install-distribution

        ${FLANG_BUILD_TARGETS}

        install-cmake-exports
        install-clang-cmake-exports
        install-lld-cmake-exports

        install-llvm-headers
        install-clang-headers

        ${MLIR_INSTALL_TARGETS}
        -- -k 0 # keep going even with error
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE SUCCESS)
if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM build did not succeed")
else ()
    message(STATUS "LLVM build complete!")
endif ()


