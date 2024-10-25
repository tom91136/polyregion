include(ProjectConfig.cmake)

if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(FATAL_ERROR "Expecting CMAKE_SYSTEM_PROCESSOR")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Expecting CMAKE_BUILD_TYPE")
endif ()

set(LLVM_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR})
set(LLVM_DIST_DIR ${CMAKE_CURRENT_BINARY_DIR}/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-dist)

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

### Patches ###

string(TIMESTAMP CURRENT_TIME UTC)

# Remove benchmarks, see https://github.com/llvm/llvm-project/issues/54941
file(WRITE ${LLVM_BUILD_DIR}/third-party/benchmark/CMakeLists.txt "")

# Patch -nostdinc++ detection failure in compiler-rt, it's unclear why this only happens in that particular module
set(RUNTIME_CMAKELIST_PATH "${LLVM_BUILD_DIR}/llvm-project-${LLVM_SRC_VERSION}.src/runtimes/CMakeLists.txt")
file(READ "${RUNTIME_CMAKELIST_PATH}" RUNTIME_CMAKELIST_CONTENT)

set(NOSTDINCXX_FRAGMENT_REGEX "check_cxx_compiler_flag\\(-nostdinc\\+\\+ CXX_SUPPORTS_NOSTDINCXX_FLAG\\)
if \\(CXX_SUPPORTS_NOSTDINCXX_FLAG\\)")

if (RUNTIME_CMAKELIST_CONTENT MATCHES "${NOSTDINCXX_FRAGMENT_REGEX}")
    string(REGEX REPLACE "${NOSTDINCXX_FRAGMENT_REGEX}" "# CXX_SUPPORTS_NOSTDINCXX_FLAG patched at ${CURRENT_TIME}
check_cxx_compiler_flag(-nostdinc++ CXX_SUPPORTS_NOSTDINCXX_FLAG)
set(CXX_SUPPORTS_NOSTDINCXX_FLAG ON)
if (CXX_SUPPORTS_NOSTDINCXX_FLAG)" RUNTIME_CMAKELIST_CONTENT "${RUNTIME_CMAKELIST_CONTENT}")
    file(WRITE "${RUNTIME_CMAKELIST_PATH}" "${RUNTIME_CMAKELIST_CONTENT}")
    message(STATUS "Patched CXX_SUPPORTS_NOSTDINCXX_FLAG in ${RUNTIME_CMAKELIST_PATH}")
endif ()

### End patches ###

if (UNIX AND NOT APPLE)
    list(APPEND BUILD_OPTIONS -DLLVM_ENABLE_LTO=Thin)
endif ()

if (UNIX AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    list(APPEND BUILD_OPTIONS "-DLLVM_USE_SANITIZER=Address\\;Undefined")
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

execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${LLVM_BUILD_DIR}
        --target
        lldCommon
        lldELF
        LLVMExecutionEngine
        install-distribution
        install-cmake-exports
        install-clang-cmake-exports
        install-lld-cmake-exports
        -- -k 0 # keep going even with error
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE SUCCESS)

if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM build did not succeed")
else ()
    message(STATUS "LLVM build complete!")
endif ()


