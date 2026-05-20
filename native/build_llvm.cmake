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

set(LLVM_TARBALL_NAME llvm-project-${LLVM_SRC_VERSION}.src.tar.xz)
set(LLVM_SRC_DIRNAME llvm-project-${LLVM_SRC_VERSION}.src)

set(DOWNLOAD_LLVM OFF)
if (EXISTS ${LLVM_BUILD_DIR}/${LLVM_TARBALL_NAME})
    file(SHA256 ${LLVM_BUILD_DIR}/${LLVM_TARBALL_NAME} EXISTING_HASH)
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
            ${LLVM_BUILD_DIR}/${LLVM_TARBALL_NAME}
            EXPECTED_HASH SHA256=${LLVM_SOURCE_SHA256}
    )
endif ()

if (NOT EXISTS ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}/llvm/CMakeLists.txt)
    message(STATUS "Extracting LLVM source...")
    file(ARCHIVE_EXTRACT INPUT ${LLVM_BUILD_DIR}/${LLVM_TARBALL_NAME} DESTINATION "${LLVM_BUILD_DIR}")
    set(LLVM_PATCHES)
    foreach (DIR ${LLVM_PATCH_DIRS})
        if (IS_DIRECTORY ${DIR})
            file(GLOB _PATCHES_IN_DIR "${DIR}/*.patch")
            list(APPEND LLVM_PATCHES ${_PATCHES_IN_DIR})
        endif ()
    endforeach ()
    if (LLVM_PATCHES)
        list(SORT LLVM_PATCHES)
        find_program(POLYREGION_PATCH_CMD patch)
        find_program(POLYREGION_GIT_CMD git)
        foreach (PATCH ${LLVM_PATCHES})
            message(STATUS "Applying ${PATCH}")
            if (POLYREGION_PATCH_CMD)
                execute_process(
                        COMMAND ${POLYREGION_PATCH_CMD} -p1 --forward --batch -i ${PATCH}
                        WORKING_DIRECTORY ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}
                        RESULT_VARIABLE PATCH_RC)
            elseif (POLYREGION_GIT_CMD)
                execute_process(
                        COMMAND ${POLYREGION_GIT_CMD} apply --whitespace=nowarn -p1 ${PATCH}
                        WORKING_DIRECTORY ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}
                        RESULT_VARIABLE PATCH_RC)
            else ()
                message(FATAL_ERROR "Neither `patch` nor `git` found on PATH; cannot apply ${PATCH}")
            endif ()
            if (NOT PATCH_RC EQUAL 0)
                message(FATAL_ERROR "Failed to apply ${PATCH} (exit ${PATCH_RC})")
            endif ()
        endforeach ()
    endif ()
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

if (POLYREGION_CROSS_FLANG_NEW AND APPLE)
    # XXX otherwise cross binaries report the host triple by default.
    list(APPEND BUILD_OPTIONS
            -DLLVM_DEFAULT_TARGET_TRIPLE=${CMAKE_SYSTEM_PROCESSOR}-apple-darwin
            -DLLVM_HOST_TRIPLE=${CMAKE_SYSTEM_PROCESSOR}-apple-darwin
            -DLLVM_TARGET_ARCH=${CMAKE_SYSTEM_PROCESSOR})
endif ()

if (POLYREGION_CROSS_FLANG_NEW)
    set(_RT_ARGS "-DCMAKE_Fortran_COMPILER=${POLYREGION_CROSS_FLANG_NEW}")
    if (POLYREGION_CROSS_CLANG)
        list(APPEND _RT_ARGS "-DCMAKE_C_COMPILER=${POLYREGION_CROSS_CLANG}")
    endif ()
    if (POLYREGION_CROSS_CLANGXX)
        list(APPEND _RT_ARGS "-DCMAKE_CXX_COMPILER=${POLYREGION_CROSS_CLANGXX}")
    endif ()
    list(APPEND BUILD_OPTIONS "-DRUNTIMES_CMAKE_ARGS=${_RT_ARGS}")
    list(APPEND BUILD_OPTIONS -DFLANG_BUILD_INTRINSIC_MODULES_CROSS=ON)
    # XXX flang/tools/f18 module recipes use COMMAND flang (bare), expecting flang on PATH.
    # Homebrew's llvm ships only flang-new; point PATH at the just-built bin/.
    set(ENV{PATH} "${LLVM_BUILD_DIR}/bin:$ENV{PATH}")
endif ()

execute_process(
        COMMAND ${CMAKE_COMMAND}
        -S ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}/llvm
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

if (POLYREGION_CROSS_FLANG_NEW)
    # XXX openmp's omp_lib.F90 needs iso_c_binding.mod from module_files; force it first.
    execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${LLVM_BUILD_DIR} --target module_files
            WORKING_DIRECTORY ${LLVM_BUILD_DIR}
            RESULT_VARIABLE SUCCESS)
    if (NOT SUCCESS EQUAL "0")
        message(FATAL_ERROR "module_files prebuild failed")
    endif ()
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
        install-mlir-cmake-exports

        install-llvm-headers
        install-clang-headers
        install-mlir-headers

        ${MLIR_INSTALL_TARGETS}
        -- -k 0 # keep going even with error
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE SUCCESS)
if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM build did not succeed")
else ()
    message(STATUS "LLVM build complete!")
endif ()


