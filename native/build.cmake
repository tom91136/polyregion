
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

if (WIN32)
    string(TOUPPER "${ARCH}" _ARCH_UPPER)
    if (_ARCH_UPPER STREQUAL "X86_64" OR _ARCH_UPPER STREQUAL "AMD64")
        set(ARCH amd64)
    elseif (_ARCH_UPPER STREQUAL "AARCH64" OR _ARCH_UPPER STREQUAL "ARM64")
        set(ARCH arm64)
    endif ()
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

set(BUILD_NAME_SUFFIX "")
set(DIST_SUFFIX "")
if (DEFINED ENV{POLYREGION_FUSED_DRIVER} AND "$ENV{POLYREGION_FUSED_DRIVER}" STREQUAL "ON" AND NOT WIN32)
    set(BUILD_NAME_SUFFIX "-fused")
endif ()
if (DEFINED ENV{POLYREGION_ASAN} AND "$ENV{POLYREGION_ASAN}" STREQUAL "ON")
    set(BUILD_NAME_SUFFIX "${BUILD_NAME_SUFFIX}-asan")
    set(DIST_SUFFIX "-asan")
endif ()
string(TOLOWER "out/build-${CMAKE_HOST_SYSTEM_NAME}-${ARCH}-${LLVM_VARIANT}${BUILD_NAME_SUFFIX}" BUILD_NAME)
set(LLVM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out/llvm-${CMAKE_BUILD_TYPE}-${ARCH}-${LLVM_VARIANT}")
set(LLVM_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out/polyregion-${CMAKE_BUILD_TYPE}-${ARCH}-${LLVM_VARIANT}${DIST_SUFFIX}-dist")

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

# XXX CMake makes absolute RUNPATH entries sysroot-relative, so a `/` sysroot strips the leading slash and the
# loader falls through to the system # libLLVM.so instead of our pinned build. Drop the trivial value.
if (CMAKE_SYSROOT STREQUAL "/")
    set(CMAKE_SYSROOT "")
    unset(ENV{CMAKE_SYSROOT})
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
    set(ENV{CMAKE_SYSROOT} ${CMAKE_SYSROOT})
    list(APPEND BUILD_OPTIONS -DCMAKE_SYSROOT=${CMAKE_SYSROOT})
endif ()

foreach (_var POLYC_JS_ENGINE POLYC_PASS_BUNDLE POLYREGION_FUSED_DRIVER POLYREGION_ASAN)
    if (DEFINED ENV{${_var}} AND NOT EXISTS "${BUILD_NAME}/CMakeCache.txt")
        list(APPEND BUILD_OPTIONS -D${_var}=$ENV{${_var}})
    endif ()
endforeach ()

macro(setup_vcpkg)
    if (DEFINED ENV{RUNVCPKG_VCPKG_ROOT})
        set(VCPKG_ROOT $ENV{RUNVCPKG_VCPKG_ROOT})
        message(STATUS "vcpkg root   = `${VCPKG_ROOT}` (from RUNVCPKG_VCPKG_ROOT)")
    elseif (DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_ROOT $ENV{VCPKG_ROOT})
        message(STATUS "vcpkg root   = `${VCPKG_ROOT}`")
    else ()
        message(FATAL_ERROR "Environment VCPKG_ROOT not defined")
    endif ()

    list(APPEND BUILD_OPTIONS -DVCPKG_ROOT=${VCPKG_ROOT})
    if (DEFINED ENV{VCPKG_INSTALLED_DIR})
        list(APPEND BUILD_OPTIONS -DVCPKG_INSTALLED_DIR=$ENV{VCPKG_INSTALLED_DIR})
        list(APPEND BUILD_OPTIONS -DVCPKG_MANIFEST_INSTALL=OFF)
    endif ()
    if (CMAKE_TOOLCHAIN_FILE)
        list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
        list(APPEND BUILD_OPTIONS -DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    else ()
        list(APPEND BUILD_OPTIONS -DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
    endif ()
    if (WIN32)
        # XXX static triplet pins vcpkg deps to /MT; the default x64-windows triplet uses /MD and trips LNK2038.
        if (ARCH STREQUAL "amd64")
            set(VCPKG_TARGET_TRIPLET x64-${CMAKE_HOST_SYSTEM_NAME}-static)
        elseif (ARCH STREQUAL "arm64")
            set(VCPKG_TARGET_TRIPLET arm64-${CMAKE_HOST_SYSTEM_NAME}-static)
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
    else ()
        # XXX runtimes USE_TOOLCHAIN off under CMAKE_CROSSCOMPILING; pin host cross tools (FLANG_NEW optional)
        foreach (_pair IN ITEMS "CLANG;clang" "CLANGXX;clang++")
            list(GET _pair 0 _var)
            list(GET _pair 1 _bin)
            set(_val "$ENV{POLYREGION_CROSS_${_var}}")
            if (NOT _val)
                set(_val "${LLVM_BUILD_DIR}/bin/${_bin}")
            endif ()
            list(APPEND BUILD_OPTIONS -DPOLYREGION_CROSS_${_var}=${_val})
        endforeach ()
        if (DEFINED ENV{POLYREGION_CROSS_FLANG_NEW})
            list(APPEND BUILD_OPTIONS -DPOLYREGION_CROSS_FLANG_NEW=$ENV{POLYREGION_CROSS_FLANG_NEW})
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
elseif (ACTION STREQUAL "DEVICE_LIBS")
    include(ProjectConfig.cmake)
    set(DEVICE_LIBS_MARKER "${POLYREGION_DEVICE_LIBS_DIR}/.staged")
    set(_cuda_versions "")
    foreach (_entry IN LISTS POLYREGION_CUDA_LIBDEVICE)
        string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _v "${_entry}")
        list(APPEND _cuda_versions "${_v}")
    endforeach ()
    set(DEVICE_LIBS_TAG "cuda=${_cuda_versions};rocm=${POLYREGION_ROCM_TAG}")
    set(_existing_tag "")
    if (EXISTS "${DEVICE_LIBS_MARKER}")
        file(READ "${DEVICE_LIBS_MARKER}" _existing_tag)
    endif ()
    if (_existing_tag STREQUAL "${DEVICE_LIBS_TAG}")
        message(STATUS "Device libs up to date at ${POLYREGION_DEVICE_LIBS_DIR}")
    else ()
        if (EXISTS "${POLYREGION_DEVICE_LIBS_DIR}")
            file(REMOVE_RECURSE "${POLYREGION_DEVICE_LIBS_DIR}")
        endif ()
        file(MAKE_DIRECTORY "${POLYREGION_DEVICE_LIBS_DIR}")
        message(STATUS "Staging device libs into ${POLYREGION_DEVICE_LIBS_DIR}")
        foreach (_ver IN LISTS _cuda_versions)
            execute_process(
                    COMMAND ${CMAKE_COMMAND}
                    -DOUT_DIR=${POLYREGION_DEVICE_LIBS_DIR}
                    -DCUDA_VERSION=${_ver}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_cuda_dist.cmake
                    COMMAND_ECHO STDERR
                    RESULT_VARIABLE SUCCESS)
            check_process_return(${SUCCESS} "libdevice prep for CUDA ${_ver}")
        endforeach ()
        execute_process(
                COMMAND ${CMAKE_COMMAND}
                -DOUT_DIR=${POLYREGION_DEVICE_LIBS_DIR}
                -DROCM_TAG=${POLYREGION_ROCM_TAG}
                -DPOLYREGION_LLVM_BIN_HINT=${LLVM_DIST_DIR}/bin
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_amdgpu_dist.cmake
                COMMAND_ECHO STDERR
                RESULT_VARIABLE SUCCESS)
        check_process_return(${SUCCESS} "AMDGPU device-libs prep")
        file(WRITE "${DEVICE_LIBS_MARKER}" "${DEVICE_LIBS_TAG}")
        file(REMOVE_RECURSE "${POLYREGION_DEVICE_LIBS_DIR}/.staging")
        message(STATUS "Device libs staged at ${POLYREGION_DEVICE_LIBS_DIR}")
    endif ()
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
elseif (ACTION STREQUAL "DIST_TEST")
    setup_vcpkg()
    set(TEST_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out/polyregion-test-${CMAKE_BUILD_TYPE}-${ARCH}-${LLVM_VARIANT}${DIST_SUFFIX}-dist")
    message(STATUS "Installing polyregion test dist into ${TEST_DIST_DIR}")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            --build ${BUILD_NAME}
            --target polyregion-test-dist
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "polyregion test dist build")
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            --install ${BUILD_NAME}
            --component test-dist
            --prefix ${TEST_DIST_DIR}
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "polyregion test dist install")
elseif (ACTION STREQUAL "CHECK")
    set(CHECK_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/out/build-dist-check-${ARCH}-${LLVM_VARIANT}")
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
            COMMAND ${CTEST_EXE} -C ${CMAKE_BUILD_TYPE} --output-on-failure --output-junit dist-check-results.xml
            WORKING_DIRECTORY ${CHECK_BUILD_DIR}
            COMMAND_ECHO STDERR
            RESULT_VARIABLE SUCCESS)
    check_process_return(${SUCCESS} "dist check ctest")
else ()
    message(FATAL_ERROR "Unknown action: ${ACTION}")
endif ()
