include(ProjectConfig.cmake)

if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(FATAL_ERROR "Expecting CMAKE_SYSTEM_PROCESSOR")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Expecting CMAKE_BUILD_TYPE")
endif ()

# XXX upstream flang refuses 32-bit targets (flang/CMakeLists.txt:73); skip flang + derived
set(POLYREGION_LLVM_HAS_FLANG TRUE)
if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm")
    set(POLYREGION_LLVM_HAS_FLANG FALSE)
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

set(LLVM_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/out/llvm-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-${LLVM_VARIANT})
set(LLVM_DIST_DIR ${CMAKE_CURRENT_BINARY_DIR}/out/polyregion-${CMAKE_BUILD_TYPE}-${CMAKE_SYSTEM_PROCESSOR}-${LLVM_VARIANT}-dist)

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
    # XXX LLVM_HOST_TRIPLE must match clang's normalised form or find_compiler_rt_library misses builtins
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL arm)
        set(_llvm_host_triple arm-unknown-linux-gnueabihf)
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL ppc64le)
        set(_llvm_host_triple powerpc64le-unknown-linux-gnu)
    else ()
        set(_llvm_host_triple ${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnu)
    endif ()
    list(APPEND BUILD_OPTIONS -DLLVM_HOST_TRIPLE=${_llvm_host_triple})
endif ()

if (POLYREGION_CROSS_FLANG_NEW AND APPLE)
    # XXX otherwise cross binaries report the host triple by default.
    list(APPEND BUILD_OPTIONS
            -DLLVM_DEFAULT_TARGET_TRIPLE=${CMAKE_SYSTEM_PROCESSOR}-apple-darwin
            -DLLVM_HOST_TRIPLE=${CMAKE_SYSTEM_PROCESSOR}-apple-darwin
            -DLLVM_TARGET_ARCH=${CMAKE_SYSTEM_PROCESSOR})
endif ()

if (POLYREGION_CROSS_CLANG)
    set(_RT_ARGS "-DCMAKE_C_COMPILER=${POLYREGION_CROSS_CLANG}")
    if (POLYREGION_CROSS_CLANGXX)
        list(APPEND _RT_ARGS "-DCMAKE_CXX_COMPILER=${POLYREGION_CROSS_CLANGXX}")
    endif ()
    if (CMAKE_TOOLCHAIN_FILE)
        list(APPEND _RT_ARGS "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
    endif ()
    if (CMAKE_SYSROOT)
        list(APPEND _RT_ARGS "-DCMAKE_SYSROOT=${CMAKE_SYSROOT}")
    endif ()
    # XXX LLVMExternalProjectUtils clobbers LINKER_FLAGS_INIT, appends LINKER_FLAGS; route lld or ld.bfd rejects cross ELFs
    foreach (_kind IN ITEMS EXE SHARED MODULE)
        list(APPEND _RT_ARGS "-DCMAKE_${_kind}_LINKER_FLAGS=-fuse-ld=lld")
    endforeach ()

    if (POLYREGION_LLVM_HAS_FLANG)
        list(APPEND BUILD_OPTIONS -DFLANG_BUILD_INTRINSIC_MODULES_CROSS=ON)
        if (POLYREGION_CROSS_FLANG_NEW)
            # Host-native flang: shim injects --target= (or -triple under -fc1, which bypasses driver)
            set(_cross_shim_dir "${LLVM_BUILD_DIR}/cross-shims")
            file(MAKE_DIRECTORY "${_cross_shim_dir}")
            set(_target_triple "${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
            file(WRITE "${_cross_shim_dir}/flang"
                "#!/bin/sh\nfor a in \"$@\"; do\n  if [ \"$a\" = \"-fc1\" ]; then\n    exec \"${POLYREGION_CROSS_FLANG_NEW}\" \"$@\" -triple ${_target_triple}\n  fi\ndone\nexec \"${POLYREGION_CROSS_FLANG_NEW}\" --target=${_target_triple} \"$@\"\n")
            file(CHMOD "${_cross_shim_dir}/flang" PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
            set(ENV{PATH} "${_cross_shim_dir}:$ENV{PATH}")
            list(APPEND _RT_ARGS "-DCMAKE_Fortran_COMPILER=${POLYREGION_CROSS_FLANG_NEW}")
            list(APPEND _RT_ARGS "-DLIBOMP_FORTRAN_MODULES_COMPILER=${_cross_shim_dir}/flang")
            # XXX x86_64 flang lacks REAL(16); skip R16 or intrinsic modules trip the host front-end
            list(APPEND BUILD_OPTIONS -DHAVE_LDBL_MANT_DIG_113=OFF)
        else ()
            # QEMU-user path: target-arch flang via binfmt (module_files only); bulk C/C++ stays host
            set(ENV{PATH} "${LLVM_BUILD_DIR}/bin:$ENV{PATH}")
            list(APPEND _RT_ARGS "-DCMAKE_Fortran_COMPILER=${LLVM_BUILD_DIR}/bin/flang")
        endif ()
    endif ()

    # XXX escape ; else RUNTIMES_CMAKE_ARGS splats elements 2..N as top-level cmake args
    string(REPLACE ";" "\\;" _RT_ARGS_ESCAPED "${_RT_ARGS}")
    list(APPEND BUILD_OPTIONS "-DRUNTIMES_CMAKE_ARGS=${_RT_ARGS_ESCAPED}")
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

if (POLYREGION_LLVM_HAS_FLANG)
    set(FLANG_BUILD_TARGETS
            module_files
            tools/flang/tools/f18/install
            install-flang-cmake-exports
            install-flang-libraries
            install-flang-headers)
else ()
    set(FLANG_BUILD_TARGETS "")
endif ()

if (POLYREGION_LLVM_HAS_FLANG)
    list(APPEND MLIR_INSTALL_TARGETS install-mlir-cmake-exports install-mlir-headers)
    # install-MLIR only exists when MLIR is built as a dylib component.
    if (NOT DEFINED ENV{POLYREGION_LLVM_DYLIB} OR NOT "$ENV{POLYREGION_LLVM_DYLIB}" STREQUAL "OFF")
        list(APPEND MLIR_INSTALL_TARGETS install-MLIR)
    endif ()
endif ()

if (POLYREGION_CROSS_CLANG AND POLYREGION_LLVM_HAS_FLANG)
    # XXX openmp's omp_lib.F90 needs iso_c_binding.mod from module_files; force it first.
    execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${LLVM_BUILD_DIR} --target module_files
            WORKING_DIRECTORY ${LLVM_BUILD_DIR}
            RESULT_VARIABLE SUCCESS)
    if (NOT SUCCESS EQUAL "0")
        message(FATAL_ERROR "module_files prebuild failed")
    endif ()
    # XXX runtimes hard-codes LIBOMP_FORTRAN_MODULES_COMPILER=bin/flang; swap shim, restore before install
    if (POLYREGION_CROSS_FLANG_NEW AND EXISTS "${LLVM_BUILD_DIR}/bin/flang-${LLVM_MAJOR}" AND EXISTS "${_cross_shim_dir}/flang")
        file(REMOVE "${LLVM_BUILD_DIR}/bin/flang")
        file(CREATE_LINK "${_cross_shim_dir}/flang" "${LLVM_BUILD_DIR}/bin/flang" SYMBOLIC)
    endif ()
endif ()

# XXX install-flang-headers has no build deps; pre-build distribution so tablegen .inc files exist before install
set(_distribution_prebuild_targets distribution)
if (POLYREGION_LLVM_HAS_FLANG)
    list(APPEND _distribution_prebuild_targets module_files)
endif ()
execute_process(
        COMMAND
        ${CMAKE_COMMAND} -E env ASAN_OPTIONS=detect_leaks=0 --
        ${CMAKE_COMMAND}
        --build ${LLVM_BUILD_DIR}
        --target
        ${_distribution_prebuild_targets}
        -- -k 0
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE SUCCESS)
if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM distribution build did not succeed")
endif ()

if (POLYREGION_CROSS_FLANG_NEW AND EXISTS "${LLVM_BUILD_DIR}/bin/flang-${LLVM_MAJOR}")
    file(REMOVE "${LLVM_BUILD_DIR}/bin/flang")
    file(CREATE_LINK "flang-${LLVM_MAJOR}" "${LLVM_BUILD_DIR}/bin/flang" SYMBOLIC)
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
        install-lld-headers

        ${MLIR_INSTALL_TARGETS}
        -- -k 0 # keep going even with error
        WORKING_DIRECTORY ${LLVM_BUILD_DIR}
        RESULT_VARIABLE SUCCESS)
if (NOT SUCCESS EQUAL "0")
    message(FATAL_ERROR "LLVM install did not succeed")
endif ()

# XXX flang-headers install matches only *.inc from the build tree; copy the generated config.h ourselves.
if (POLYREGION_LLVM_HAS_FLANG)
    file(COPY "${LLVM_BUILD_DIR}/tools/flang/include/flang/Config/config.h"
            DESTINATION "${LLVM_DIST_DIR}/include/flang/Config")
endif ()

# XXX flang-rt install rule tagged Unspecified; install-distribution skips, drive script directly
set(_flang_rt_install "${LLVM_BUILD_DIR}/runtimes/runtimes-bins/flang-rt/lib/runtime/cmake_install.cmake")
if (EXISTS "${_flang_rt_install}")
    execute_process(
            COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_COMPONENT= -P "${_flang_rt_install}"
            RESULT_VARIABLE SUCCESS)
    if (NOT SUCCESS EQUAL "0")
        message(FATAL_ERROR "flang-rt install did not succeed")
    endif ()
endif ()

# XXX libcxx/libcxxabi static archives share COMPONENT cxx/cxxabi with the shared lib target;
# install-distribution skips them too, drive their install scripts directly like flang-rt above.
foreach (_lib cxx cxxabi)
    foreach (_install_script
            "${LLVM_BUILD_DIR}/runtimes/runtimes-bins/lib${_lib}/src/cmake_install.cmake"
            "${LLVM_BUILD_DIR}/lib${_lib}/src/cmake_install.cmake")
        if (EXISTS "${_install_script}")
            execute_process(
                    COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_COMPONENT= -P "${_install_script}"
                    RESULT_VARIABLE SUCCESS)
            if (NOT SUCCESS EQUAL "0")
                message(FATAL_ERROR "lib${_lib} static archive install did not succeed")
            endif ()
        endif ()
    endforeach ()
endforeach ()

# XXX compiler-rt's sanitizer archives aren't covered by install-distribution's "runtimes"
# component either (only builtins is); drive its install script directly, same as above.
set(_compiler_rt_install "${LLVM_BUILD_DIR}/runtimes/runtimes-bins/compiler-rt/cmake_install.cmake")
if (EXISTS "${_compiler_rt_install}")
    execute_process(
            COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_COMPONENT= -P "${_compiler_rt_install}"
            RESULT_VARIABLE SUCCESS)
    if (NOT SUCCESS EQUAL "0")
        message(FATAL_ERROR "compiler-rt install did not succeed")
    endif ()
endif ()

# XXX POLYREGION_FUSED_DRIVER compiles a handful of LLVM driver .cpp files directly into
# polycpp/polyfc. CI caches only the dist, so on cache hit the unpacked LLVM source is gone
# and the fused-driver build can't find these. Stage the exact files we need.
set(FUSED_DRIVER_STAGE "${LLVM_DIST_DIR}/share/polyregion-fused-driver")
foreach (F driver.cpp cc1_main.cpp cc1as_main.cpp cc1gen_reproducer_main.cpp)
    file(COPY "${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}/clang/tools/driver/${F}"
            DESTINATION "${FUSED_DRIVER_STAGE}/clang-tools-driver")
endforeach ()
file(COPY "${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}/flang/tools/flang-driver/fc1_main.cpp"
        DESTINATION "${FUSED_DRIVER_STAGE}/flang-tools-flang-driver")

message(STATUS "LLVM build complete!")

