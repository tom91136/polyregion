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

# Order by file name (patch number) across all matching dirs, not by dir.
set(_KEYED_PATCHES)
foreach (DIR ${LLVM_PATCH_DIRS})
    if (IS_DIRECTORY ${DIR})
        file(GLOB _PATCHES_IN_DIR "${DIR}/*.patch")
        foreach (_PATCH ${_PATCHES_IN_DIR})
            get_filename_component(_NAME ${_PATCH} NAME)
            list(APPEND _KEYED_PATCHES "${_NAME}|${_PATCH}")
        endforeach ()
    endif ()
endforeach ()
list(SORT _KEYED_PATCHES)
set(LLVM_PATCHES)
foreach (_KEYED ${_KEYED_PATCHES})
    string(REGEX REPLACE "^[^|]*\\|" "" _PATCH "${_KEYED}")
    list(APPEND LLVM_PATCHES ${_PATCH})
endforeach ()

set(_patch_inputs "${LLVM_SOURCE_SHA256}")
foreach (PATCH ${LLVM_PATCHES})
    file(SHA256 ${PATCH} _patch_hash)
    get_filename_component(_patch_name ${PATCH} NAME)
    string(APPEND _patch_inputs "\n${_patch_name}=${_patch_hash}")
endforeach ()
string(SHA256 LLVM_PATCH_STAMP "${_patch_inputs}")
set(LLVM_PATCH_STAMP_FILE ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}.patches.sha256)

set(EXTRACT_LLVM OFF)
if (NOT EXISTS ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}/llvm/CMakeLists.txt)
    set(EXTRACT_LLVM ON)
else ()
    set(_applied_stamp "")
    if (EXISTS ${LLVM_PATCH_STAMP_FILE})
        file(READ ${LLVM_PATCH_STAMP_FILE} _applied_stamp)
        string(STRIP "${_applied_stamp}" _applied_stamp)
    endif ()
    if (NOT "${_applied_stamp}" STREQUAL "${LLVM_PATCH_STAMP}")
        message(WARNING "LLVM source tree patch set is `${_applied_stamp}`, expected `${LLVM_PATCH_STAMP}`; re-extracting ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}")
        file(REMOVE_RECURSE ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME})
        set(EXTRACT_LLVM ON)
    endif ()
endif ()

if (EXTRACT_LLVM)
    message(STATUS "Extracting LLVM source...")
    file(REMOVE ${LLVM_PATCH_STAMP_FILE})
    file(ARCHIVE_EXTRACT INPUT ${LLVM_BUILD_DIR}/${LLVM_TARBALL_NAME} DESTINATION "${LLVM_BUILD_DIR}")
    if (LLVM_PATCHES)
        find_program(POLYREGION_PATCH_CMD patch)
        find_program(POLYREGION_GIT_CMD git)
        foreach (PATCH ${LLVM_PATCHES})
            # A major-wide patch may already be upstream in a later point release; skip it if so.
            if (POLYREGION_PATCH_CMD)
                execute_process(
                        COMMAND ${POLYREGION_PATCH_CMD} -p1 -R --dry-run --batch --force --fuzz=0 -s -i ${PATCH}
                        WORKING_DIRECTORY ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}
                        RESULT_VARIABLE REVERSE_RC OUTPUT_QUIET ERROR_QUIET)
            elseif (POLYREGION_GIT_CMD)
                execute_process(
                        COMMAND ${POLYREGION_GIT_CMD} apply --reverse --check -p1 ${PATCH}
                        WORKING_DIRECTORY ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}
                        RESULT_VARIABLE REVERSE_RC OUTPUT_QUIET ERROR_QUIET)
            else ()
                set(REVERSE_RC 1)
            endif ()
            if (REVERSE_RC EQUAL 0)
                message(STATUS "Skipping ${PATCH} (already applied upstream)")
                continue ()
            endif ()
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
    file(WRITE ${LLVM_PATCH_STAMP_FILE} "${LLVM_PATCH_STAMP}\n")
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

foreach (_compression_var IN ITEMS
        zstd_INCLUDE_DIR
        zstd_LIBRARY
        zstd_STATIC_LIBRARY
        ZLIB_INCLUDE_DIR
        ZLIB_LIBRARY
        ZLIB_LIBRARY_RELEASE)
    if (DEFINED ${_compression_var} AND NOT "${${_compression_var}}" STREQUAL "")
        list(APPEND BUILD_OPTIONS "-D${_compression_var}=${${_compression_var}}")
    endif ()
endforeach ()

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

if (POLYREGION_CROSS_CLANG AND APPLE)
    # XXX otherwise cross binaries report the host triple by default.
    list(APPEND BUILD_OPTIONS
            -DLLVM_DEFAULT_TARGET_TRIPLE=${CMAKE_SYSTEM_PROCESSOR}-apple-darwin
            -DLLVM_HOST_TRIPLE=${CMAKE_SYSTEM_PROCESSOR}-apple-darwin
            -DLLVM_TARGET_ARCH=${CMAKE_SYSTEM_PROCESSOR})
endif ()

if (POLYREGION_CROSS_CLANG)
    if (POLYREGION_LLVM_HAS_FLANG AND NOT POLYREGION_CROSS_FLANG_NEW)
        set(_RT_ARGS "-DCMAKE_C_COMPILER=${LLVM_BUILD_DIR}/bin/clang" "-DCMAKE_CXX_COMPILER=${LLVM_BUILD_DIR}/bin/clang++")
    else ()
        set(_RT_ARGS "-DCMAKE_C_COMPILER=${POLYREGION_CROSS_CLANG}")
        if (POLYREGION_CROSS_CLANGXX)
            list(APPEND _RT_ARGS "-DCMAKE_CXX_COMPILER=${POLYREGION_CROSS_CLANGXX}")
        endif ()
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
        if (LLVM_MAJOR LESS 23)
            list(APPEND BUILD_OPTIONS -DFLANG_BUILD_INTRINSIC_MODULES_CROSS=ON)
        endif ()
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
            # QEMU-user path: target-arch flang via binfmt (intrinsic modules only); bulk C/C++ stays host
            set(ENV{PATH} "${LLVM_BUILD_DIR}/bin:$ENV{PATH}")
            list(APPEND _RT_ARGS "-DCMAKE_Fortran_COMPILER=${LLVM_BUILD_DIR}/bin/flang")
        endif ()
    endif ()

    # XXX escape ; else RUNTIMES_CMAKE_ARGS splats elements 2..N as top-level cmake args
    string(REPLACE ";" "\\;" _RT_ARGS_ESCAPED "${_RT_ARGS}")
    list(APPEND BUILD_OPTIONS "-DRUNTIMES_CMAKE_ARGS=${_RT_ARGS_ESCAPED}")
endif ()

function(polyregion_train_clang clang include_dir work_dir)
    set(_src "${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}/llvm")
    file(GLOB _sources "${_src}/lib/Support/*.cpp")
    set(_ninja "rule cxx\n  command = ${clang} --driver-mode=g++ -std=c++17 -fno-rtti -fno-exceptions -I${_src}/include -I${_src}/../third-party/siphash/include -I${_src}/../libc -I${include_dir} -$opt -c $in -o $out\n")
    foreach (_source ${_sources})
        get_filename_component(_name ${_source} NAME_WE)
        foreach (_opt O0 O3)
            string(APPEND _ninja "build ${_name}.${_opt}.o: cxx ${_source}\n  opt = ${_opt}\n")
        endforeach ()
    endforeach ()
    file(REMOVE_RECURSE ${work_dir})
    file(WRITE ${work_dir}/build.ninja "${_ninja}")
    set(_jobs "")
    if (NOT "$ENV{POLYREGION_LLVM_PARALLEL_COMPILE_JOBS}" STREQUAL "")
        set(_jobs -j "$ENV{POLYREGION_LLVM_PARALLEL_COMPILE_JOBS}")
    endif ()
    execute_process(COMMAND ninja ${_jobs} WORKING_DIRECTORY ${work_dir} COMMAND_ERROR_IS_FATAL ANY)
    file(REMOVE_RECURSE ${work_dir})
endfunction()

set(POLYREGION_LLVM_PGO_BOLT OFF)
if ("$ENV{POLYREGION_LLVM_PGO_BOLT}" STREQUAL "ON")
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" OR POLYREGION_CROSS_CLANG)
        message(FATAL_ERROR "POLYREGION_LLVM_PGO_BOLT needs a native Linux Release build")
    endif ()
    set(POLYREGION_LLVM_PGO_BOLT ON)

    set(_pgo_dir ${LLVM_BUILD_DIR}-pgo)
    set(_profdata ${LLVM_BUILD_DIR}/clang.profdata)
    file(REMOVE_RECURSE ${_pgo_dir})
    execute_process(
            COMMAND ${CMAKE_COMMAND}
            -S ${LLVM_BUILD_DIR}/${LLVM_SRC_DIRNAME}/llvm
            -B ${_pgo_dir}
            -DLLVM_BUILD_INSTRUMENTED=IR
            -DLLVM_TARGETS_TO_BUILD=Native
            -DLLVM_ENABLE_PROJECTS=clang
            -DLLVM_ENABLE_RUNTIMES=
            -DLLVM_DISTRIBUTION_COMPONENTS=
            -C ${CMAKE_CURRENT_BINARY_DIR}/build_llvm_cache.cmake
            ${BUILD_OPTIONS}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -GNinja
            COMMAND_ECHO STDERR
            COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${_pgo_dir} --target clang COMMAND_ERROR_IS_FATAL ANY)

    file(REMOVE_RECURSE ${_pgo_dir}/profiles)
    polyregion_train_clang(${_pgo_dir}/bin/clang ${_pgo_dir}/include ${_pgo_dir}/train)
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --print-prog-name=llvm-profdata
            OUTPUT_VARIABLE _llvm_profdata OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${_llvm_profdata} merge -output=${_profdata} ${_pgo_dir}/profiles COMMAND_ERROR_IS_FATAL ANY)
    file(REMOVE_RECURSE ${_pgo_dir})
    list(APPEND BUILD_OPTIONS -DLLVM_PROFDATA_FILE=${_profdata})
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

# XXX LLVM 23 moved the intrinsic .mod files from flang's module_files into flang-rt, installed
# under the resource dir (lib/clang/<major>/finclude/flang/<triple>) as their own component.
if (LLVM_MAJOR GREATER_EQUAL 23)
    set(FLANG_MODULE_TARGET flang-rt-mod)
    set(FLANG_MODULE_INSTALL_TARGET install-flang-rt-mod)
else ()
    set(FLANG_MODULE_TARGET module_files)
    set(FLANG_MODULE_INSTALL_TARGET tools/flang/tools/f18/install)
endif ()

if (POLYREGION_LLVM_HAS_FLANG)
    set(FLANG_BUILD_TARGETS
            ${FLANG_MODULE_TARGET}
            ${FLANG_MODULE_INSTALL_TARGET}
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
    # XXX openmp's omp_lib.F90 needs iso_c_binding.mod from the intrinsic modules; force it first.
    execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${LLVM_BUILD_DIR} --target ${FLANG_MODULE_TARGET}
            WORKING_DIRECTORY ${LLVM_BUILD_DIR}
            RESULT_VARIABLE SUCCESS)
    if (NOT SUCCESS EQUAL "0")
        message(FATAL_ERROR "${FLANG_MODULE_TARGET} prebuild failed")
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
    list(APPEND _distribution_prebuild_targets ${FLANG_MODULE_TARGET})
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

if (WIN32)
    set(_llvm_ar "${LLVM_DIST_DIR}/bin/llvm-ar.exe")
    set(_support_lib "${LLVM_DIST_DIR}/lib/LLVMSupport.lib")
    set(_rpmalloc_lib "${LLVM_DIST_DIR}/lib/LLVMRpmalloc.lib")
    execute_process(COMMAND ${_llvm_ar} t ${_support_lib} OUTPUT_VARIABLE _support_members RESULT_VARIABLE SUCCESS)
    if (NOT SUCCESS EQUAL "0")
        message(FATAL_ERROR "Cannot list ${_support_lib}")
    endif ()
    string(REGEX MATCH "[^\r\n]*rpmalloc[.]c[.]obj" _rpmalloc_member "${_support_members}")
    if (_rpmalloc_member)
        set(_split_dir "${LLVM_BUILD_DIR}/rpmalloc-split")
        file(REMOVE_RECURSE "${_split_dir}")
        file(MAKE_DIRECTORY "${_split_dir}")
        execute_process(COMMAND ${_llvm_ar} x ${_support_lib} "${_rpmalloc_member}" WORKING_DIRECTORY "${_split_dir}" RESULT_VARIABLE SUCCESS)
        if (NOT SUCCESS EQUAL "0")
            message(FATAL_ERROR "Cannot extract ${_rpmalloc_member}")
        endif ()
        file(REMOVE "${_rpmalloc_lib}")
        execute_process(COMMAND ${_llvm_ar} rcs ${_rpmalloc_lib} rpmalloc.c.obj WORKING_DIRECTORY "${_split_dir}" RESULT_VARIABLE SUCCESS)
        if (NOT SUCCESS EQUAL "0")
            message(FATAL_ERROR "Cannot create ${_rpmalloc_lib}")
        endif ()
        execute_process(COMMAND ${_llvm_ar} d ${_support_lib} "${_rpmalloc_member}" RESULT_VARIABLE SUCCESS)
        if (NOT SUCCESS EQUAL "0")
            message(FATAL_ERROR "Cannot strip ${_rpmalloc_member} from ${_support_lib}")
        endif ()
    elseif (NOT EXISTS ${_rpmalloc_lib})
        message(FATAL_ERROR "No rpmalloc member in ${_support_lib}; LLVM_ENABLE_RPMALLOC is not in effect")
    endif ()
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

if (POLYREGION_LLVM_PGO_BOLT)
    if (LLVM_VARIANT STREQUAL "dylib")
        file(GLOB _bolt_targets "${LLVM_DIST_DIR}/lib/libLLVMpolyregion.so.*" "${LLVM_DIST_DIR}/lib/libclang-cpp.so.*")
    else ()
        set(_bolt_targets "${LLVM_DIST_DIR}/bin/clang-${LLVM_MAJOR}")
    endif ()

    set(_bolt_dir ${LLVM_BUILD_DIR}-bolt)
    file(REMOVE_RECURSE ${_bolt_dir})
    file(MAKE_DIRECTORY ${_bolt_dir})
    foreach (_target ${_bolt_targets})
        get_filename_component(_name ${_target} NAME)
        # XXX BOLT reads .debug_line_str raw; compressed DWARF 5 input fails its string offset assertion
        execute_process(
                COMMAND ${LLVM_DIST_DIR}/bin/llvm-objcopy --decompress-debug-sections ${_target} ${_bolt_dir}/${_name}
                COMMAND_ERROR_IS_FATAL ANY)
        execute_process(
                COMMAND ${LLVM_DIST_DIR}/bin/llvm-bolt ${_bolt_dir}/${_name} -o ${_target}
                -instrument --instrumentation-file-append-pid --instrumentation-file=${_bolt_dir}/${_name}.profile
                COMMAND_ERROR_IS_FATAL ANY)
    endforeach ()

    polyregion_train_clang(${LLVM_DIST_DIR}/bin/clang ${LLVM_BUILD_DIR}/include ${_bolt_dir}/train)

    foreach (_target ${_bolt_targets})
        get_filename_component(_name ${_target} NAME)
        file(GLOB _profiles "${_bolt_dir}/${_name}.profile.*.fdata")
        execute_process(
                COMMAND ${LLVM_DIST_DIR}/bin/merge-fdata -o ${_bolt_dir}/${_name}.fdata ${_profiles}
                COMMAND_ERROR_IS_FATAL ANY)
        execute_process(
                COMMAND ${LLVM_DIST_DIR}/bin/llvm-bolt ${_bolt_dir}/${_name} -o ${_target} -data ${_bolt_dir}/${_name}.fdata
                -reorder-blocks=ext-tsp -reorder-functions=cdsort -split-functions -split-all-cold -split-eh
                -use-gnu-stack -update-debug-sections
                COMMAND_ERROR_IS_FATAL ANY)
        execute_process(
                COMMAND ${LLVM_DIST_DIR}/bin/llvm-objcopy --compress-debug-sections=zlib ${_target}
                COMMAND_ERROR_IS_FATAL ANY)
    endforeach ()
    file(REMOVE_RECURSE ${_bolt_dir})

    file(GLOB _dist_files LIST_DIRECTORIES false "${LLVM_DIST_DIR}/bin/*" "${LLVM_DIST_DIR}/lib/*.so*")
    foreach (_file ${_dist_files})
        file(READ ${_file} _magic LIMIT 4 HEX)
        if (_magic STREQUAL "7f454c46" AND NOT IS_SYMLINK ${_file})
            execute_process(
                    COMMAND ${LLVM_DIST_DIR}/bin/llvm-objcopy -w --remove-section=.rela.* --remove-section=!.rela.dyn --remove-section=!.rela.plt ${_file}
                    COMMAND_ERROR_IS_FATAL ANY)
        endif ()
    endforeach ()
endif ()

message(STATUS "LLVM build complete!")
