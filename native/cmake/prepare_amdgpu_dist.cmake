cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED ROCM_TAG)
  set(ROCM_TAG "rocm-7.2.3")
endif()
if(NOT DEFINED OUT_DIR)
  set(OUT_DIR "${CMAKE_CURRENT_LIST_DIR}/dist")
endif()

if(NOT DEFINED POLYREGION_LLVM_BIN_HINT)
  set(POLYREGION_LLVM_BIN_HINT "")
endif()

function(_find_tool var)
  if(DEFINED ${var} AND ${var})
    return()
  endif()
  find_program(${var} NAMES ${ARGN}
               HINTS "${POLYREGION_LLVM_BIN_HINT}" NO_DEFAULT_PATH)
  if(NOT ${var})
    find_program(${var} NAMES ${ARGN})
  endif()
  if(NOT ${var})
    list(GET ARGN 0 _primary)
    message(FATAL_ERROR "${_primary} not found; set -D ${var}=...")
  endif()
endfunction()

_find_tool(CLANG_BIN     clang clang-22 clang-21 clang-20 clang-19)
_find_tool(LLVM_LINK_BIN llvm-link)
_find_tool(OPT_BIN       opt)
_find_tool(LLVM_NM_BIN   llvm-nm)

find_package(Git REQUIRED)

set(STAGING "${OUT_DIR}/.staging/amdgpu/${ROCM_TAG}")
set(SRC_DIR "${STAGING}/src")
set(BLD_DIR "${STAGING}/build")
file(MAKE_DIRECTORY "${STAGING}" "${OUT_DIR}")

# Sparse-clone keeps the fetch under ~4 MB by skipping LLVM proper.
set(checkout_marker "${SRC_DIR}/.checkout-complete")
if(NOT EXISTS "${checkout_marker}")
  file(REMOVE_RECURSE "${SRC_DIR}")
  message(STATUS "sparse-cloning ROCm/llvm-project @ ${ROCM_TAG} (amd/device-libs only)")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --filter=blob:none --sparse
            --branch "${ROCM_TAG}"
            https://github.com/ROCm/llvm-project.git "${SRC_DIR}"
    RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "git clone failed (${rc}); tag ${ROCM_TAG} may not exist")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" sparse-checkout set amd/device-libs
    WORKING_DIRECTORY "${SRC_DIR}"
    RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "git sparse-checkout failed (${rc})")
  endif()
  file(TOUCH "${checkout_marker}")
endif()

set(DEVLIBS_ROOT "${SRC_DIR}/amd/device-libs")
if(NOT EXISTS "${DEVLIBS_ROOT}/ocml/CMakeLists.txt")
  message(FATAL_ERROR "expected amd/device-libs/ocml/ under ${SRC_DIR}")
endif()

# Mirrors amd/device-libs/cmake/OCL.cmake @ rocm-7.2.3. The -mcpu pin below
# is the non-obvious bit; everything else is verbatim upstream.
set(CLANG_OCL_FLAGS
  -fcolor-diagnostics
  -Werror -Wno-error=atomic-alignment
  -x cl
  -Xclang -cl-std=CL2.0
  -target amdgcn-amd-amdhsa
  -fvisibility=hidden
  -fomit-frame-pointer
  -Xclang -finclude-default-header
  -Xclang -fexperimental-strict-floating-point
  -Xclang -fdenormal-fp-math=dynamic
  -nogpulib -cl-no-stdinc
  -Xclang -mcode-object-version=none
  # Pin a concrete subtarget so amdgcn builtins (cube/image/lerp/...) clear
  # LLVM 22's target-feature gates. The chosen cpu acts only as a "feature
  # superset"; bitcode is target-cpu-tagged but consumer codegen overrides.
  # gfx900 (Vega) has the broadest feature mix among non-experimental archs.
  -mcpu=gfx900
  -emit-llvm)

set(INC_FLAGS
  "-I${DEVLIBS_ROOT}/irif/inc"
  "-I${DEVLIBS_ROOT}/oclc/inc")

# Every defined symbol matching ${public_prefix} stays external; everything
# else gets internalised and DCEd if unreachable. An exact allow-list would
# couple this script to polyc's per-op codegen choices in llvm_amdgpu.cpp.
function(build_bc_lib name comp_src_dir public_prefix)
  set(BLD_COMP "${BLD_DIR}/${name}")
  file(MAKE_DIRECTORY "${BLD_COMP}")
  file(GLOB cl_srcs "${comp_src_dir}/*.cl")
  list(LENGTH cl_srcs n_cl)
  if(n_cl EQUAL 0)
    message(FATAL_ERROR "no .cl files in ${comp_src_dir}")
  endif()
  message(STATUS "compiling ${name}: ${n_cl} .cl files")

  set(comp_inc
    "-I${comp_src_dir}/../inc"
    "-I${comp_src_dir}")

  set(bc_outputs)
  foreach(src IN LISTS cl_srcs)
    get_filename_component(stem "${src}" NAME_WE)
    set(obj "${BLD_COMP}/${stem}.bc")
    # Match upstream's set_source_files_properties for these two specials.
    set(per_file_flags)
    if(stem STREQUAL "native_logF" OR stem STREQUAL "native_expF")
      list(APPEND per_file_flags -fapprox-func)
    elseif(stem STREQUAL "sqrtF")
      list(APPEND per_file_flags -cl-fp32-correctly-rounded-divide-sqrt)
    endif()
    execute_process(
      COMMAND "${CLANG_BIN}" ${CLANG_OCL_FLAGS} ${INC_FLAGS} ${comp_inc}
              ${per_file_flags} -c "${src}" -o "${obj}"
      RESULT_VARIABLE rc
      OUTPUT_VARIABLE log
      ERROR_VARIABLE  log)
    if(NOT rc EQUAL 0)
      message("--- ${stem} compile output ---\n${log}\n--- end ---")
      message(FATAL_ERROR "compile failed for ${src} (${rc})")
    endif()
    list(APPEND bc_outputs "${obj}")
  endforeach()

  # Response file because the per-file argv can exceed OS limits.
  set(resp "${BLD_DIR}/${name}.rsp")
  string(REPLACE ";" "\n" resp_content "${bc_outputs}")
  file(WRITE "${resp}" "${resp_content}\n")

  set(link0_bc "${BLD_DIR}/${name}.link0.bc")
  message(STATUS "llvm-link -> ${name}.link0.bc")
  execute_process(
    COMMAND "${LLVM_LINK_BIN}" "@${resp}" -o "${link0_bc}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE log
    ERROR_VARIABLE  log)
  if(NOT rc EQUAL 0)
    message("--- llvm-link output ---\n${log}\n--- end ---")
    message(FATAL_ERROR "llvm-link failed (${rc})")
  endif()

  # opt's internalize takes exact names only, so materialise the prefix-matching
  # symbols into a file for it.
  execute_process(
    COMMAND "${LLVM_NM_BIN}" --defined-only --format=just-symbols "${link0_bc}"
    OUTPUT_VARIABLE nm_out
    RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "llvm-nm on ${link0_bc} failed (${rc})")
  endif()
  string(REPLACE "\n" ";" nm_lines "${nm_out}")
  list(FILTER nm_lines INCLUDE REGEX "^${public_prefix}")
  list(LENGTH nm_lines n_public)
  set(api_file "${BLD_DIR}/${name}.public-api")
  string(REPLACE ";" "\n" api_content "${nm_lines}")
  file(WRITE "${api_file}" "${api_content}\n")

  set(out_bc "${BLD_DIR}/${name}.bc")
  message(STATUS "opt internalize+globaldce (${n_public} public) -> ${name}.bc")
  execute_process(
    COMMAND "${OPT_BIN}"
      -passes=internalize,globaldce
      "--internalize-public-api-file=${api_file}"
      "${link0_bc}" -o "${out_bc}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE log
    ERROR_VARIABLE  log)
  if(NOT rc EQUAL 0)
    message("--- opt output ---\n${log}\n--- end ---")
    message(FATAL_ERROR "opt failed (${rc})")
  endif()
  set(${name}_OUT "${out_bc}" PARENT_SCOPE)
endfunction()

build_bc_lib(ocml "${DEVLIBS_ROOT}/ocml/src" "__ocml_")
build_bc_lib(ockl "${DEVLIBS_ROOT}/ockl/src" "__ockl_")

file(COPY_FILE "${ocml_OUT}" "${OUT_DIR}/ocml.bc")
file(COPY_FILE "${ockl_OUT}" "${OUT_DIR}/ockl.bc")

file(GLOB oclc_srcs "${DEVLIBS_ROOT}/oclc/src/*.cl")
list(LENGTH oclc_srcs n_oclc)
message(STATUS "compiling oclc: ${n_oclc} .cl files")
set(oclc_inc "-I${DEVLIBS_ROOT}/oclc/inc")
foreach(src IN LISTS oclc_srcs)
  get_filename_component(stem "${src}" NAME_WE)
  set(out_bc "${OUT_DIR}/oclc_${stem}.bc")
  execute_process(
    COMMAND "${CLANG_BIN}" ${CLANG_OCL_FLAGS} ${oclc_inc} -c "${src}" -o "${out_bc}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE log
    ERROR_VARIABLE  log)
  if(NOT rc EQUAL 0)
    message("--- oclc ${stem} compile output ---\n${log}\n--- end ---")
    message(FATAL_ERROR "compile failed for ${src} (${rc})")
  endif()
endforeach()

set(lic_src "")
foreach(l LICENSE.TXT LICENSE COPYING)
  if(EXISTS "${DEVLIBS_ROOT}/${l}")
    set(lic_src "${DEVLIBS_ROOT}/${l}")
    break()
  endif()
endforeach()
if(lic_src)
  file(COPY_FILE "${lic_src}" "${OUT_DIR}/ocml.bc.LICENSE")
  file(COPY_FILE "${lic_src}" "${OUT_DIR}/ockl.bc.LICENSE")
endif()

file(SIZE "${OUT_DIR}/ocml.bc" ocml_size)
file(SIZE "${OUT_DIR}/ockl.bc" ockl_size)
message(STATUS "AMD device libs ${ROCM_TAG} staged at: ${OUT_DIR}")
message(STATUS "  ocml.bc: ${ocml_size} bytes")
message(STATUS "  ockl.bc: ${ockl_size} bytes")
