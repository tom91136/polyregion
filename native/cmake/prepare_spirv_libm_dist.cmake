cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED OUT_DIR)
  set(OUT_DIR "${CMAKE_CURRENT_LIST_DIR}/dist")
endif()
if(NOT DEFINED LLVM_SRC_DIR)
  message(FATAL_ERROR "LLVM_SRC_DIR must point at the llvm-project source tree")
endif()
if(NOT DEFINED POLYREGION_LLVM_BIN_HINT)
  set(POLYREGION_LLVM_BIN_HINT "")
endif()

function(_find_tool var)
  if(DEFINED ${var} AND ${var})
    return()
  endif()
  find_program(${var} NAMES ${ARGN} HINTS "${POLYREGION_LLVM_BIN_HINT}" NO_DEFAULT_PATH)
  if(NOT ${var})
    find_program(${var} NAMES ${ARGN})
  endif()
  if(NOT ${var})
    list(GET ARGN 0 _primary)
    message(FATAL_ERROR "${_primary} not found; set -D ${var}=...")
  endif()
endfunction()

_find_tool(CLANGXX_BIN clang++ clang++-23 clang++-22)
_find_tool(LLVM_LINK_BIN llvm-link)

set(LIBC_ROOT "${LLVM_SRC_DIR}/libc")
if (NOT EXISTS "${LIBC_ROOT}/src/math/generic/exp.cpp")
    message(FATAL_ERROR "LLVM libc source tree missing at ${LIBC_ROOT}; f64 exp cannot be staged")
endif ()
set(ENTRY "${OUT_DIR}/.staging/spirv_libm_entry.cpp")
file(MAKE_DIRECTORY "${OUT_DIR}/.staging")
file(WRITE "${ENTRY}"
"#include \"src/math/exp.h\"
extern \"C\" double __polyregion_exp_f64(double x) { return LIBC_NAMESPACE::exp(x); }
")

set(COMMON_FLAGS
  --target=spirv64-unknown-unknown -std=c++17 -O2 -emit-llvm -c
  -I "${LIBC_ROOT}" -DLIBC_NAMESPACE=__llvm_libc -DLIBC_FULL_BUILD
  "-DLIBC_MATH=(LIBC_MATH_NO_ERRNO | LIBC_MATH_NO_EXCEPT)"
  -DLIBC_MATH_HAS_SKIP_ACCURATE_PASS -DLIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY)

execute_process(
  COMMAND "${CLANGXX_BIN}" ${COMMON_FLAGS} "${LIBC_ROOT}/src/math/generic/exp.cpp"
          -o "${OUT_DIR}/.staging/exp.bc"
  COMMAND_ECHO STDERR RESULT_VARIABLE SUCCESS)
if(NOT SUCCESS EQUAL 0)
  message(FATAL_ERROR "failed to compile LLVM libc exp.cpp for SPIR-V")
endif()
execute_process(
  COMMAND "${CLANGXX_BIN}" ${COMMON_FLAGS} "${ENTRY}" -o "${OUT_DIR}/.staging/entry.bc"
  COMMAND_ECHO STDERR RESULT_VARIABLE SUCCESS)
if(NOT SUCCESS EQUAL 0)
  message(FATAL_ERROR "failed to compile the LLVM libc exp entry adapter")
endif()
execute_process(
  COMMAND "${LLVM_LINK_BIN}" "${OUT_DIR}/.staging/exp.bc" "${OUT_DIR}/.staging/entry.bc"
          -o "${OUT_DIR}/spirv-libm.bc"
  COMMAND_ECHO STDERR RESULT_VARIABLE SUCCESS)
if(NOT SUCCESS EQUAL 0)
  message(FATAL_ERROR "failed to link the LLVM libc SPIR-V bitcode")
endif()
configure_file("${LLVM_SRC_DIR}/LICENSE.TXT" "${OUT_DIR}/spirv-libm.bc.LICENSE" COPYONLY)
file(SIZE "${OUT_DIR}/spirv-libm.bc" bc_size)
message(STATUS "spirv-libm.bc (LLVM libc exp): ${bc_size} bytes")
