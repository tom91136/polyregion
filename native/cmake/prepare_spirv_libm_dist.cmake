cmake_minimum_required(VERSION 3.21)

include("${CMAKE_CURRENT_LIST_DIR}/../ProjectConfig.cmake")

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
set(LICENSE_SRC "${LLVM_SRC_DIR}/LICENSE.TXT")
if (NOT EXISTS "${LIBC_ROOT}/src/math/generic/exp.cpp")
    set(STAGE_DIR "${OUT_DIR}/.staging/spirv_libm")
    file(MAKE_DIRECTORY "${STAGE_DIR}")
    set(LLVM_TARBALL_NAME llvm-project-${LLVM_SRC_VERSION}.src.tar.xz)
    get_filename_component(LLVM_BUILD_DIR "${LLVM_SRC_DIR}" DIRECTORY)

    set(LLVM_TARBALL "${LLVM_BUILD_DIR}/${LLVM_TARBALL_NAME}")
    set(DOWNLOAD_LLVM FALSE)
    if (EXISTS "${LLVM_TARBALL}")
        file(SHA256 "${LLVM_TARBALL}" _tarball_hash)
        if (NOT "${_tarball_hash}" STREQUAL "${LLVM_SOURCE_SHA256}")
            set(LLVM_TARBALL "${STAGE_DIR}/${LLVM_TARBALL_NAME}")
            set(DOWNLOAD_LLVM TRUE)
        endif ()
    else ()
        set(LLVM_TARBALL "${STAGE_DIR}/${LLVM_TARBALL_NAME}")
        set(DOWNLOAD_LLVM TRUE)
    endif ()
    if (DOWNLOAD_LLVM)
        message(STATUS "Downloading LLVM source for libc maths from ${LLVM_SOURCE_URL}")
        file(DOWNLOAD "${LLVM_SOURCE_URL}" "${LLVM_TARBALL}"
                EXPECTED_HASH SHA256=${LLVM_SOURCE_SHA256}
                STATUS DOWNLOAD_STATUS)
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_CODE)
        if (NOT DOWNLOAD_CODE EQUAL 0)
            file(REMOVE "${LLVM_TARBALL}")
            message(FATAL_ERROR "LLVM source download failed: ${DOWNLOAD_STATUS}")
        endif ()
    endif ()

    set(EXTRACT_DIR "${STAGE_DIR}/src")
    set(EXTRACT_MARKER "${EXTRACT_DIR}/.extracted")
    if (NOT EXISTS "${EXTRACT_MARKER}")
        message(STATUS "Extracting LLVM libc sources from ${LLVM_TARBALL}")
        file(REMOVE_RECURSE "${EXTRACT_DIR}")
        file(MAKE_DIRECTORY "${EXTRACT_DIR}")
        file(ARCHIVE_EXTRACT INPUT "${LLVM_TARBALL}" DESTINATION "${EXTRACT_DIR}"
                PATTERNS "*/libc/*" "*/LICENSE.TXT")
        set(EXTRACTED_ROOT "${EXTRACT_DIR}/llvm-project-${LLVM_SRC_VERSION}.src")
        if (NOT EXISTS "${EXTRACTED_ROOT}/libc/src/math/generic/exp.cpp")
            message(FATAL_ERROR "LLVM libc exp.cpp missing from ${LLVM_TARBALL}")
        endif ()
        file(TOUCH "${EXTRACT_MARKER}")
    endif ()
    set(LIBC_ROOT "${EXTRACT_DIR}/llvm-project-${LLVM_SRC_VERSION}.src/libc")
    set(LICENSE_SRC "${EXTRACT_DIR}/llvm-project-${LLVM_SRC_VERSION}.src/LICENSE.TXT")
endif ()

set(FUNC_DEF "${CMAKE_CURRENT_LIST_DIR}/../polyc/backend/spirv_libm.def")
if (NOT EXISTS "${FUNC_DEF}")
  message(FATAL_ERROR "function list missing at ${FUNC_DEF}; f64 maths cannot be staged")
endif ()
file(STRINGS "${FUNC_DEF}" _def_lines REGEX "^POLYREGION_LIBM_FN\\(")
set(FUNCS)
set(ARITIES)
set(KINDS)
set(FLAGS)
foreach(_line IN LISTS _def_lines)
  if(NOT _line MATCHES "POLYREGION_LIBM_FN\\(([a-z0-9]+), ([12]), ([a-z]+), (skip|accurate)\\)")
    message(FATAL_ERROR "malformed function list entry: ${_line}")
  endif()
  list(APPEND FUNCS "${CMAKE_MATCH_1}")
  list(APPEND ARITIES "${CMAKE_MATCH_2}")
  list(APPEND KINDS "${CMAKE_MATCH_3}")
  list(APPEND FLAGS "${CMAKE_MATCH_4}")
endforeach()
list(LENGTH FUNCS _count)
if(_count EQUAL 0)
  message(FATAL_ERROR "empty function list")
endif()

file(MAKE_DIRECTORY "${OUT_DIR}/.staging")

set(COMPOSED_SINH [=[
#include "src/math/exp.h"
#include "src/math/expm1.h"
namespace LIBC_NAMESPACE_DECL {
double sinh(double x) {
  const double ax = __builtin_fabs(x);
  if (ax > 710.4758600739439) {
    const unsigned long long ux = __builtin_bit_cast(unsigned long long, x);
    return __builtin_bit_cast(double, 0x7ff0000000000000ULL | (ux & 0x8000000000000000ULL));
  }
  double r;
  if (ax > 709.78) {
    const double h = exp(0.5 * ax);
    const unsigned long long uh = __builtin_bit_cast(unsigned long long, h);
    r = __builtin_bit_cast(double, uh - 0x0010000000000000ULL) * h;
  } else {
    r = (expm1(x) - expm1(-x)) * 0.5;
  }
  const unsigned long long ux = __builtin_bit_cast(unsigned long long, x);
  const unsigned long long ur = __builtin_bit_cast(unsigned long long, r);
  return __builtin_bit_cast(double, (ur & 0x7fffffffffffffffULL) | (ux & 0x8000000000000000ULL));
}
}
]=])
set(COMPOSED_COSH [=[
#include "src/math/exp.h"
#include "src/math/expm1.h"
namespace LIBC_NAMESPACE_DECL {
double cosh(double x) {
  const double ax = __builtin_fabs(x);
  if (ax > 710.4758600739439) return __builtin_bit_cast(double, 0x7ff0000000000000ULL);
  if (ax > 709.78) {
    const double h = exp(0.5 * ax);
    const unsigned long long uh = __builtin_bit_cast(unsigned long long, h);
    return __builtin_bit_cast(double, uh - 0x0010000000000000ULL) * h;
  }
  return 1.0 + (expm1(x) + expm1(-x)) * 0.5;
}
}
]=])
set(COMPOSED_TANH [=[
#include "src/math/expm1.h"
namespace LIBC_NAMESPACE_DECL {
double tanh(double x) {
  if (__builtin_fabs(x) > 20.0) {
    const unsigned long long ux = __builtin_bit_cast(unsigned long long, x);
    return __builtin_bit_cast(double, 0x3ff0000000000000ULL | (ux & 0x8000000000000000ULL));
  }
  const double t = expm1(2.0 * x);
  return t / (t + 2.0);
}
}
]=])

set(COMMON_FLAGS
  --target=spirv64-unknown-unknown -nostdlibinc -std=c++17 -O2 -emit-llvm -c
  -I "${LIBC_ROOT}" -DLIBC_NAMESPACE=__llvm_libc -DLIBC_FULL_BUILD
  "-DLIBC_MATH=(LIBC_MATH_NO_ERRNO | LIBC_MATH_NO_EXCEPT)"
  -DLIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY)

set(OBJECTS)
math(EXPR _last "${_count} - 1")
foreach(_i RANGE ${_last})
  list(GET FUNCS ${_i} _fn)
  list(GET ARITIES ${_i} _arity)
  list(GET KINDS ${_i} _kind)
  list(GET FLAGS ${_i} _flags)
  if(_kind STREQUAL "libc")
    set(_src_file "${LIBC_ROOT}/src/math/generic/${_fn}.cpp")
  else()
    set(_src_file "${OUT_DIR}/.staging/${_fn}_composed.cpp")
    if(_kind STREQUAL "sinh")
      file(WRITE "${_src_file}" "${COMPOSED_SINH}")
    elseif(_kind STREQUAL "cosh")
      file(WRITE "${_src_file}" "${COMPOSED_COSH}")
    elseif(_kind STREQUAL "tanh")
      file(WRITE "${_src_file}" "${COMPOSED_TANH}")
    else()
      message(FATAL_ERROR "no composed source for kind ${_kind}")
    endif()
  endif()
  set(_flags_now ${COMMON_FLAGS})
  if(_flags STREQUAL "skip")
    list(APPEND _flags_now -DLIBC_MATH_HAS_SKIP_ACCURATE_PASS)
  endif()
  set(_entry "${OUT_DIR}/.staging/${_fn}_entry.cpp")
  file(WRITE "${_entry}"
    "#include \"src/math/${_fn}.h\"\n"
    "extern \"C\" double __polyregion_${_fn}_f64(double x")
  if(_arity EQUAL 2)
    file(APPEND "${_entry}" ", double y")
  endif()
  file(APPEND "${_entry}" ") { return LIBC_NAMESPACE::${_fn}(x")
  if(_arity EQUAL 2)
    file(APPEND "${_entry}" ", y")
  endif()
  file(APPEND "${_entry}" "); }\n")
  execute_process(
    COMMAND "${CLANGXX_BIN}" ${_flags_now} "${_src_file}" -o "${OUT_DIR}/.staging/${_fn}.bc"
    COMMAND_ECHO STDERR RESULT_VARIABLE SUCCESS)
  if(NOT SUCCESS EQUAL 0)
    message(FATAL_ERROR "failed to compile LLVM libc ${_fn} source for SPIR-V")
  endif()
  execute_process(
    COMMAND "${CLANGXX_BIN}" ${_flags_now} "${_entry}" -o "${OUT_DIR}/.staging/${_fn}_entry.bc"
    COMMAND_ECHO STDERR RESULT_VARIABLE SUCCESS)
  if(NOT SUCCESS EQUAL 0)
    message(FATAL_ERROR "failed to compile the LLVM libc ${_fn} entry adapter")
  endif()
  list(APPEND OBJECTS "${OUT_DIR}/.staging/${_fn}.bc" "${OUT_DIR}/.staging/${_fn}_entry.bc")
endforeach()

execute_process(
  COMMAND "${LLVM_LINK_BIN}" ${OBJECTS} -o "${OUT_DIR}/spirv-libm.bc"
  COMMAND_ECHO STDERR RESULT_VARIABLE SUCCESS)
if(NOT SUCCESS EQUAL 0)
  message(FATAL_ERROR "failed to link the LLVM libc SPIR-V bitcode")
endif()
configure_file("${LICENSE_SRC}" "${OUT_DIR}/spirv-libm.bc.LICENSE" COPYONLY)
file(SIZE "${OUT_DIR}/spirv-libm.bc" bc_size)
message(STATUS "spirv-libm.bc (LLVM libc f64 maths, ${_count} functions): ${bc_size} bytes")
