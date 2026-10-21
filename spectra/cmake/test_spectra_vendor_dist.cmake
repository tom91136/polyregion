cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED TEST_ROOT)
  set(TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/spectra-vendor-staging-test")
endif()
get_filename_component(script "${CMAKE_CURRENT_LIST_DIR}/prepare_spectra_vendor_dist.cmake" ABSOLUTE)
file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/archives")

function(make_archive kind revision complete forbidden out_archive out_sha256)
  set(source "${TEST_ROOT}/source-${kind}-${revision}")
  set(root "${source}/${kind}-${revision}")
  file(REMOVE_RECURSE "${source}")
  file(MAKE_DIRECTORY "${root}")
  if(kind STREQUAL "onedpl")
    if(complete)
      file(MAKE_DIRECTORY "${root}/include/oneapi/dpl")
      file(WRITE "${root}/include/oneapi/dpl/execution" "onedpl-${revision}\n")
    endif()
  elseif(kind STREQUAL "rocthrust")
    file(MAKE_DIRECTORY "${root}/thrust")
    file(WRITE "${root}/CMakeLists.txt" "set(VERSION_STRING 3.4.0)\n")
    file(WRITE "${root}/thrust/version.h" "#pragma once\n")
    file(WRITE "${root}/thrust/rocthrust_version.hpp.in"
         "#define ROCTHRUST_VERSION @rocthrust_VERSION_NUMBER@\n"
         "#define ROCTHRUST_VERSION_MAJOR @rocthrust_VERSION_MAJOR@\n"
         "#define ROCTHRUST_VERSION_MINOR @rocthrust_VERSION_MINOR@\n"
         "#define ROCTHRUST_VERSION_PATCH @rocthrust_VERSION_PATCH@\n")
  elseif(kind STREQUAL "rocprim")
    file(MAKE_DIRECTORY "${root}/rocprim/include/rocprim")
    file(WRITE "${root}/CMakeLists.txt" "set(VERSION_STRING 4.1.0)\n")
    file(WRITE "${root}/rocprim/include/rocprim/rocprim.hpp" "#pragma once\n")
    file(WRITE "${root}/rocprim/include/rocprim/rocprim_version.hpp.in"
         "#define ROCPRIM_VERSION @rocprim_VERSION_NUMBER@\n"
         "#define ROCPRIM_VERSION_MAJOR @rocprim_VERSION_MAJOR@\n"
         "#define ROCPRIM_VERSION_MINOR @rocprim_VERSION_MINOR@\n"
         "#define ROCPRIM_VERSION_PATCH @rocprim_VERSION_PATCH@\n")
  elseif(kind STREQUAL "cuda_cccl")
    file(MAKE_DIRECTORY "${root}/include/thrust")
    file(WRITE "${root}/include/thrust/execution_policy.h" "// cuda-cccl-${revision}\n")
  elseif(kind STREQUAL "cuda_cudart")
    file(MAKE_DIRECTORY "${root}/include")
    file(WRITE "${root}/include/cuda_runtime.h" "// cuda-cudart-${revision}\n")
  elseif(kind STREQUAL "cuda_nvcc")
    file(MAKE_DIRECTORY "${root}/include/crt" "${root}/nvvm/libdevice" "${root}/bin")
    file(WRITE "${root}/bin/ptxas" "ptxas-${revision}\n")
    file(WRITE "${root}/bin/fatbinary" "fatbinary-${revision}\n")
    file(WRITE "${root}/include/crt/host_config.h" "// cuda-nvcc-${revision}\n")
    file(WRITE "${root}/nvvm/libdevice/libdevice.10.bc" "libdevice-${revision}\n")
  else()
    message(FATAL_ERROR "unknown mock vendor: ${kind}")
  endif()
  if(forbidden)
    file(WRITE "${root}/local.rej" "not pristine\n")
  endif()

  set(archive "${TEST_ROOT}/archives/${kind}-${revision}.tar.gz")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar cf "${archive}" --format=gnutar -- "${kind}-${revision}"
    WORKING_DIRECTORY "${source}"
    RESULT_VARIABLE result
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "cannot create ${kind} mock archive: ${error}")
  endif()
  file(SHA256 "${archive}" sha256)
  set(${out_archive} "${archive}" PARENT_SCOPE)
  set(${out_sha256} "${sha256}" PARENT_SCOPE)
endfunction()

function(run_stage out_dir onedpl_revision onedpl_archive onedpl_sha256 verify_only expect_success)
  set(command
      "${CMAKE_COMMAND}"
      -DOUT_DIR=${out_dir}
      -DONEDPL_COMMIT=${onedpl_revision}
      -DONEDPL_SHA256=${onedpl_sha256}
      -DROCTHRUST_COMMIT=rocthrust-v1
      -DROCTHRUST_SHA256=${rocthrust_sha256}
      -DROCPRIM_COMMIT=rocprim-v1
      -DROCPRIM_SHA256=${rocprim_sha256}
      -DCUDA_CCCL_VERSION=cccl-v1
      -DCUDA_CCCL_PLATFORM=linux-test
      -DCUDA_CCCL_SHA256=${cuda_cccl_sha256}
      -DCUDA_CUDART_VERSION=cudart-v1
      -DCUDA_CUDART_SHA256=${cuda_cudart_sha256}
      -DCUDA_NVCC_VERSION=nvcc-v1
      -DCUDA_NVCC_SHA256=${cuda_nvcc_sha256}
      -DONEDPL_URL=${onedpl_archive}
      -DROCTHRUST_URL=${rocthrust_archive}
      -DROCPRIM_URL=${rocprim_archive}
      -DCUDA_CCCL_URL=${cuda_cccl_archive}
      -DCUDA_CUDART_URL=${cuda_cudart_archive}
      -DCUDA_NVCC_URL=${cuda_nvcc_archive})
  if(verify_only)
    list(APPEND command -DSPECTRA_VENDOR_VERIFY_ONLY=ON)
  endif()
  list(APPEND command -P "${script}")
  execute_process(
    COMMAND ${command}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(expect_success AND NOT result EQUAL 0)
    message(FATAL_ERROR "staging unexpectedly failed:\n${output}\n${error}")
  elseif(NOT expect_success AND result EQUAL 0)
    message(FATAL_ERROR "staging unexpectedly succeeded:\n${output}\n${error}")
  endif()
endfunction()

make_archive(onedpl v1 TRUE FALSE onedpl_v1_archive onedpl_v1_sha256)
make_archive(onedpl v2 TRUE FALSE onedpl_v2_archive onedpl_v2_sha256)
make_archive(onedpl incomplete FALSE FALSE onedpl_incomplete_archive onedpl_incomplete_sha256)
make_archive(onedpl contaminated TRUE TRUE onedpl_contaminated_archive onedpl_contaminated_sha256)
make_archive(rocthrust v1 TRUE FALSE rocthrust_archive rocthrust_sha256)
make_archive(rocprim v1 TRUE FALSE rocprim_archive rocprim_sha256)
make_archive(cuda_cccl v1 TRUE FALSE cuda_cccl_archive cuda_cccl_sha256)
make_archive(cuda_cudart v1 TRUE FALSE cuda_cudart_archive cuda_cudart_sha256)
make_archive(cuda_nvcc v1 TRUE FALSE cuda_nvcc_archive cuda_nvcc_sha256)

set(destination "${TEST_ROOT}/vendor")
run_stage("${destination}" v1 "${onedpl_v1_archive}" "${onedpl_v1_sha256}" FALSE TRUE)
foreach(required .staged .manifest onedpl/include/oneapi/dpl/execution
                 rocm/rocThrust/thrust/rocthrust_version.hpp
                 rocm/rocPRIM/rocprim/include/rocprim/rocprim_version.hpp
                 cuda_cccl/include/thrust/execution_policy.h
                 cuda/include/cuda_runtime.h cuda/include/crt/host_config.h
                 cuda/nvvm/libdevice/libdevice.10.bc cuda/version.txt cuda/bin/ptxas)
  if(NOT EXISTS "${destination}/${required}")
    message(FATAL_ERROR "successful stage omitted ${required}")
  endif()
endforeach()

# An identical invocation must take the marker/manifest fast path without
# replacing the destination.
file(TIMESTAMP "${destination}/.staged" marker_before "%s")
execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
run_stage("${destination}" v1 "${onedpl_v1_archive}" "${onedpl_v1_sha256}" FALSE TRUE)
file(TIMESTAMP "${destination}/.staged" marker_after "%s")
if(NOT marker_before STREQUAL marker_after)
  message(FATAL_ERROR "idempotent staging replaced an up-to-date destination")
endif()

# Verify-only mode detects edits, while normal staging repairs them from the
# hash-pinned archive.
file(APPEND "${destination}/onedpl/include/oneapi/dpl/execution" "modified\n")
run_stage("${destination}" v1 "${onedpl_v1_archive}" "${onedpl_v1_sha256}" TRUE FALSE)
run_stage("${destination}" v1 "${onedpl_v1_archive}" "${onedpl_v1_sha256}" FALSE TRUE)
file(READ "${destination}/onedpl/include/oneapi/dpl/execution" repaired)
if(NOT repaired STREQUAL "onedpl-v1\n")
  message(FATAL_ERROR "normal staging did not repair a modified vendor file")
endif()

# A changed vendor identity replaces the complete tree.
run_stage("${destination}" v2 "${onedpl_v2_archive}" "${onedpl_v2_sha256}" FALSE TRUE)
file(READ "${destination}/onedpl/include/oneapi/dpl/execution" replaced)
if(NOT replaced STREQUAL "onedpl-v2\n")
  message(FATAL_ERROR "changed vendor identity did not replace the staged tree")
endif()
file(READ "${destination}/.staged" marker_v2)

# Incomplete or contaminated new inputs fail without disturbing the last good
# destination and without leaving a publication staging directory behind.
run_stage("${destination}" incomplete "${onedpl_incomplete_archive}" "${onedpl_incomplete_sha256}" FALSE FALSE)
file(READ "${destination}/.staged" marker_after_failure)
if(NOT marker_after_failure STREQUAL marker_v2)
  message(FATAL_ERROR "failed staging changed the published marker")
endif()
run_stage("${destination}" contaminated "${onedpl_contaminated_archive}" "${onedpl_contaminated_sha256}" FALSE FALSE)
file(GLOB leftovers "${TEST_ROOT}/.vendor.staging-*" "${TEST_ROOT}/.vendor.previous-*")
if(leftovers)
  message(FATAL_ERROR "failed staging left transaction artefacts: ${leftovers}")
endif()

# An incomplete first stage must not publish a destination at all.
set(missing_destination "${TEST_ROOT}/missing-vendor")
run_stage("${missing_destination}" incomplete "${onedpl_incomplete_archive}" "${onedpl_incomplete_sha256}" FALSE FALSE)
if(EXISTS "${missing_destination}")
  message(FATAL_ERROR "failed initial staging published an incomplete destination")
endif()

message(STATUS "Spectra vendor staging tests passed")
