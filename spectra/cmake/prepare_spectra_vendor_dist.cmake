cmake_minimum_required(VERSION 3.23)

foreach(required OUT_DIR ONEDPL_COMMIT ONEDPL_SHA256 ROCTHRUST_COMMIT ROCTHRUST_SHA256
                 ROCPRIM_COMMIT ROCPRIM_SHA256 CUDA_CCCL_VERSION CUDA_CCCL_PLATFORM
                 CUDA_CCCL_SHA256 CUDA_CUDART_VERSION CUDA_CUDART_SHA256 CUDA_NVCC_VERSION CUDA_NVCC_SHA256)
  if(NOT DEFINED ${required} OR NOT ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

if(NOT DEFINED ONEDPL_URL)
  set(ONEDPL_URL "https://codeload.github.com/uxlfoundation/oneDPL/tar.gz/${ONEDPL_COMMIT}")
endif()
if(NOT DEFINED ROCTHRUST_URL)
  set(ROCTHRUST_URL "https://codeload.github.com/ROCm/rocThrust/tar.gz/${ROCTHRUST_COMMIT}")
endif()
if(NOT DEFINED ROCPRIM_URL)
  set(ROCPRIM_URL "https://codeload.github.com/ROCm/rocPRIM/tar.gz/${ROCPRIM_COMMIT}")
endif()
if(NOT DEFINED CUDA_CCCL_URL)
  set(CUDA_CCCL_URL
      "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/${CUDA_CCCL_PLATFORM}/cuda_cccl-${CUDA_CCCL_PLATFORM}-${CUDA_CCCL_VERSION}-archive.tar.xz")
endif()
if(NOT DEFINED CUDA_CUDART_URL)
  set(CUDA_CUDART_URL
      "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/${CUDA_CCCL_PLATFORM}/cuda_cudart-${CUDA_CCCL_PLATFORM}-${CUDA_CUDART_VERSION}-archive.tar.xz")
endif()
if(NOT DEFINED CUDA_NVCC_URL)
  set(CUDA_NVCC_URL
      "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/${CUDA_CCCL_PLATFORM}/cuda_nvcc-${CUDA_CCCL_PLATFORM}-${CUDA_NVCC_VERSION}-archive.tar.xz")
endif()

string(CONCAT SPECTRA_VENDOR_TAG
       "format=3\n"
       "onedpl=${ONEDPL_COMMIT}:${ONEDPL_SHA256}\n"
       "rocthrust=${ROCTHRUST_COMMIT}:${ROCTHRUST_SHA256}\n"
       "rocprim=${ROCPRIM_COMMIT}:${ROCPRIM_SHA256}\n"
       "cuda-cccl=${CUDA_CCCL_PLATFORM}:${CUDA_CCCL_VERSION}:${CUDA_CCCL_SHA256}\n"
       "cuda-cudart=${CUDA_CUDART_VERSION}:${CUDA_CUDART_SHA256}\n"
       "cuda-nvcc=${CUDA_NVCC_VERSION}:${CUDA_NVCC_SHA256}\n")

function(vendor_manifest root out_manifest)
  file(GLOB_RECURSE entries LIST_DIRECTORIES true RELATIVE "${root}" "${root}/*")
  list(SORT entries)
  set(manifest "")
  foreach(entry IN LISTS entries)
    if(entry STREQUAL ".manifest" OR entry STREQUAL ".staged" OR IS_DIRECTORY "${root}/${entry}")
      continue()
    endif()
    if(IS_SYMLINK "${root}/${entry}")
      file(READ_SYMLINK "${root}/${entry}" target)
      string(SHA256 digest "symlink:${target}")
    else()
      file(SHA256 "${root}/${entry}" digest)
    endif()
    string(APPEND manifest "${digest}  ${entry}\n")
  endforeach()
  set(${out_manifest} "${manifest}" PARENT_SCOPE)
endfunction()

function(validate_vendor_tree root require_marker out_valid out_error)
  set(error "")
  if(NOT IS_DIRECTORY "${root}")
    set(error "vendor root is absent: ${root}")
  elseif(require_marker)
    if(NOT EXISTS "${root}/.staged")
      set(error "vendor marker is absent")
    else()
      file(READ "${root}/.staged" marker)
      if(NOT marker STREQUAL SPECTRA_VENDOR_TAG)
        set(error "vendor marker differs")
      endif()
    endif()
  endif()

  if(NOT error)
    foreach(required_file
            onedpl/include/oneapi/dpl/execution
            rocm/rocThrust/thrust/version.h
            rocm/rocThrust/thrust/rocthrust_version.hpp
            rocm/rocPRIM/rocprim/include/rocprim/rocprim.hpp
            rocm/rocPRIM/rocprim/include/rocprim/rocprim_version.hpp
            cuda_cccl/include/thrust/execution_policy.h
            cuda/include/cuda_runtime.h
            cuda/nvvm/libdevice/libdevice.10.bc
            cuda/version.txt)
      if(NOT EXISTS "${root}/${required_file}")
        set(error "staged vendor tree is incomplete: ${required_file}")
        break()
      endif()
    endforeach()
  endif()

  if(NOT error AND require_marker)
    if(NOT EXISTS "${root}/.manifest")
      set(error "vendor content manifest is absent")
    else()
      file(READ "${root}/.manifest" expected_manifest)
      vendor_manifest("${root}" actual_manifest)
      if(NOT actual_manifest STREQUAL expected_manifest)
        set(error "staged vendor content differs from its pinned manifest")
      endif()
    endif()
  endif()

  if(NOT error)
    file(GLOB_RECURSE entries LIST_DIRECTORIES true RELATIVE "${root}" "${root}/*")
    foreach(entry IN LISTS entries)
      # Pinned upstream archives may legitimately contain .patch/.diff test
      # fixtures (oneDPL does). Archive hashes plus the complete content
      # manifest prove those files are upstream and detect any local addition.
      if(entry MATCHES "(^|/)(\\.git|\\.svn)(/|$)"
         OR entry MATCHES "(\\.rej|\\.orig|\\.bak|~)$")
        set(error "staged vendor tree contains a forbidden artefact: ${entry}")
        break()
      endif()
    endforeach()
  endif()

  if(error)
    set(${out_valid} FALSE PARENT_SCOPE)
    set(${out_error} "${error}" PARENT_SCOPE)
  else()
    set(${out_valid} TRUE PARENT_SCOPE)
    set(${out_error} "" PARENT_SCOPE)
  endif()
endfunction()

function(generate_rocm_version_header tree prefix input)
  file(STRINGS "${tree}/CMakeLists.txt" version_line REGEX "^set\\(VERSION_STRING")
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" version "${version_line}")
  if(NOT version)
    message(FATAL_ERROR "cannot read VERSION_STRING from ${tree}/CMakeLists.txt")
  endif()
  string(REPLACE "." ";" parts "${version}")
  list(GET parts 0 major)
  list(GET parts 1 minor)
  list(GET parts 2 patch)
  math(EXPR number "${major} * 100000 + ${minor} * 100 + ${patch}")
  file(READ "${tree}/${input}.in" contents)
  string(REPLACE "@${prefix}_VERSION_NUMBER@" "${number}" contents "${contents}")
  string(REPLACE "@${prefix}_VERSION_MAJOR@" "${major}" contents "${contents}")
  string(REPLACE "@${prefix}_VERSION_MINOR@" "${minor}" contents "${contents}")
  string(REPLACE "@${prefix}_VERSION_PATCH@" "${patch}" contents "${contents}")
  file(WRITE "${tree}/${input}" "${contents}")
endfunction()

if(SPECTRA_VENDOR_WORKER)
  if(EXISTS "${OUT_DIR}")
    file(REMOVE_RECURSE "${OUT_DIR}")
  endif()
  file(MAKE_DIRECTORY "${OUT_DIR}")
  if(NOT DEFINED DOWNLOAD_CACHE)
    set(DOWNLOAD_CACHE "${OUT_DIR}/.downloads")
  endif()
  file(MAKE_DIRECTORY "${DOWNLOAD_CACHE}")

  function(download_and_extract name url sha256 destination)
    set(archive "${DOWNLOAD_CACHE}/${name}-${sha256}.archive")
    if(EXISTS "${archive}")
      file(SHA256 "${archive}" existing_sha256)
      if(NOT existing_sha256 STREQUAL sha256)
        file(REMOVE "${archive}")
      endif()
    endif()
    message(STATUS "fetching ${name}: ${url}")
    if(EXISTS "${url}")
      file(COPY_FILE "${url}" "${archive}" ONLY_IF_DIFFERENT)
      file(SHA256 "${archive}" downloaded_sha256)
      if(NOT downloaded_sha256 STREQUAL sha256)
        file(REMOVE "${archive}")
        message(FATAL_ERROR "${name} local archive hash differs: expected ${sha256}, got ${downloaded_sha256}")
      endif()
    else()
      file(DOWNLOAD "${url}" "${archive}"
           EXPECTED_HASH "SHA256=${sha256}"
           STATUS status TLS_VERIFY ON)
      list(GET status 0 code)
      list(GET status 1 detail)
      if(NOT code EQUAL 0)
        file(REMOVE "${archive}")
        message(FATAL_ERROR "${name} fetch failed (${code}): ${detail}")
      endif()
    endif()

    set(extracted "${OUT_DIR}/.extract-${name}")
    file(REMOVE_RECURSE "${extracted}" "${destination}")
    file(MAKE_DIRECTORY "${extracted}" "${destination}")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${extracted}")
    file(GLOB roots LIST_DIRECTORIES true "${extracted}/*")
    list(LENGTH roots root_count)
    if(NOT root_count EQUAL 1)
      message(FATAL_ERROR "${name} archive must contain one root directory, found ${root_count}")
    endif()
    list(GET roots 0 root)
    if(NOT IS_DIRECTORY "${root}")
      message(FATAL_ERROR "${name} archive root is not a directory: ${root}")
    endif()
    file(COPY "${root}/" DESTINATION "${destination}")
    file(REMOVE_RECURSE "${extracted}")
  endfunction()

  download_and_extract(onedpl "${ONEDPL_URL}" "${ONEDPL_SHA256}" "${OUT_DIR}/onedpl")
  download_and_extract(rocthrust "${ROCTHRUST_URL}" "${ROCTHRUST_SHA256}" "${OUT_DIR}/rocm/rocThrust")
  download_and_extract(rocprim "${ROCPRIM_URL}" "${ROCPRIM_SHA256}" "${OUT_DIR}/rocm/rocPRIM")
  download_and_extract(cuda_cccl "${CUDA_CCCL_URL}" "${CUDA_CCCL_SHA256}" "${OUT_DIR}/cuda_cccl")
  download_and_extract(cuda_cudart "${CUDA_CUDART_URL}" "${CUDA_CUDART_SHA256}" "${OUT_DIR}/.cuda-cudart")
  download_and_extract(cuda_nvcc "${CUDA_NVCC_URL}" "${CUDA_NVCC_SHA256}" "${OUT_DIR}/.cuda-nvcc")
  # clang needs include/, nvvm/libdevice, a version file and the ptxas/fatbinary it runs for device objects
  file(COPY "${OUT_DIR}/.cuda-cudart/include/" "${OUT_DIR}/.cuda-nvcc/include/" DESTINATION "${OUT_DIR}/cuda/include")
  file(COPY "${OUT_DIR}/.cuda-nvcc/nvvm/libdevice/" DESTINATION "${OUT_DIR}/cuda/nvvm/libdevice")
  file(COPY "${OUT_DIR}/.cuda-nvcc/bin/ptxas" "${OUT_DIR}/.cuda-nvcc/bin/fatbinary" DESTINATION "${OUT_DIR}/cuda/bin")
  # clang's CUDA wrapper force-includes this cuRAND header; Thrust never uses cuRAND, so an empty one avoids the redist
  file(WRITE "${OUT_DIR}/cuda/include/curand_mtgp32_kernel.h" "#pragma once\n")
  file(WRITE "${OUT_DIR}/cuda/version.txt" "CUDA Version ${CUDA_CUDART_VERSION}\n")
  file(REMOVE_RECURSE "${OUT_DIR}/.cuda-cudart" "${OUT_DIR}/.cuda-nvcc")

  generate_rocm_version_header("${OUT_DIR}/rocm/rocThrust" rocthrust thrust/rocthrust_version.hpp)
  generate_rocm_version_header("${OUT_DIR}/rocm/rocPRIM" rocprim rocprim/include/rocprim/rocprim_version.hpp)
  validate_vendor_tree("${OUT_DIR}" FALSE valid error)
  if(NOT valid)
    message(FATAL_ERROR "${error}")
  endif()
  return()
endif()

validate_vendor_tree("${OUT_DIR}" TRUE current_valid current_error)
if(SPECTRA_VENDOR_VERIFY_ONLY)
  if(NOT current_valid)
    message(FATAL_ERROR "Spectra vendor verification failed: ${current_error}")
  endif()
  message(STATUS "Spectra vendor inputs verified at ${OUT_DIR}")
  return()
endif()
if(current_valid)
  message(STATUS "Spectra vendor inputs are up to date at ${OUT_DIR}")
  return()
endif()

get_filename_component(out_parent "${OUT_DIR}" DIRECTORY)
get_filename_component(out_name "${OUT_DIR}" NAME)
file(MAKE_DIRECTORY "${out_parent}")
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef nonce)
set(staging "${out_parent}/.${out_name}.staging-${nonce}")
set(backup "${out_parent}/.${out_name}.previous-${nonce}")
set(download_cache "${out_parent}/.${out_name}.downloads")

set(worker_args
    -DSPECTRA_VENDOR_WORKER=ON
    -DOUT_DIR=${staging}
    -DDOWNLOAD_CACHE=${download_cache}
    -DONEDPL_COMMIT=${ONEDPL_COMMIT}
    -DONEDPL_SHA256=${ONEDPL_SHA256}
    -DROCTHRUST_COMMIT=${ROCTHRUST_COMMIT}
    -DROCTHRUST_SHA256=${ROCTHRUST_SHA256}
    -DROCPRIM_COMMIT=${ROCPRIM_COMMIT}
    -DROCPRIM_SHA256=${ROCPRIM_SHA256}
    -DCUDA_CCCL_VERSION=${CUDA_CCCL_VERSION}
    -DCUDA_CCCL_PLATFORM=${CUDA_CCCL_PLATFORM}
    -DCUDA_CCCL_SHA256=${CUDA_CCCL_SHA256}
    -DCUDA_CUDART_VERSION=${CUDA_CUDART_VERSION}
    -DCUDA_CUDART_SHA256=${CUDA_CUDART_SHA256}
    -DCUDA_NVCC_VERSION=${CUDA_NVCC_VERSION}
    -DCUDA_NVCC_SHA256=${CUDA_NVCC_SHA256}
    -DONEDPL_URL=${ONEDPL_URL}
    -DROCTHRUST_URL=${ROCTHRUST_URL}
    -DROCPRIM_URL=${ROCPRIM_URL}
    -DCUDA_CCCL_URL=${CUDA_CCCL_URL}
    -DCUDA_CUDART_URL=${CUDA_CUDART_URL}
    -DCUDA_NVCC_URL=${CUDA_NVCC_URL})
execute_process(
  COMMAND "${CMAKE_COMMAND}" ${worker_args} -P "${CMAKE_CURRENT_LIST_FILE}"
  COMMAND_ECHO STDERR
  RESULT_VARIABLE worker_result)
if(NOT worker_result EQUAL 0)
  file(REMOVE_RECURSE "${staging}")
  message(FATAL_ERROR "Spectra vendor staging failed; existing destination was preserved")
endif()

vendor_manifest("${staging}" staged_manifest)
file(WRITE "${staging}/.manifest" "${staged_manifest}")
file(WRITE "${staging}/.staged" "${SPECTRA_VENDOR_TAG}")
validate_vendor_tree("${staging}" TRUE staged_valid staged_error)
if(NOT staged_valid)
  file(REMOVE_RECURSE "${staging}")
  message(FATAL_ERROR "${staged_error}")
endif()

if(EXISTS "${OUT_DIR}")
  file(RENAME "${OUT_DIR}" "${backup}" RESULT rename_error)
  if(rename_error)
    file(REMOVE_RECURSE "${staging}")
    message(FATAL_ERROR "cannot preserve existing vendor destination: ${rename_error}")
  endif()
endif()
file(RENAME "${staging}" "${OUT_DIR}" RESULT rename_error)
if(rename_error)
  if(EXISTS "${backup}")
    file(RENAME "${backup}" "${OUT_DIR}")
  endif()
  file(REMOVE_RECURSE "${staging}")
  message(FATAL_ERROR "cannot publish staged vendor destination: ${rename_error}")
endif()
file(REMOVE_RECURSE "${backup}")
message(STATUS "Spectra vendor inputs staged at ${OUT_DIR}")
