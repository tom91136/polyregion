cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CUDA_VERSION)
  message(FATAL_ERROR "CUDA_VERSION not set; pass -DCUDA_VERSION=<x.y.z>")
endif()
if(NOT DEFINED OUT_DIR)
  message(FATAL_ERROR "OUT_DIR not set; pass -DOUT_DIR=/path/to/staging")
endif()

# XXX libdevice.10.bc is target-independent, use linux-x86_66 for simplicity
set(CUDA_ARCH "linux-x86_64")

# CUDA 13 split libdevice out of cuda_nvcc into a new libnvvm redist component.
string(REGEX MATCH "^([0-9]+)" cuda_major "${CUDA_VERSION}")
if(cuda_major GREATER_EQUAL 13)
  set(LIBDEVICE_COMP "libnvvm")
else()
  set(LIBDEVICE_COMP "cuda_nvcc")
endif()

set(COMPONENTS "${LIBDEVICE_COMP}")

set(REDIST_BASE "https://developer.download.nvidia.com/compute/cuda/redist")
set(INDEX_URL "${REDIST_BASE}/redistrib_${CUDA_VERSION}.json")

set(STAGING "${OUT_DIR}/.staging/cuda/${CUDA_VERSION}")
file(MAKE_DIRECTORY "${STAGING}" "${OUT_DIR}")

set(INDEX_FILE "${STAGING}/redistrib.json")
if(NOT EXISTS "${INDEX_FILE}")
  message(STATUS "fetching index: ${INDEX_URL}")
  file(DOWNLOAD "${INDEX_URL}" "${INDEX_FILE}" STATUS dl_status TLS_VERIFY ON)
  list(GET dl_status 0 code)
  list(GET dl_status 1 msg)
  if(NOT code EQUAL 0)
    file(REMOVE "${INDEX_FILE}")
    message(FATAL_ERROR "index fetch failed (${code}): ${msg}")
  endif()
endif()
file(READ "${INDEX_FILE}" INDEX_JSON)

foreach(comp IN LISTS COMPONENTS)
  string(JSON has ERROR_VARIABLE jerr GET "${INDEX_JSON}" "${comp}")
  if(jerr)
    message(FATAL_ERROR "component '${comp}' not in index ${INDEX_URL}: ${jerr}")
  endif()
  string(JSON has_arch ERROR_VARIABLE jerr GET "${INDEX_JSON}" "${comp}" "${CUDA_ARCH}")
  if(jerr)
    message(FATAL_ERROR "component '${comp}' has no '${CUDA_ARCH}' build: ${jerr}")
  endif()

  string(JSON rel_path GET "${INDEX_JSON}" "${comp}" "${CUDA_ARCH}" "relative_path")
  string(JSON sha256   GET "${INDEX_JSON}" "${comp}" "${CUDA_ARCH}" "sha256")
  string(JSON comp_ver GET "${INDEX_JSON}" "${comp}" "version")

  get_filename_component(archive_name "${rel_path}" NAME)
  set(archive_path "${STAGING}/${archive_name}")
  set(url "${REDIST_BASE}/${rel_path}")

  # file(DOWNLOAD EXPECTED_HASH) skips the transfer when a hash-matching file
  # already exists, so no manual pre-check needed.
  message(STATUS "fetching ${comp} ${comp_ver}")
  file(DOWNLOAD "${url}" "${archive_path}"
       EXPECTED_HASH SHA256=${sha256}
       STATUS dl_status TLS_VERIFY ON)
  list(GET dl_status 0 code)
  list(GET dl_status 1 msg)
  if(NOT code EQUAL 0)
    file(REMOVE "${archive_path}")
    message(FATAL_ERROR "${comp} fetch failed (${code}): ${msg}")
  endif()

  set(extract_marker "${STAGING}/${comp}/.extracted")
  if(NOT EXISTS "${extract_marker}")
    file(REMOVE_RECURSE "${STAGING}/${comp}")
    file(MAKE_DIRECTORY "${STAGING}/${comp}")
    message(STATUS "extracting ${archive_name}")
    file(ARCHIVE_EXTRACT INPUT "${archive_path}" DESTINATION "${STAGING}/${comp}")
    file(TOUCH "${extract_marker}")
  endif()

  file(GLOB inner_root "${STAGING}/${comp}/${comp}-*-archive")
  if(NOT inner_root)
    message(FATAL_ERROR "expected ${comp}-*-archive inside ${STAGING}/${comp}")
  endif()
  set(${comp}_ROOT "${inner_root}")
endforeach()

set(libdev "")
set(libdev_lic "")
foreach(comp IN LISTS COMPONENTS)
  set(candidate "${${comp}_ROOT}/nvvm/libdevice/libdevice.10.bc")
  if(EXISTS "${candidate}")
    set(libdev "${candidate}")
    set(libdev_lic "${${comp}_ROOT}/LICENSE")
    break()
  endif()
endforeach()
if(NOT libdev)
  message(FATAL_ERROR "libdevice.10.bc not found in any of: ${COMPONENTS}")
endif()
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _ "${CUDA_VERSION}")
set(label "${CMAKE_MATCH_1}-${CMAKE_MATCH_2}")
file(COPY_FILE "${libdev}" "${OUT_DIR}/libdevice.10.cuda${label}.bc")
if(EXISTS "${libdev_lic}")
  file(COPY_FILE "${libdev_lic}" "${OUT_DIR}/libdevice.10.cuda${label}.bc.LICENSE")
endif()

message(STATUS "CUDA libdevice ${CUDA_VERSION} (${CUDA_ARCH}) -> libdevice.10.cuda${label}.bc")
