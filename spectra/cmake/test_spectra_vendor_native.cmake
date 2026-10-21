cmake_minimum_required(VERSION 3.23)

foreach(required BACKEND MODE SOURCE_DIR VENDOR_ROOT OUTPUT_DIR CONTAINER_RUNTIME CONTAINER_IMAGE CONTAINER_PLATFORM)
  if(NOT DEFINED ${required} OR NOT ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
if(NOT MODE MATCHES "^(compile|run)$")
  message(FATAL_ERROR "MODE must be compile or run, got: ${MODE}")
endif()
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Spectra vendor-native validation is supported only on native Linux hosts")
endif()
if(NOT EXISTS "${VENDOR_ROOT}/.staged" OR NOT EXISTS "${VENDOR_ROOT}/.manifest")
  message(FATAL_ERROR "Spectra vendor inputs are not staged and verified at ${VENDOR_ROOT}")
endif()

find_program(container_executable NAMES "${CONTAINER_RUNTIME}" REQUIRED)
find_program(id_executable id REQUIRED)
execute_process(COMMAND "${id_executable}" -u OUTPUT_VARIABLE uid OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${id_executable}" -g OUTPUT_VARIABLE gid OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)

get_filename_component(source_dir "${SOURCE_DIR}" ABSOLUTE)
get_filename_component(vendor_root "${VENDOR_ROOT}" ABSOLUTE)
get_filename_component(output_dir "${OUTPUT_DIR}" ABSOLUTE)
file(MAKE_DIRECTORY "${output_dir}")

set(container_prefix)
set(container_args
    run --rm --pull=missing --platform=${CONTAINER_PLATFORM}
    --env=HOME=/tmp
    --volume=${vendor_root}:/vendor:ro
    --volume=${source_dir}:/spectra:ro
    --volume=${output_dir}:/out)
foreach(environment SPECTRA_NATIVE_JOBS SPECTRA_HIP_ARCH)
  if(DEFINED ENV{${environment}} AND NOT "$ENV{${environment}}" STREQUAL "")
    list(APPEND container_args "--env=${environment}=$ENV{${environment}}")
  endif()
endforeach()
get_filename_component(container_name "${container_executable}" NAME)
execute_process(
  COMMAND "${container_executable}" --version
  OUTPUT_VARIABLE container_version
  ERROR_VARIABLE container_version_error
  RESULT_VARIABLE container_version_result)
if(NOT container_version_result EQUAL 0)
  message(FATAL_ERROR "failed to identify container runtime: ${container_version_error}")
endif()
string(TOLOWER "${container_version}" container_version)
if(container_name STREQUAL "podman" OR container_version MATCHES "podman")
  if(DEFINED ENV{SPECTRA_CDI_SPEC_DIR} AND IS_DIRECTORY "$ENV{SPECTRA_CDI_SPEC_DIR}")
    list(APPEND container_prefix "--cdi-spec-dir=$ENV{SPECTRA_CDI_SPEC_DIR}")
  endif()
  list(APPEND container_args --userns=keep-id --security-opt=label=disable)
  set(is_podman ON)
else()
  list(APPEND container_args --user=${uid}:${gid})
  set(is_podman OFF)
endif()

if(MODE STREQUAL "run")
  if(BACKEND STREQUAL "cuda")
    if(is_podman)
      list(APPEND container_args --device=nvidia.com/gpu=all)
    else()
      list(APPEND container_args --gpus=all)
    endif()
  elseif(BACKEND STREQUAL "hip")
    if(NOT EXISTS "/dev/kfd" OR NOT EXISTS "/dev/dri")
      message(FATAL_ERROR "HIP runtime validation requires /dev/kfd and /dev/dri")
    endif()
    list(APPEND container_args --device=/dev/kfd --device=/dev/dri --group-add=video --group-add=render)
  endif()
endif()

list(APPEND container_args --entrypoint=/spectra/tools/spectra_vendor_native.sh
     "${CONTAINER_IMAGE}" "${BACKEND}" "${MODE}")
message(STATUS "running ${BACKEND} vendor-native ${MODE} validation in ${CONTAINER_IMAGE}")
execute_process(
  COMMAND "${container_executable}" ${container_prefix} ${container_args}
  RESULT_VARIABLE result
  COMMAND_ECHO STDERR)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "${BACKEND} vendor-native ${MODE} validation failed (${result})")
endif()
