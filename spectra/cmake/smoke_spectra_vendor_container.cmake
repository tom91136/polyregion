cmake_minimum_required(VERSION 3.23)

foreach(required BACKEND VENDOR_ROOT OUTPUT_DIR CONTAINER_RUNTIME CONTAINER_IMAGE CONTAINER_PLATFORM)
  if(NOT DEFINED ${required} OR NOT ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Spectra vendor container smoke is supported only on native Linux hosts")
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

get_filename_component(smoke_source_dir "${CMAKE_CURRENT_LIST_DIR}/vendor_smoke" ABSOLUTE)
get_filename_component(vendor_root "${VENDOR_ROOT}" ABSOLUTE)
get_filename_component(output_dir "${OUTPUT_DIR}" ABSOLUTE)
file(MAKE_DIRECTORY "${output_dir}")

set(container_args
    run --rm --pull=missing --platform=${CONTAINER_PLATFORM}
    --env=HOME=/tmp
    --volume=${vendor_root}:/vendor:ro
    --volume=${smoke_source_dir}:/smoke:ro
    --volume=${output_dir}:/out)
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
  list(APPEND container_args --userns=keep-id --security-opt=label=disable)
else()
  list(APPEND container_args --user=${uid}:${gid})
endif()

if(BACKEND STREQUAL "cuda")
  list(APPEND container_args --entrypoint=nvcc "${CONTAINER_IMAGE}"
       --allow-unsupported-compiler -std=c++17 -I/vendor/cuda_cccl/include
       -c /smoke/cuda.cu -o /out/cuda.o)
elseif(BACKEND STREQUAL "hip")
  list(APPEND container_args --entrypoint=/opt/rocm/bin/hipcc "${CONTAINER_IMAGE}"
       --offload-arch=gfx90a -std=c++17
       -I/vendor/rocm/rocThrust -I/vendor/rocm/rocPRIM/rocprim/include
       -c /smoke/hip.cpp -o /out/hip.o)
elseif(BACKEND STREQUAL "onedpl")
  list(APPEND container_args --entrypoint=icpx "${CONTAINER_IMAGE}"
       -fsycl -std=c++17 -I/vendor/onedpl/include
       -fsyntax-only /smoke/onedpl.cpp)
else()
  message(FATAL_ERROR "unknown Spectra vendor smoke backend: ${BACKEND}")
endif()

message(STATUS "compiling ${BACKEND} vendor smoke in ${CONTAINER_IMAGE}")
execute_process(
  COMMAND "${container_executable}" ${container_args}
  RESULT_VARIABLE result
  COMMAND_ECHO STDERR)
if(NOT result EQUAL 0)
  file(REMOVE "${output_dir}/${BACKEND}.o")
  message(FATAL_ERROR "${BACKEND} vendor container smoke failed (${result})")
endif()
message(STATUS "${BACKEND} vendor container smoke passed")
