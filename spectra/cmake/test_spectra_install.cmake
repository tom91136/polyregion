if(NOT DEFINED BUILD_DIR OR NOT DEFINED WORK_DIR OR NOT DEFINED CXX_COMPILER)
  message(FATAL_ERROR "BUILD_DIR, WORK_DIR and CXX_COMPILER are required")
endif()

set(_prefix "${WORK_DIR}/install")
set(_source "${WORK_DIR}/consumer")
set(_build "${WORK_DIR}/build")
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${_source}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${_prefix}"
  RESULT_VARIABLE _install_rc
  OUTPUT_VARIABLE _install_stdout
  ERROR_VARIABLE _install_stderr
)
if(NOT _install_rc EQUAL 0)
  message(FATAL_ERROR "Spectra install failed (${_install_rc}):\n${_install_stdout}${_install_stderr}")
endif()

file(WRITE "${_source}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(spectra_install_consumer LANGUAGES CXX)
find_package(spectra CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE spectra::spectra)
]=])
file(WRITE "${_source}/main.cpp" [=[
#include <polyregion/spectra.hpp>
int main() { return 0; }
]=])

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${_source}" -B "${_build}"
          "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_PREFIX_PATH=${_prefix}"
  RESULT_VARIABLE _configure_rc
  OUTPUT_VARIABLE _configure_stdout
  ERROR_VARIABLE _configure_stderr
)
if(NOT _configure_rc EQUAL 0)
  message(FATAL_ERROR "Spectra install consumer configure failed (${_configure_rc}):\n${_configure_stdout}${_configure_stderr}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${_build}"
  RESULT_VARIABLE _build_rc
  OUTPUT_VARIABLE _build_stdout
  ERROR_VARIABLE _build_stderr
)
if(NOT _build_rc EQUAL 0)
  message(FATAL_ERROR "Spectra install consumer build failed (${_build_rc}):\n${_build_stdout}${_build_stderr}")
endif()
