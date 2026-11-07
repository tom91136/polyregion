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

if(DEFINED PACKAGES)
  string(REPLACE "," ";" _packages "${PACKAGES}")
  set(_package_root "${_prefix}/${LIBDIR}/polyregion/packages")
  foreach(_package IN LISTS _packages)
    if(NOT EXISTS "${_package_root}/${_package}/lib.polyast")
      message(FATAL_ERROR "Spectra install has no ${_package} package under ${_package_root}")
    endif()
  endforeach()
  set(_package_include "${_package_root}/spectra/include")
  if(NOT EXISTS "${_package_include}/polyregion/spectra_api.hpp")
    message(FATAL_ERROR "Spectra install has no polyregion/spectra_api.hpp under ${_package_include}")
  endif()
  file(WRITE "${_source}/stdpar.cpp" [=[
#include <polyregion/spectra/stdpar.hpp>
int main() { return 0; }
]=])
  execute_process(
    COMMAND "${CXX_COMPILER}" -std=c++17 -fsyntax-only -isystem "${_package_include}" "${_source}/stdpar.cpp"
    RESULT_VARIABLE _stdpar_rc
    OUTPUT_VARIABLE _stdpar_stdout
    ERROR_VARIABLE _stdpar_stderr
  )
  if(NOT _stdpar_rc EQUAL 0)
    message(FATAL_ERROR "stdpar.hpp does not compile against the installed package headers (${_stdpar_rc}):\n${_stdpar_stdout}${_stdpar_stderr}")
  endif()
endif()

if(DEFINED POLYCPP)
  file(WRITE "${_source}/core.cpp" [=[
#include <algorithm>
#include <execution>
#include <vector>
void scale(std::vector<int> &xs) { std::for_each(std::execution::par_unseq, xs.begin(), xs.end(), [](int &x) { x *= 2; }); }
]=])
  set(_core_flags -std=c++17 -fstdpar "-fstdpar-arch=${CORE_ARCH}" -c "${_source}/core.cpp")
  foreach(_lookup IN ITEMS flag env)
    if(_lookup STREQUAL "flag")
      set(_command "${POLYCPP}" ${_core_flags} "-fstdpar-library-path=${_package_root}"
                   "-fstdpar-library-capabilities=${CORE_CAPABILITY}" -o "${_build}/core-flag.o")
    else()
      set(_command "${CMAKE_COMMAND}" -E env "POLYFRONT_LIBRARY_PATH=${_package_root}"
                   "POLYFRONT_LIBRARY_CAPABILITIES=${CORE_CAPABILITY}" "${POLYCPP}" ${_core_flags} -o "${_build}/core-env.o")
    endif()
    execute_process(
      COMMAND ${_command}
      RESULT_VARIABLE _core_rc
      OUTPUT_VARIABLE _core_stdout
      ERROR_VARIABLE _core_stderr
    )
    if(NOT _core_rc EQUAL 0)
      message(FATAL_ERROR "polycpp against the installed packages (${_lookup}) failed (${_core_rc}):\n${_core_stdout}${_core_stderr}")
    endif()
    string(FIND "${_core_stderr}" "bound to polyregion.core" _bound)
    if(_bound EQUAL -1)
      message(FATAL_ERROR "polycpp did not bind the installed polyregion.core package (${_lookup}):\n${_core_stderr}")
    endif()
  endforeach()
endif()
