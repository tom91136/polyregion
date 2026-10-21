cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
option(SPECTRA_VENDOR_ONLY "Configure only Spectra vendor acquisition and validation targets" OFF)
if(SPECTRA_VENDOR_ONLY)
  project(spectra_vendor LANGUAGES NONE)
else()
  project(spectra LANGUAGES CXX)
  # MinGW gfortran is the only Fortran on stock Windows hosts and cannot link in an MSVC environment
  if(NOT MSVC)
    include(CheckLanguage)
    check_language(Fortran)
  endif()
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
  endif()
endif()

enable_testing()

if(NOT SPECTRA_VENDOR_ONLY)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  include(ProcessorCount)
  ProcessorCount(_spectra_processor_count)
  if(NOT _spectra_processor_count)
    set(_spectra_processor_count 1)
  endif()
  option(SPECTRA_ENABLE_PACKAGE "Build the Spectra PolyAST package" OFF)
  option(SPECTRA_ENABLE_PACKAGE_TESTS "Register the generated package-import tests" ${SPECTRA_ENABLE_PACKAGE})
  set(SPECTRA_TEST_PROFILE "$ENV{POLYREGION_TEST_PROFILE}" CACHE STRING
      "Polyregion test profile selecting the tested targets; empty resolves by hostname")
  set(SPECTRA_PACKAGE_CUDA_THRUST "" CACHE STRING "Include the CUDA Thrust implementation; empty follows the test profile")
  set(SPECTRA_PACKAGE_ROCTHRUST "" CACHE STRING "Include the rocThrust implementation; empty follows the test profile")
  set(SPECTRA_PACKAGE_ONEDPL "" CACHE STRING "Include the oneDPL implementation; empty follows the test profile")
  set(SPECTRA_CUDA_PATH "" CACHE PATH "Producer CUDA toolkit root; empty uses the staged vendor toolkit")
  set(SPECTRA_CUDA_ARCH "" CACHE STRING "CUDA producer architecture; empty follows the test profile")
  set(SPECTRA_CUDA_MAX_ARCH 86 CACHE STRING "Newest sm_ architecture the producer CUDA toolkit supports")
  if(EXISTS "/usr/lib64/rocm/llvm/lib/clang")
    set(_spectra_default_rocm_path "/usr/lib64/rocm")
  else()
    set(_spectra_default_rocm_path "/opt/rocm")
  endif()
  set(SPECTRA_ROCM_PATH "${_spectra_default_rocm_path}" CACHE PATH "ROCm toolkit root")
  set(SPECTRA_ROCM_DEVICE_LIB_PATH "" CACHE PATH "Optional ROCm device bitcode directory")
  set(SPECTRA_ROCM_ARCH "" CACHE STRING "ROCm producer architecture; empty follows the test profile")
  set(SPECTRA_PACKAGE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/packages" CACHE PATH "Generated package root")
  set(SPECTRA_PREBUILT_PACKAGE "" CACHE FILEPATH "Existing Spectra package used instead of compiling implementations")
  set(SPECTRA_JAVA_HOME "$ENV{JAVA_HOME}" CACHE PATH "JDK that runs SpectraCodeGen; empty uses java on PATH")

  if(SPECTRA_ENABLE_PACKAGE)
    if(SPECTRA_PREBUILT_PACKAGE)
      find_package(Polyregion CONFIG REQUIRED)
    else()
      find_package(Polyregion CONFIG REQUIRED COMPONENTS PackageCompiler)
    endif()

    function(_spectra_target_location output target)
      get_target_property(_location "${target}" IMPORTED_LOCATION)
      if(NOT _location)
        message(FATAL_ERROR "${target} has no imported location")
      endif()
      set("${output}" "${_location}" PARENT_SCOPE)
    endfunction()

    _spectra_target_location(_spectra_polycpp Polyregion::polycpp)
    _spectra_target_location(_spectra_polyc Polyregion::polyc)
    set(_spectra_polypass "")
    if(TARGET Polyregion::polypass)
      _spectra_target_location(_spectra_polypass Polyregion::polypass)
    endif()
    _spectra_target_location(_spectra_clangxx Polyregion::clangxx)
    set(_spectra_polycpp_plugin "")
    if(TARGET Polyregion::polycpp-plugin)
      _spectra_target_location(_spectra_polycpp_plugin Polyregion::polycpp-plugin)
    endif()
    set(_spectra_package_tools
        "${_spectra_polycpp}" "${_spectra_polyc}" "${_spectra_polypass}" "${_spectra_clangxx}")
    if(_spectra_polycpp_plugin)
      list(APPEND _spectra_package_tools "${_spectra_polycpp_plugin}")
    endif()
    _spectra_target_location(_spectra_polytest Polyregion::polytest)

    list(JOIN Polyregion_RUNTIME_LIBRARY_DIRS ":" _spectra_runtime_library_path)
    if(APPLE)
      set(_spectra_tool_library_env
          "DYLD_FALLBACK_LIBRARY_PATH=${_spectra_runtime_library_path}:$ENV{DYLD_FALLBACK_LIBRARY_PATH}")
    elseif(UNIX)
      set(_spectra_tool_library_env
          "LD_LIBRARY_PATH=${_spectra_runtime_library_path}:$ENV{LD_LIBRARY_PATH}")
    else()
      set(_spectra_tool_library_env "")
    endif()

    set(_spectra_suite "${CMAKE_CURRENT_BINARY_DIR}/spectra.suite.env")
    set(_spectra_suite_profile "")
    if(SPECTRA_TEST_PROFILE)
      set(_spectra_suite_profile "POLYTEST_DEFAULT_ENV+=:POLYREGION_TEST_PROFILE=${SPECTRA_TEST_PROFILE}\n")
    endif()
    file(WRITE "${_spectra_suite}" "${_spectra_suite_profile}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "POLYTEST_SUITE=${_spectra_suite}" "${_spectra_polytest}" --list-targets
        OUTPUT_VARIABLE _spectra_profile_targets
        RESULT_VARIABLE _spectra_profile_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _spectra_profile_result EQUAL 0)
      message(FATAL_ERROR "polytest --list-targets failed (${_spectra_profile_result})")
    endif()
    string(REPLACE "\n" ";" _spectra_profile_targets "${_spectra_profile_targets}")
    message(STATUS "Spectra test targets: ${_spectra_profile_targets}")

    set(_spectra_profile_cuda_arch "")
    set(_spectra_profile_rocm_arch "")
    set(_spectra_profile_dpl OFF)
    foreach(_target IN LISTS _spectra_profile_targets)
      if(NOT _target MATCHES "^([^@!]+)@([^!]*)")
        continue()
      endif()
      string(TOLOWER "${CMAKE_MATCH_1}" _backend)
      set(_arch "${CMAKE_MATCH_2}")
      if(_backend STREQUAL "cuda" AND NOT _spectra_profile_cuda_arch)
        set(_spectra_profile_cuda_arch "${_arch}")
      elseif(_backend MATCHES "^(hip|hsa)$" AND NOT _spectra_profile_rocm_arch)
        set(_spectra_profile_rocm_arch "${_arch}")
      elseif(_backend MATCHES "^(opencl|opencl1_1|opencl2_0|spirv64_kernel|spirv_glcompute|vulkan|vulkan_spirv|metal|metal1_0)$")
        set(_spectra_profile_dpl ON)
      endif()
    endforeach()
    set(_spectra_CUDA_THRUST "${SPECTRA_PACKAGE_CUDA_THRUST}")
    set(_spectra_ROCTHRUST "${SPECTRA_PACKAGE_ROCTHRUST}")
    set(_spectra_ONEDPL "${SPECTRA_PACKAGE_ONEDPL}")
    if(_spectra_CUDA_THRUST STREQUAL "")
      set(_spectra_CUDA_THRUST OFF)
      if(_spectra_profile_cuda_arch)
        set(_spectra_CUDA_THRUST ON)
      endif()
    endif()
    if(_spectra_ROCTHRUST STREQUAL "")
      set(_spectra_ROCTHRUST OFF)
      if(_spectra_profile_rocm_arch)
        set(_spectra_ROCTHRUST ON)
      endif()
    endif()
    if(_spectra_ONEDPL STREQUAL "")
      set(_spectra_ONEDPL ${_spectra_profile_dpl})
    endif()
    set(_spectra_cuda_arch "${SPECTRA_CUDA_ARCH}")
    if(NOT _spectra_cuda_arch)
      set(_spectra_cuda_arch "${_spectra_profile_cuda_arch}")
      if(_spectra_cuda_arch MATCHES "^sm_([0-9]+)$" AND CMAKE_MATCH_1 GREATER SPECTRA_CUDA_MAX_ARCH)
        set(_spectra_cuda_arch "sm_${SPECTRA_CUDA_MAX_ARCH}")
      endif()
    endif()
    set(_spectra_rocm_arch "${SPECTRA_ROCM_ARCH}")
    if(NOT _spectra_rocm_arch)
      set(_spectra_rocm_arch "${_spectra_profile_rocm_arch}")
    endif()
    if(_spectra_CUDA_THRUST AND NOT _spectra_cuda_arch)
      message(FATAL_ERROR "The CUDA Thrust implementation needs SPECTRA_CUDA_ARCH or a cuda@ target in the test profile")
    endif()
    if(_spectra_ROCTHRUST AND NOT _spectra_rocm_arch)
      message(FATAL_ERROR "The rocThrust implementation needs SPECTRA_ROCM_ARCH or a hip@/hsa@ target in the test profile")
    endif()
    message(STATUS "Spectra implementations: thrust=${_spectra_CUDA_THRUST} (${_spectra_cuda_arch}) "
                   "rocthrust=${_spectra_ROCTHRUST} (${_spectra_rocm_arch}) onedpl=${_spectra_ONEDPL}")
  endif()
  if(SPECTRA_PREBUILT_PACKAGE)
    get_filename_component(_spectra_package "${SPECTRA_PREBUILT_PACKAGE}" ABSOLUTE
                           BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    get_filename_component(_spectra_package_dir "${_spectra_package}" DIRECTORY)
    get_filename_component(_spectra_package_name "${_spectra_package}" NAME)
    get_filename_component(_spectra_package_namespace "${_spectra_package_dir}" NAME)
    get_filename_component(SPECTRA_PACKAGE_ROOT "${_spectra_package_dir}" DIRECTORY)
    if(NOT _spectra_package_name STREQUAL "lib.polyast" OR NOT _spectra_package_namespace STREQUAL "spectra")
      message(FATAL_ERROR "SPECTRA_PREBUILT_PACKAGE must name a <package-root>/spectra/lib.polyast file")
    endif()
    if(NOT EXISTS "${_spectra_package}")
      message(FATAL_ERROR "SPECTRA_PREBUILT_PACKAGE does not exist: ${_spectra_package}")
    endif()
  else()
    set(_spectra_package "${SPECTRA_PACKAGE_ROOT}/spectra/lib.polyast")
  endif()
endif()

if(NOT SPECTRA_VENDOR_ONLY)
  add_library(spectra INTERFACE)
  add_library(spectra::spectra ALIAS spectra)
  target_compile_features(spectra INTERFACE cxx_std_17)
  target_include_directories(
    spectra
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
              $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generated/cpp/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  add_library(spectra_cpp_consumer OBJECT test/cpp/check_surface.cpp)
  target_link_libraries(spectra_cpp_consumer PRIVATE spectra::spectra)

  if(CMAKE_Fortran_COMPILER)
    add_library(
      spectra_fortran_consumer OBJECT
      generated/fortran/spectra_api.f90
      test/fortran/check_surface.f90
    )
  endif()

  install(TARGETS spectra EXPORT spectra-targets)
  install(
    FILES generated/cpp/include/polyregion/spectra_api.hpp
          include/polyregion/spectra.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/polyregion
  )
  install(
    DIRECTORY include/polyregion/spectra/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/polyregion/spectra
  )
  install(
    EXPORT spectra-targets
    FILE spectra-targets.cmake
    NAMESPACE spectra::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spectra
  )
  configure_package_config_file(
    cmake/spectra-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/spectra-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spectra
  )
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/spectra-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spectra
  )
  add_test(
    NAME spectra.install-consumer
    COMMAND ${CMAKE_COMMAND}
            -DBUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/install-consumer-test
            -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_spectra_install.cmake
  )
  set_tests_properties(spectra.install-consumer PROPERTIES LABELS "spectra;install")
endif()

include(cmake/spectra_vendor_versions.cmake)
set(SPECTRA_VENDOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vendor" CACHE PATH
    "Location of Spectra's staged upstream vendor distributions")
set(SPECTRA_CONTAINER_RUNTIME "docker" CACHE STRING "OCI runtime used for Spectra vendor smoke tests")

set(_spectra_vendor_platform "")
set(_spectra_container_platform "")
set(_spectra_container_backends "")
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(_spectra_vendor_platform "linux-x86_64")
    set(_spectra_cuda_cccl_sha256 "${SPECTRA_CUDA_CCCL_LINUX_X86_64_SHA256}")
    set(_spectra_cuda_cudart_sha256 "${SPECTRA_CUDA_CUDART_LINUX_X86_64_SHA256}")
    set(_spectra_cuda_nvcc_sha256 "${SPECTRA_CUDA_NVCC_LINUX_X86_64_SHA256}")
    set(_spectra_container_platform "linux/amd64")
    set(_spectra_container_backends cuda hip onedpl)
  elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_spectra_vendor_platform "linux-sbsa")
    set(_spectra_cuda_cccl_sha256 "${SPECTRA_CUDA_CCCL_LINUX_SBSA_SHA256}")
    set(_spectra_cuda_cudart_sha256 "${SPECTRA_CUDA_CUDART_LINUX_SBSA_SHA256}")
    set(_spectra_cuda_nvcc_sha256 "${SPECTRA_CUDA_NVCC_LINUX_SBSA_SHA256}")
    set(_spectra_container_platform "linux/arm64")
    set(_spectra_container_backends cuda)
  endif()
endif()

if(_spectra_vendor_platform)
  set(
    _spectra_vendor_args
    -DOUT_DIR=${SPECTRA_VENDOR_ROOT}
    -DONEDPL_COMMIT=${SPECTRA_ONEDPL_COMMIT}
    -DONEDPL_SHA256=${SPECTRA_ONEDPL_SHA256}
    -DROCTHRUST_COMMIT=${SPECTRA_ROCTHRUST_COMMIT}
    -DROCTHRUST_SHA256=${SPECTRA_ROCTHRUST_SHA256}
    -DROCPRIM_COMMIT=${SPECTRA_ROCPRIM_COMMIT}
    -DROCPRIM_SHA256=${SPECTRA_ROCPRIM_SHA256}
    -DCUDA_CCCL_VERSION=${SPECTRA_CUDA_CCCL_VERSION}
    -DCUDA_CCCL_PLATFORM=${_spectra_vendor_platform}
    -DCUDA_CCCL_SHA256=${_spectra_cuda_cccl_sha256}
    -DCUDA_CUDART_VERSION=${SPECTRA_CUDA_CUDART_VERSION}
    -DCUDA_CUDART_SHA256=${_spectra_cuda_cudart_sha256}
    -DCUDA_NVCC_VERSION=${SPECTRA_CUDA_NVCC_VERSION}
    -DCUDA_NVCC_SHA256=${_spectra_cuda_nvcc_sha256}
  )

  set(_spectra_vendor_stamp "${SPECTRA_VENDOR_ROOT}/.staged")
  add_custom_command(
    OUTPUT "${_spectra_vendor_stamp}"
    COMMAND ${CMAKE_COMMAND} ${_spectra_vendor_args}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_vendor_dist.cmake
    COMMAND ${CMAKE_COMMAND} -E touch "${_spectra_vendor_stamp}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_vendor_dist.cmake
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/spectra_vendor_versions.cmake
    COMMENT "Staging Spectra vendor distributions"
    VERBATIM
  )
  add_custom_target(
    spectra-vendor
    COMMAND ${CMAKE_COMMAND} ${_spectra_vendor_args}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_vendor_dist.cmake
    DEPENDS "${_spectra_vendor_stamp}"
    COMMENT "Validating and staging Spectra vendor distributions"
    VERBATIM
  )
  add_custom_target(
    spectra-vendor-verify
    COMMAND ${CMAKE_COMMAND} ${_spectra_vendor_args} -DSPECTRA_VENDOR_VERIFY_ONLY=ON
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_vendor_dist.cmake
    COMMENT "Verifying Spectra vendor distributions"
    VERBATIM
  )

  add_custom_target(spectra-vendor-container-smoke)
  add_custom_target(spectra-vendor-native-validation)
  foreach(_backend IN LISTS _spectra_container_backends)
    string(TOUPPER "${_backend}" _backend_upper)
    add_custom_target(
      spectra-vendor-container-smoke-${_backend}
      COMMAND
        ${CMAKE_COMMAND} -DBACKEND=${_backend} -DVENDOR_ROOT=${SPECTRA_VENDOR_ROOT}
        -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/vendor-container-smoke
        -DCONTAINER_RUNTIME=${SPECTRA_CONTAINER_RUNTIME}
        -DCONTAINER_IMAGE=${SPECTRA_${_backend_upper}_CONTAINER}
        -DCONTAINER_PLATFORM=${_spectra_container_platform}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/smoke_spectra_vendor_container.cmake
      COMMENT "Compiling ${_backend} vendor smoke in its pinned official container"
      VERBATIM
    )
    add_dependencies(spectra-vendor-container-smoke-${_backend} spectra-vendor)
    add_dependencies(spectra-vendor-container-smoke spectra-vendor-container-smoke-${_backend})

    add_custom_target(
      spectra-vendor-native-validation-${_backend}
      COMMAND
        ${CMAKE_COMMAND} -DBACKEND=${_backend} -DMODE=compile
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DVENDOR_ROOT=${SPECTRA_VENDOR_ROOT}
        -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/vendor-container-smoke
        -DCONTAINER_RUNTIME=${SPECTRA_CONTAINER_RUNTIME}
        -DCONTAINER_IMAGE=${SPECTRA_${_backend_upper}_CONTAINER}
        -DCONTAINER_PLATFORM=${_spectra_container_platform}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_spectra_vendor_native.cmake
      COMMENT "Compiling and linking all ${_backend} vendor-native validation cases"
      VERBATIM
    )
    add_dependencies(spectra-vendor-native-validation-${_backend} spectra-vendor)
    add_dependencies(spectra-vendor-native-validation spectra-vendor-native-validation-${_backend})

    add_custom_target(
      spectra-vendor-native-validation-run-${_backend}
      COMMAND
        ${CMAKE_COMMAND} -DBACKEND=${_backend} -DMODE=run
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DVENDOR_ROOT=${SPECTRA_VENDOR_ROOT}
        -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/vendor-container-smoke
        -DCONTAINER_RUNTIME=${SPECTRA_CONTAINER_RUNTIME}
        -DCONTAINER_IMAGE=${SPECTRA_${_backend_upper}_CONTAINER}
        -DCONTAINER_PLATFORM=${_spectra_container_platform}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_spectra_vendor_native.cmake
      COMMENT "Compiling and running all ${_backend} vendor-native validation cases"
      VERBATIM
    )
    add_dependencies(spectra-vendor-native-validation-run-${_backend} spectra-vendor)
  endforeach()
else()
  message(STATUS
          "Spectra vendor staging and container smoke are unavailable on ${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()

add_custom_target(
  spectra-vendor-staging-test
  COMMAND
    ${CMAKE_COMMAND} -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/vendor-staging-test
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_spectra_vendor_dist.cmake
  COMMENT "Testing Spectra vendor staging transactions"
  VERBATIM
)

if(NOT SPECTRA_VENDOR_ONLY AND SPECTRA_ENABLE_PACKAGE)
  if(SPECTRA_PREBUILT_PACKAGE)
    add_custom_target(spectra-package ALL DEPENDS "${_spectra_package}")
  else()
    set(_spectra_producers)
    if(_spectra_CUDA_THRUST)
      list(APPEND _spectra_producers thrust)
    endif()
    if(_spectra_ROCTHRUST)
      list(APPEND _spectra_producers rocthrust)
    endif()
    if(_spectra_ONEDPL)
      list(APPEND _spectra_producers onedpl)
    endif()
    if(NOT _spectra_producers)
      message(FATAL_ERROR "SPECTRA_ENABLE_PACKAGE requires at least one implementation")
    endif()
    if(NOT _spectra_vendor_platform)
      message(FATAL_ERROR "Producing the Spectra package needs staged vendor sources, which are Linux-only; "
                          "pass SPECTRA_PREBUILT_PACKAGE on ${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_HOST_SYSTEM_PROCESSOR}")
    endif()

    set(_spectra_interface "${CMAKE_CURRENT_SOURCE_DIR}/generated/package/spectra.interface.polyast")
    set(_spectra_java_env)
    if(SPECTRA_JAVA_HOME)
      set(_spectra_java_env "JAVA_HOME=${SPECTRA_JAVA_HOME}" "PATH=${SPECTRA_JAVA_HOME}/bin:$ENV{PATH}")
    endif()
    file(GLOB_RECURSE _spectra_codegen_inputs CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/src/main/scala/*.scala" "${Polyregion_MAVEN_REPOSITORY}/*.jar")
    add_custom_command(
      OUTPUT "${_spectra_interface}"
      COMMAND ${CMAKE_COMMAND} -E env ${_spectra_java_env} "POLYREGION_DIST=${Polyregion_ROOT}"
              bash "${CMAKE_CURRENT_SOURCE_DIR}/sbtx" -no-colors -batch -J-Dsbt.server.autostart=false
              genSpectraInterface
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      DEPENDS ${_spectra_codegen_inputs}
              "${CMAKE_CURRENT_SOURCE_DIR}/build.sbt"
              "${CMAKE_CURRENT_SOURCE_DIR}/project/build.properties"
              "${CMAKE_CURRENT_SOURCE_DIR}/project/plugins.sbt"
      COMMENT "Generating the Spectra interface package"
      VERBATIM
    )

    file(GLOB _spectra_impl_includes CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/implementation/*.inc")
    set(_spectra_linked_fragments)
    set(_spectra_fragment_index 0)
    set(_spectra_cuda_overlay "${CMAKE_CURRENT_BINARY_DIR}/cuda-package-overlay")
    set(_spectra_cuda_path "${SPECTRA_CUDA_PATH}")
    if(NOT _spectra_cuda_path)
      set(_spectra_cuda_path "${SPECTRA_VENDOR_ROOT}/cuda")
    endif()
    set(_spectra_rocthrust_overlay "${CMAKE_CURRENT_BINARY_DIR}/rocthrust-package-overlay")
    set(_spectra_onedpl_overlay "${CMAKE_CURRENT_BINARY_DIR}/onedpl-package-overlay")
    if(_spectra_CUDA_THRUST)
      set(_spectra_cuda_overlay_stamp "${_spectra_cuda_overlay}/.prepared")
      add_custom_command(
        OUTPUT "${_spectra_cuda_overlay_stamp}"
        COMMAND ${CMAKE_COMMAND} -DBACKEND=cuda -DVENDOR_ROOT=${SPECTRA_VENDOR_ROOT}/cuda_cccl/include
                -DOUT_DIR=${_spectra_cuda_overlay}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_package_overlay.cmake
        COMMAND ${CMAKE_COMMAND} -E touch "${_spectra_cuda_overlay_stamp}"
        DEPENDS "${_spectra_vendor_stamp}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_package_overlay.cmake"
        COMMENT "Preparing the derived CUDA package overlay"
        VERBATIM
      )
    endif()
    if(_spectra_ROCTHRUST)
      set(_spectra_rocthrust_overlay_stamp "${_spectra_rocthrust_overlay}/.prepared")
      add_custom_command(
        OUTPUT "${_spectra_rocthrust_overlay_stamp}"
        COMMAND ${CMAKE_COMMAND} -DBACKEND=hip -DVENDOR_ROOT=${SPECTRA_VENDOR_ROOT}/rocm/rocThrust
                -DROCPRIM_ROOT=${SPECTRA_VENDOR_ROOT}/rocm/rocPRIM/rocprim/include
                -DOUT_DIR=${_spectra_rocthrust_overlay}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_package_overlay.cmake
        COMMAND ${CMAKE_COMMAND} -E touch "${_spectra_rocthrust_overlay_stamp}"
        DEPENDS "${_spectra_vendor_stamp}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_package_overlay.cmake"
        COMMENT "Preparing the derived HIP package overlay"
        VERBATIM
      )
    endif()
    if(_spectra_ONEDPL)
      set(_spectra_onedpl_overlay_stamp "${_spectra_onedpl_overlay}/.prepared")
      add_custom_command(
        OUTPUT "${_spectra_onedpl_overlay_stamp}"
        COMMAND ${CMAKE_COMMAND} -DBACKEND=dpl -DVENDOR_ROOT=${SPECTRA_VENDOR_ROOT}/onedpl/include
                -DOUT_DIR=${_spectra_onedpl_overlay}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_package_overlay.cmake
        COMMAND ${CMAKE_COMMAND} -E touch "${_spectra_onedpl_overlay_stamp}"
        DEPENDS "${_spectra_vendor_stamp}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_spectra_package_overlay.cmake"
        COMMENT "Preparing the derived oneDPL package overlay"
        VERBATIM
      )
    endif()

    foreach(_producer IN LISTS _spectra_producers)
      set(_spectra_part_count 8)
      if(_producer STREQUAL "thrust")
        set(_source "${CMAKE_CURRENT_SOURCE_DIR}/src/implementation/thrust.cu")
        set(_capability spectra.thrust)
        set(_producer_impl_header "${CMAKE_CURRENT_SOURCE_DIR}/include/polyregion/spectra/implementation/thrust.hpp")
        set(_producer_deps "${_spectra_cuda_overlay_stamp}")
        set(_flags
            -fstdpar-arch=cuda@${_spectra_cuda_arch}
            -x cuda --cuda-gpu-arch=${_spectra_cuda_arch} --cuda-path=${_spectra_cuda_path}
            -Wno-error=missing-template-arg-list-after-template-kw
            -I ${_spectra_cuda_overlay}
            -I ${SPECTRA_VENDOR_ROOT}/cuda_cccl/include)
      elseif(_producer STREQUAL "rocthrust")
        set(_source "${CMAKE_CURRENT_SOURCE_DIR}/src/implementation/rocthrust.hip")
        set(_capability spectra.hip)
        set(_producer_impl_header "${CMAKE_CURRENT_SOURCE_DIR}/include/polyregion/spectra/implementation/thrust.hpp")
        set(_producer_deps "${_spectra_rocthrust_overlay_stamp}")
        set(_spectra_rocm_device_lib_flags)
        if(SPECTRA_ROCM_DEVICE_LIB_PATH)
          list(APPEND _spectra_rocm_device_lib_flags "--rocm-device-lib-path=${SPECTRA_ROCM_DEVICE_LIB_PATH}")
        elseif(NOT EXISTS "${SPECTRA_ROCM_PATH}/amdgcn/bitcode/ocml.bc")
          file(GLOB _spectra_rocm_device_lib_candidates LIST_DIRECTORIES true
               "${SPECTRA_ROCM_PATH}/llvm/lib/clang/*/lib/amdgcn/bitcode"
               "/usr/lib64/rocm/llvm/lib/clang/*/lib/amdgcn/bitcode")
          list(SORT _spectra_rocm_device_lib_candidates COMPARE NATURAL ORDER DESCENDING)
          foreach(_candidate IN LISTS _spectra_rocm_device_lib_candidates)
            if(EXISTS "${_candidate}/ocml.bc")
              list(APPEND _spectra_rocm_device_lib_flags "--rocm-device-lib-path=${_candidate}")
              break()
            endif()
          endforeach()
        endif()
        set(_flags
            -fstdpar-arch=hsa@${_spectra_rocm_arch}
            -x hip --offload-arch=${_spectra_rocm_arch} --rocm-path=${SPECTRA_ROCM_PATH}
            ${_spectra_rocm_device_lib_flags}
            -DROCPRIM_DISABLE_DPP=1 -DROCPRIM_NAVI=1 -DTHRUST_HIP_PRINTF_ENABLED=0
            -I ${_spectra_rocthrust_overlay}
            -I ${SPECTRA_VENDOR_ROOT}/rocm/rocThrust
            -I ${SPECTRA_VENDOR_ROOT}/rocm/rocPRIM/rocprim/include)
      else()
        set(_source "${CMAKE_CURRENT_SOURCE_DIR}/src/implementation/onedpl.cpp")
        set(_capability spectra.dpl)
        set(_producer_impl_header "${CMAKE_CURRENT_SOURCE_DIR}/include/polyregion/spectra/implementation/onedpl.hpp")
        file(GLOB_RECURSE _spectra_onedpl_shim CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/package/onedpl/include/*")
        set(_producer_deps "${_spectra_onedpl_overlay_stamp}" ${_spectra_onedpl_shim})
        set(_flags
            -fstdpar-arch=spirv64_kernel@host -fsycl -fsycl-device-only -x c++
            -D_ONEDPL_COMPILE_KERNEL=0 -Oz
            -isystem ${CMAKE_CURRENT_SOURCE_DIR}/src/package/onedpl/include
            -I ${_spectra_onedpl_overlay}
            -I ${SPECTRA_VENDOR_ROOT}/onedpl/include)
      endif()

      foreach(_part RANGE 1 ${_spectra_part_count})
        set(_base "${CMAKE_CURRENT_BINARY_DIR}/fragments/${_producer}-${_part}")
        set(_program "${_base}.polyast")
        add_custom_command(
          OUTPUT "${_program}"
          BYPRODUCTS "${_base}.o"
          COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/fragments"
          COMMAND ${CMAKE_COMMAND} -E env ${_spectra_tool_library_env} "POLYPASS_PLUGINS=${_spectra_polypass}"
                  "${_spectra_polycpp}" --driver "${_spectra_clangxx}"
                  -std=c++17 -w -DNDEBUG -DSPECTRA_PART=${_part}
                  -I ${CMAKE_CURRENT_SOURCE_DIR}/include -I ${CMAKE_CURRENT_SOURCE_DIR}/src/implementation
                  -fstdpar -fstdpar-mem=direct -fstdpar-emit-library=${_program} ${_flags}
                  -c "${_source}" -o "${_base}.o"
          DEPENDS "${_source}" ${_spectra_impl_includes} "${_producer_impl_header}" ${_producer_deps}
                  ${_spectra_package_tools}
          COMMENT "Compiling Spectra ${_producer} fragment ${_part}/${_spectra_part_count}"
          VERBATIM
        )
        set(_linked "${_base}.package.polyast")
        add_custom_command(
          OUTPUT "${_linked}"
          COMMAND ${CMAKE_COMMAND} -E env ${_spectra_tool_library_env} "POLYPASS_PLUGINS=${_spectra_polypass}"
                  "${_spectra_polyc}" package link-fragment "--capabilities=${_capability}"
                  "${_spectra_interface}"
                  ${_spectra_fragment_index} "${_program}" "${_linked}"
          DEPENDS "${_spectra_interface}"
                  "${_program}" "${_spectra_polyc}" "${_spectra_polypass}"
          COMMENT "Linking Spectra ${_producer} fragment ${_part}/${_spectra_part_count}"
          VERBATIM
        )
        list(APPEND _spectra_linked_fragments "${_linked}")
        math(EXPR _spectra_fragment_index "${_spectra_fragment_index} + 1")
      endforeach()
    endforeach()

    add_custom_command(
      OUTPUT "${_spectra_package}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${SPECTRA_PACKAGE_ROOT}"
      COMMAND ${CMAKE_COMMAND} -E env ${_spectra_tool_library_env} "POLYPASS_PLUGINS=${_spectra_polypass}"
              "${_spectra_polyc}" package pack
              "${_spectra_interface}"
              "${SPECTRA_PACKAGE_ROOT}" ${_spectra_linked_fragments}
      DEPENDS "${_spectra_interface}"
              ${_spectra_linked_fragments} "${_spectra_polyc}"
      COMMENT "Packing the Spectra package"
      VERBATIM
    )
    add_custom_target(spectra-package ALL DEPENDS "${_spectra_package}")
  endif()
  if(TARGET spectra-vendor AND NOT SPECTRA_PREBUILT_PACKAGE)
    add_dependencies(spectra-package spectra-vendor)
  endif()
  install(FILES "${_spectra_package}" DESTINATION "${CMAKE_INSTALL_LIBDIR}/polyregion/packages/spectra")
endif()

if(NOT SPECTRA_VENDOR_ONLY AND SPECTRA_ENABLE_PACKAGE_TESTS)
  if(NOT SPECTRA_ENABLE_PACKAGE)
    message(FATAL_ERROR "SPECTRA_ENABLE_PACKAGE_TESTS requires SPECTRA_ENABLE_PACKAGE")
  endif()
  file(GLOB _spectra_generated_tests CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test/generated/*.cpp")
  if(NOT _spectra_generated_tests)
    message(FATAL_ERROR "Spectra package matrix has no generated cases")
  endif()

  # Each implementation serves only the compile targets it was produced for; unserved targets are not tested.
  set(_spectra_capabilities)
  if(_spectra_CUDA_THRUST OR SPECTRA_PREBUILT_PACKAGE)
    list(APPEND _spectra_capabilities cuda=spectra.thrust)
  endif()
  if(_spectra_ROCTHRUST OR SPECTRA_PREBUILT_PACKAGE)
    list(APPEND _spectra_capabilities hip=spectra.hip hsa=spectra.hip)
  endif()
  if(_spectra_ONEDPL OR SPECTRA_PREBUILT_PACKAGE)
    list(APPEND _spectra_capabilities
         opencl1_1=spectra.dpl opencl2_0=spectra.dpl spirv64_kernel=spectra.dpl spirv_glcompute=spectra.dpl metal1_0=spectra.dpl)
  endif()

  set(_spectra_suite_text "${_spectra_suite_profile}")
  string(APPEND _spectra_suite_text
         "POLYTEST_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-work\n"
         "POLYTEST_ARCH_VAR=spectra_arch\n"
         "POLYTEST_STDPAR_VAR=spectra_stdpar\n"
         "POLYTEST_STDPAR={apple_target}-std=c++17 -fstdpar -fstdpar-mem=direct -fstdpar-rt={stdpar_rt} "
         "-fstdpar-arch={spectra_arch} -fstdpar-library-path=${SPECTRA_PACKAGE_ROOT} "
         "-fstdpar-library-capabilities={spectra_capability} -I${CMAKE_CURRENT_SOURCE_DIR}/generated/cpp/include "
         "-I${CMAKE_CURRENT_SOURCE_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/test/generated\n"
         "POLYTEST_TARGET_VAR=spectra_capability\n"
         "POLYTEST_OUTPUT_PREFIX=spectra_test_\n"
         "POLYTEST_TEMP_PREFIX=spectra_\n"
         "POLYTEST_DEFAULT_ENV+=:POLYTEST_REPRO_CHECK=0\n")
  foreach(_capability IN LISTS _spectra_capabilities)
    string(APPEND _spectra_suite_text "POLYTEST_TARGET_VALUE+=:${_capability}\n")
  endforeach()
  foreach(_source IN LISTS _spectra_generated_tests)
    string(APPEND _spectra_suite_text "POLYTEST_FILES+=:${_source}\n")
  endforeach()
  file(WRITE "${_spectra_suite}" "${_spectra_suite_text}")
  polyregion_add_polytest_suite(spectra SUITE "${_spectra_suite}" DEPENDS ${_spectra_generated_tests})
endif()
