set(LIB_SOURCES
        backend/backend.cpp
        backend/llvm_amdgpu.cpp
        backend/llvm_nvptx.cpp
        backend/llvm_spirv_cl.cpp
        backend/llvm_cpu.cpp
        backend/llvm_target.cpp
        backend/llvm.cpp
        backend/llvmc.cpp
        backend/c_source.cpp
        backend/lld_lite.cpp
        backend/memoryfs.cpp
        qjs_runner.cpp
        compiler.cpp
)

add_library(polyc ${LIB_SOURCES})
add_library(polyc-driver driver_polyc.cpp)

add_executable(polyc-tests test/ast_codegen.cpp test/ast_offload.cpp test/backend.cpp test/ast.cpp test/qjs_runner.cpp test/vendor_bitcode.cpp)
add_executable(polyc-benchmarks benchmark/babelstream.cpp)
add_executable(polyc-bin driver_main.cpp)

target_include_directories(polyc-tests PRIVATE . ../include ../runtime)
target_include_directories(polyc-benchmarks PRIVATE . ../include ../runtime)
target_include_directories(polyc-bin PRIVATE . ../include ${fire_SOURCE_DIR})

target_include_directories(polyc-driver PUBLIC . ../include ${fire_SOURCE_DIR})
target_include_directories(polyc PUBLIC
        ${LLVM_INCLUDE_DIRS}
        ${LLD_INCLUDE_DIRS}
        . ../include)

if (NOT WIN32)
    target_precompile_headers(polyc-tests PRIVATE
            compiler.h

            backend/backend.h
            backend/c_source.h
            backend/llvm.h
            backend/llvmc.h
            backend/memoryfs.h
            backend/suppress_fd.h

            <optional>
            <ostream>
            <unordered_map>
            <utility>
            <vector>
    )
endif ()

if (POLYREGION_LLVM_DYLIB)
    set(LLVM_LIBS LLVM)
else ()
    llvm_map_components_to_libnames(LLVM_LIBS
            ${LLVM_TARGETS_TO_BUILD}
            ${LLVM_EXPERIMENTAL_TARGETS_TO_BUILD}
            analysis
            asmparser
            asmprinter
            codegen
            core
            object
            irreader
            passes
            ipo
            instcombine
            mc
            remarks
            scalaropts
            selectiondag
            support
            target
            transformutils
            vectorize
            targetparser
            objcarcopts)
endif ()

set(LLD_LIBS lldCommon lldELF)
message(STATUS "[compiler] Adding LLVM libs = ${LLVM_LIBS} ")
message(STATUS "[compiler] Adding LLD libs = ${LLD_LIBS} ")


#set(LLVM_TARGETS_SOURCE_HEADERS ${LLVM_TARGETS_TO_BUILD})
#set(LLVM_TARGETS_BINARY_HEADERS ${LLVM_TARGETS_TO_BUILD})
#list(TRANSFORM LLVM_TARGETS_SOURCE_HEADERS PREPEND ${llvm_SOURCE_DIR}/lib/Target/)
#list(TRANSFORM LLVM_TARGETS_BINARY_HEADERS PREPEND ${llvm_BINARY_DIR}/lib/Target/)
#
#message(STATUS "[compiler] Target source headers = ${LLVM_TARGETS_SOURCE_HEADERS} ")
#message(STATUS "[compiler] Target binary headers = ${LLVM_TARGETS_BINARY_HEADERS} ")
#target_include_directories(polyc PRIVATE ${LLVM_TARGETS_SOURCE_HEADERS} ${LLVM_TARGETS_BINARY_HEADERS})


target_link_libraries(polyc
        PUBLIC
        PolyAST
        PRIVATE
        Aspartame::Aspartame
        magic_enum::magic_enum
        fmt::fmt
        qjs
        ${POLYREGION_SYSTEM_LIBS}
        ${LLVM_LIBS}
        ${LLD_LIBS}
)
target_link_libraries(polyc-tests PRIVATE polyc polyinvoke-static Catch2::Catch2WithMain Aspartame::Aspartame)
add_test(NAME polyc-tests COMMAND polyc-tests)
polyregion_test_dist_record(NAME polyc-tests COMMAND :bin:polyc-tests)
target_link_libraries(polyc-benchmarks PRIVATE polyc polyinvoke-static polytest Aspartame::Aspartame)
target_compile_definitions(polyc-benchmarks PRIVATE
        POLYREGION_TEST_PROFILE_DIR="${CMAKE_SOURCE_DIR}/test-profiles")
target_link_libraries(polyc-driver PRIVATE polyc ${LLVM_LIBS} ${LLD_LIBS} Aspartame::Aspartame)
target_link_libraries(polyc-bin PRIVATE polyc polyc-driver)

target_compile_options(polyc-tests PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc-benchmarks PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc-driver PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc-bin PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyc-tests PRIVATE ${LINK_OPTIONS})
target_link_options(polyc-benchmarks PRIVATE ${LINK_OPTIONS})
target_link_options(polyc-driver PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyc-bin PRIVATE ${LINK_OPTIONS})
target_link_options(polyc PRIVATE ${LINK_OPTIONS})

set_target_properties(polyc-bin PROPERTIES OUTPUT_NAME polyc)

# XXX dev-path lets polyc resolve polypass.js out of the source tree when run from a build dir
# without an install step; dist install ships it under lib/ next to the binaries.
target_compile_definitions(polyc PRIVATE POLYPASS_JS_DEV_PATH="${CMAKE_CURRENT_SOURCE_DIR}/polypass.js")

set(POLYREGION_CUDA_LIBDEVICE_VERSIONS "11.8.0;12.9.2;13.2.0" CACHE STRING
    "CUDA versions to stage libdevice.10.cuda<M>-<m>.bc from. polyc picks the newest applicable to the target sm_*.")
set(POLYREGION_CUDA_LIBDEVICE_MIN_ARCH  "35;50;75" CACHE STRING
    "sm_NN floor per libdevice version (parallel to POLYREGION_CUDA_LIBDEVICE_VERSIONS). NVIDIA's drop schedule is irregular so this stays an explicit policy list.")
set(POLYREGION_ROCM_TAG     "rocm-7.2.3" CACHE STRING "ROCm/llvm-project tag for device-libs prep")
set(POLYREGION_BITCODE_DIR  "${CMAKE_CURRENT_BINARY_DIR}/vendor-bitcode")
option(POLYREGION_PREPARE_VENDOR_BITCODE "Run vendor bitcode prep at configure time" ON)

if(POLYREGION_PREPARE_VENDOR_BITCODE)
  foreach(_ver IN LISTS POLYREGION_CUDA_LIBDEVICE_VERSIONS)
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _ "${_ver}")
    set(_marker "${POLYREGION_BITCODE_DIR}/libdevice.10.cuda${CMAKE_MATCH_1}-${CMAKE_MATCH_2}.bc")
    if(NOT EXISTS "${_marker}")
      execute_process(
        COMMAND "${CMAKE_COMMAND}"
          "-DOUT_DIR=${POLYREGION_BITCODE_DIR}"
          "-DCUDA_VERSION=${_ver}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/prepare_cuda_dist.cmake"
        RESULT_VARIABLE _rc)
      if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "libdevice prep for CUDA ${_ver} failed (${_rc})")
      endif()
    endif()
  endforeach()
  if(NOT EXISTS "${POLYREGION_BITCODE_DIR}/ocml.bc" OR NOT EXISTS "${POLYREGION_BITCODE_DIR}/oclc_abi_version_500.bc")
    execute_process(
      COMMAND "${CMAKE_COMMAND}"
        "-DOUT_DIR=${POLYREGION_BITCODE_DIR}"
        "-DROCM_TAG=${POLYREGION_ROCM_TAG}"
        "-DPOLYREGION_LLVM_BIN_HINT=${LLVM_DIST_DIR}/bin"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/prepare_amdgpu_dist.cmake"
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "device-libs prep failed (${_rc})")
    endif()
  endif()
endif()

target_compile_definitions(polyc PRIVATE
  POLYREGION_BITCODE_DEV_DIR="${POLYREGION_BITCODE_DIR}")
target_compile_definitions(polyc-tests PRIVATE
  POLYREGION_BITCODE_DEV_DIR="${POLYREGION_BITCODE_DIR}")

list(LENGTH POLYREGION_CUDA_LIBDEVICE_VERSIONS _n_libdev)
list(LENGTH POLYREGION_CUDA_LIBDEVICE_MIN_ARCH _n_min_arch)
if(NOT _n_libdev EQUAL _n_min_arch)
  message(FATAL_ERROR "POLYREGION_CUDA_LIBDEVICE_VERSIONS (${_n_libdev}) and POLYREGION_CUDA_LIBDEVICE_MIN_ARCH (${_n_min_arch}) must be parallel lists")
endif()
set(POLYREGION_LIBDEVICE_ENTRIES "")
math(EXPR _last_libdev "${_n_libdev} - 1")
foreach(_i RANGE ${_last_libdev} 0 -1)
  list(GET POLYREGION_CUDA_LIBDEVICE_VERSIONS ${_i} _ver)
  list(GET POLYREGION_CUDA_LIBDEVICE_MIN_ARCH ${_i} _arch)
  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _ "${_ver}")
  string(APPEND POLYREGION_LIBDEVICE_ENTRIES
    "    {\"libdevice.10.cuda${CMAKE_MATCH_1}-${CMAKE_MATCH_2}.bc\", ${_arch}u},\n")
endforeach()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/vendor_libdevice.inc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/vendor_libdevice.inc"
  @ONLY)
target_include_directories(polyc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

install(TARGETS polyc-bin RUNTIME DESTINATION bin)
install(FILES polypass.js DESTINATION lib OPTIONAL)
install(DIRECTORY "${POLYREGION_BITCODE_DIR}/" DESTINATION lib
        FILES_MATCHING PATTERN "*.bc" PATTERN "*.LICENSE")
