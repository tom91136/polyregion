if (POLYC_JS_ENGINE STREQUAL "hermes")
    set(POLYC_JS_RUNNER_SRC hermes_runner.cpp)
else ()
    set(POLYC_JS_RUNNER_SRC qjs_runner.cpp)
endif ()

set(LIB_SOURCES
        backend/backend.cpp
        backend/llvm_amdgpu.cpp
        backend/llvm_nvptx.cpp
        backend/llvm_spirv_cl.cpp
        backend/llvm_vulkan.cpp
        backend/llvm_cpu.cpp
        backend/llvm_target.cpp
        backend/llvm.cpp
        backend/llvmc.cpp
        backend/c_source.cpp
        backend/lld_lite.cpp
        backend/memoryfs.cpp
        ${POLYC_JS_RUNNER_SRC}
        dso_runner.cpp
        polypass_locate.cpp
        js_cache.cpp
        compiler.cpp
)

add_library(polyc ${LIB_SOURCES})
add_library(polyc-driver driver_polyc.cpp)

add_executable(polyc-tests test/ast_codegen.cpp test/backend.cpp test/ast.cpp test/js_runner.cpp test/vendor_bitcode.cpp)
add_executable(polyc-benchmarks benchmark/babelstream.cpp)
add_executable(polyc-bin driver_main.cpp)

target_include_directories(polyc-tests PRIVATE . ../include ../runtime)
target_include_directories(polyc-benchmarks PRIVATE . ../include ../runtime)
target_include_directories(polyc-bin PRIVATE . ../include ${fire_SOURCE_DIR})

target_include_directories(polyc-driver PUBLIC . ../include ${fire_SOURCE_DIR})
target_include_directories(polyc PUBLIC
        ${LLVM_INCLUDE_DIRS}
        ${LLD_INCLUDE_DIRS}
        . ../include
        include)

if (NOT WIN32)
    target_precompile_headers(polyc-tests PRIVATE
            compiler.h

            backend/backend.h
            backend/c_source.h
            backend/llvm.h
            backend/llvmc.h
            backend/memoryfs.h

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


if (POLYC_JS_ENGINE STREQUAL "hermes")
    # hermesvm_a = all-in-one VM archive;
    # compileJS = source-to-HBC entry point
    # jsi = public C++ API
    # winmm = sampling profiler deps for windows
    set(POLYC_JS_LIBS compileJS hermesvm_a jsi)
    if (MSVC)
        list(APPEND POLYC_JS_LIBS winmm)
    endif ()
else ()
    set(POLYC_JS_LIBS qjs)
endif ()
list(APPEND POLYC_JS_LIBS ${CMAKE_DL_LIBS})

target_link_libraries(polyc
        PUBLIC
        PolyAST
        PRIVATE
        Aspartame::Aspartame
        magic_enum::magic_enum
        fmt::fmt
        ${POLYC_JS_LIBS}
        ${POLYREGION_SYSTEM_LIBS}
        ${LLVM_LIBS}
        ${LLD_LIBS}
)
target_link_libraries(polyc-tests PRIVATE polyc Catch2::Catch2WithMain Aspartame::Aspartame)
add_test(NAME polyc-tests COMMAND polyc-tests)
polyregion_test_dist_record(NAME polyc-tests COMMAND :bin:polyc-tests)
target_link_libraries(polyc-benchmarks PRIVATE polyc polyinvoke-static polytest Aspartame::Aspartame)
polyregion_test_profile_dir(polyc-benchmarks)
target_link_libraries(polyc-driver PRIVATE polyc Aspartame::Aspartame)
target_link_libraries(polyc-bin PRIVATE polyc polyc-driver)

polyregion_apply_common_options(polyc-tests EXTRA_LINK ${POLYC_LINK_HERMES_OPTS})
polyregion_apply_common_options(polyc-benchmarks EXTRA_LINK ${POLYC_LINK_HERMES_OPTS})
polyregion_apply_common_options(polyc-driver)
polyregion_apply_common_options(polyc-bin EXTRA_LINK ${POLYC_LINK_HERMES_OPTS})
polyregion_apply_common_options(polyc EXTRA_LINK ${POLYC_LINK_HERMES_OPTS})

set_target_properties(polyc-bin PROPERTIES OUTPUT_NAME polyc)
if (WIN32)
    set_target_properties(polyc-bin PROPERTIES ARCHIVE_OUTPUT_NAME polyc-bin)
endif ()

# dev-path: lets polyc resolve the bundle/DSO out of the source tree without
# an install step. Dist install ships them under lib/ alongside the binaries.
# Built by sbt (just pass-native | just pass-js) into native/polyc/.
if (WIN32)
    set(_polypass_dso_name "libpolypass.dll")
elseif (APPLE)
    set(_polypass_dso_name "libpolypass.dylib")
else ()
    set(_polypass_dso_name "libpolypass.so")
endif ()

if (POLYC_PASS_BUNDLE STREQUAL "dso")
    set(_polypass_artefact "${CMAKE_CURRENT_SOURCE_DIR}/${_polypass_dso_name}")
    set(_polypass_dev_macro POLYPASS_DSO_DEV_PATH)
    set(_polypass_build_hint "just pass-native")
else ()
    set(_polypass_artefact "${CMAKE_CURRENT_SOURCE_DIR}/polypass.js")
    set(_polypass_dev_macro POLYPASS_JS_DEV_PATH)
    set(_polypass_build_hint "just pass-js")
endif ()
if (NOT EXISTS "${_polypass_artefact}")
    message(FATAL_ERROR
            "POLYC_PASS_BUNDLE=${POLYC_PASS_BUNDLE} but ${_polypass_artefact} is missing.\n"
            "Run: ${_polypass_build_hint}")
endif ()
target_compile_definitions(polyc PRIVATE ${_polypass_dev_macro}="${_polypass_artefact}")

if (NOT EXISTS "${POLYREGION_DEVICE_LIBS_DIR}/.staged")
  message(FATAL_ERROR
          "Device libs are not staged at ${POLYREGION_DEVICE_LIBS_DIR}.\n"
          "Run: cmake -DACTION=DEVICE_LIBS -P build.cmake")
endif ()

target_compile_definitions(polyc PRIVATE
  POLYREGION_BITCODE_DEV_DIR="${POLYREGION_DEVICE_LIBS_DIR}")
target_compile_definitions(polyc-tests PRIVATE
  POLYREGION_BITCODE_DEV_DIR="${POLYREGION_DEVICE_LIBS_DIR}")

set(POLYREGION_LIBDEVICE_ENTRIES "")
list(LENGTH POLYREGION_CUDA_LIBDEVICE _n_libdev)
math(EXPR _last_libdev "${_n_libdev} - 1")
foreach(_i RANGE ${_last_libdev} 0 -1)
  list(GET POLYREGION_CUDA_LIBDEVICE ${_i} _entry)
  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.[0-9]+:([0-9]+)$" _ "${_entry}")
  string(APPEND POLYREGION_LIBDEVICE_ENTRIES
    "    {\"libdevice.10.cuda${CMAKE_MATCH_1}-${CMAKE_MATCH_2}.bc\", ${CMAKE_MATCH_3}u},\n")
endforeach()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/backend/vendor_libdevice.inc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/vendor_libdevice.inc"
  @ONLY)
target_include_directories(polyc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

install(TARGETS polyc-bin RUNTIME DESTINATION bin)
install(FILES "${_polypass_artefact}" DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/polyregion"
        DESTINATION include
        FILES_MATCHING PATTERN "polypass.h")
install(DIRECTORY "${POLYREGION_DEVICE_LIBS_DIR}/" DESTINATION lib
        FILES_MATCHING PATTERN "*.bc" PATTERN "*.LICENSE")
