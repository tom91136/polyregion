add_subdirectory(polydco)
add_subdirectory(flang-plugin)

set(POLYFC_HEADER_SET_DIR "lib/polyfc/include")
set(POLYFC_LIB_DIR "lib/polyfc/lib")

function(add_copy_lib_targets target)
    foreach (target ${ARGN})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${target}>" "$<TARGET_FILE_DIR:polyfc>/${POLYFC_LIB_DIR}/")
    endforeach ()
    add_custom_target(${target}
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:polyfc>/${POLYFC_LIB_DIR}/"
            ${commands})
endfunction()
if (POLYREGION_FUSED_DRIVER)
    add_copy_lib_targets(polyfc-lib-resource polydco-static polydco)
else ()
    add_copy_lib_targets(polyfc-lib-resource polyfc-flang-plugin polyreflect-plugin polydco-static polydco)
endif ()


if (POLYREGION_FUSED_DRIVER)
    set(POLYFC_FLANG_DRIVER_DIR "${POLYREGION_LLVM_SRC_DIR}/flang/tools/flang-driver")
    if (NOT EXISTS "${POLYFC_FLANG_DRIVER_DIR}/fc1_main.cpp")
        message(FATAL_ERROR "flang driver sources not found at ${POLYFC_FLANG_DRIVER_DIR}; LLVM source tree required for POLYREGION_FUSED_DRIVER builds")
    endif ()
    add_executable(polyfc
            driver_main.cpp
            flang_driver_wrapper.cpp
            ${POLYFC_FLANG_DRIVER_DIR}/fc1_main.cpp
            flang-plugin/plugin.cpp
            flang-plugin/rewriter.cpp
            flang-plugin/remapper.cpp
            flang-plugin/codegen.cpp
            flang-plugin/mlir_utils.cpp
            flang-plugin/parallel.cpp
            flang-plugin/mirrors.cpp
            flang-plugin/fexpr.cpp)
    # XXX Mixed C++ standard: polyfc plugin sources (remapper, rewriter, parallel, ...) use
    # C++20 designated initializers and concepts. Files that pull in flang/Evaluate headers
    # must stay on C++17 because the C++20 stdlib eagerly instantiates
    # is_trivially_destructible on Fortran::evaluate::Expr before it is complete.
    set_target_properties(polyfc PROPERTIES CXX_STANDARD 20)
    if (MSVC)
        # XXX match flangFrontend.lib EH/RTTI ABI or linking against flang_rt yields LNK2038.
        set_source_files_properties(
                flang_driver_wrapper.cpp
                flang-plugin/plugin.cpp
                "${POLYFC_FLANG_DRIVER_DIR}/fc1_main.cpp"
                PROPERTIES SKIP_PRECOMPILE_HEADERS ON
                COMPILE_FLAGS "/EHs-c- /GR- /std:c++17 /permissive-"
                COMPILE_DEFINITIONS "_HAS_EXCEPTIONS=0")
        target_compile_options(polyfc PRIVATE /EHs-c- /GR-)
        target_compile_definitions(polyfc PRIVATE _HAS_EXCEPTIONS=0)
    else ()
        set_source_files_properties(
                flang_driver_wrapper.cpp
                flang-plugin/plugin.cpp
                "${POLYFC_FLANG_DRIVER_DIR}/fc1_main.cpp"
                PROPERTIES SKIP_PRECOMPILE_HEADERS ON
                COMPILE_FLAGS "-std=c++17")
    endif ()
    if (WIN32)
        # XXX flang's GetProgramPath looks for `flang` to spawn integrated tools; ship the
        # user-facing polyfc.exe as a copy.
        set_target_properties(polyfc PROPERTIES OUTPUT_NAME "flang")
        add_custom_command(TARGET polyfc POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:polyfc>" "$<TARGET_FILE_DIR:polyfc>/polyfc.exe")
    endif ()
    # XXX flang's getIntrinsicDir() resolves <exec_dir>/../include/flang; mirror that layout
    # in the build tree at configure time so tests find the intrinsic modules.
    if (EXISTS "${LLVM_DIST_DIR}/include/flang")
        file(COPY "${LLVM_DIST_DIR}/include/flang"
                DESTINATION "${CMAKE_BINARY_DIR}/include")
    endif ()
    target_include_directories(polyfc PRIVATE
            flang-plugin
            ${LLVM_INCLUDE_DIRS}
            ${CLANG_INCLUDE_DIRS}
            ${FLANG_INCLUDE_DIRS}
            ${MLIR_INCLUDE_DIRS}
            polydco)
    target_compile_definitions(polyfc PRIVATE
            ${LLVM_DEFINITIONS}
            FLANG_LITTLE_ENDIAN
            POLYREGION_FUSED_DRIVER=1
            "POLYFC_FUSED_LLVM_BUILD_DIR=\"${LLVM_BUILD_DIR}\""
            "POLYFC_FUSED_DIST_DIR=\"${LLVM_DIST_DIR}\"")
    # XXX Link the umbrella libLLVM + libclang-cpp under dylib, individual archives under
    # static; mixing the two double-registers MachineSchedRegistry entries at init.
    if (POLYREGION_LLVM_DYLIB)
        set(POLYFC_LLVM_LIBS LLVM)
        set(POLYFC_CLANG_LIBS clang-cpp)
    else ()
        llvm_map_components_to_libnames(POLYFC_LLVM_LIBS
                ${LLVM_TARGETS_TO_BUILD}
                Analysis CodeGen Core IPO AggressiveInstCombine InstCombine
                Instrumentation MC MCParser ObjCARCOpts Option ScalarOpts
                Support TargetParser TransformUtils Vectorize)
        set(POLYFC_CLANG_LIBS clangDriver clangOptions clangBasic)
    endif ()
    target_link_libraries(polyfc PRIVATE
            flangFrontend flangFrontendTool
            ${POLYFC_CLANG_LIBS}
            ${POLYFC_LLVM_LIBS}
            polycommon polyc-driver polyfront-options PolyAST
            Aspartame::Aspartame fmt::fmt)
    target_precompile_headers(polyfc PRIVATE
            <optional> <ostream> <unordered_map> <unordered_set> <utility> <vector> <cstdio>
            [["flang/Optimizer/Dialect/FIROps.h"]]
            [["flang/Optimizer/Dialect/FIRType.h"]]
            [["mlir/Dialect/LLVMIR/LLVMDialect.h"]]
            [["mlir/Dialect/LLVMIR/LLVMTypes.h"]]
            [["mlir/IR/BuiltinAttributes.h"]]
            [["mlir/IR/BuiltinTypes.h"]]
            [["mlir/IR/Dialect.h"]]
            [["mlir/IR/Value.h"]]
            [["clang/Basic/Diagnostic.h"]]
            [["llvm/ADT/DenseSet.h"]]
            [["llvm/ADT/DenseMap.h"]]
            [["llvm/IR/Constant.h"]]
            [["llvm/Support/Casting.h"]]
            [["aspartame/all.hpp"]]
            [["ast.h"]])
    add_dependencies(polyfc polyfc-lib-resource)
else ()
    add_executable(polyfc driver_main.cpp)
    add_dependencies(polyfc polyfc-lib-resource)
endif ()

add_executable(polyfc-tests test/test_all.cpp)
add_dependencies(polyfc-tests polyfc polydco)
target_include_directories(polyfc-tests PRIVATE
        ../include
        polydco
        ${LLVM_INCLUDE_DIRS}
        ${fire_SOURCE_DIR}
        "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(polyfc-tests PRIVATE
        POLYREGION_TEST_PROFILE_DIR="${CMAKE_SOURCE_DIR}/test-profiles")

# === test_all.h.in variables ===
get_filename_component(POLYFC_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
# Isolated cache for generated test binaries so `cmake clean` removes them and stale builds
# cannot shadow a fresh run.
set(POLYFC_TEST_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-cache")
file(MAKE_DIRECTORY "${POLYFC_TEST_CACHE_DIR}")
set(POLYFC_TEST_INPUT_FILES "")
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/check_*.*")
list(LENGTH TEST_FILE_PATHS _n_polyfc_tests)
message(STATUS "polyfc tests: ${_n_polyfc_tests} files registered (x 2 modes)")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYFC_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()
set(POLYFC_FLANG_DRIVER "${LLVM_DIST_DIR}/bin/flang-new")
set(POLYFC_POLYDCO_LIB "$<TARGET_FILE:polydco>")
set(POLYFC_POLYDCO_LIBRARY_PATH "$<TARGET_FILE_DIR:polydco>")
set(POLYFC_POLYDCO_INCLUDE "${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polyfc>/$<TARGET_NAME:polydco>/include:${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polyinvoke>/include:${CMAKE_SOURCE_DIR}/include:${CMAKE_SOURCE_DIR}/include")
# ===  end test_all.h.in variables ===
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test/test_all.h.in" "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in")

file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in"
)

target_include_directories(polyfc PRIVATE ../compiler ../include ${LLVM_INCLUDE_DIRS})
target_link_libraries(polyfc PRIVATE
        polycommon
        polyc-driver
        polyfront-options
        Aspartame::Aspartame
        fmt::fmt
)
target_link_libraries(polyfc-tests PRIVATE polytest)


target_compile_options(polyfc PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyfc PRIVATE ${LINK_OPTIONS})

target_compile_options(polyfc-tests PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyfc-tests PRIVATE ${LINK_OPTIONS})

# XXX one ctest entry per test file x {offload, passthrough}.
foreach (_test_file IN LISTS TEST_FILE_PATHS)
    get_filename_component(_test_name "${_test_file}" NAME_WE)
    string(REGEX REPLACE "^check_" "" _test_short "${_test_name}")
    foreach (_mode IN ITEMS offload passthrough)
        set(_name "polyfc-${_test_short}-${_mode}")
        add_test(NAME ${_name}
                COMMAND polyfc-tests --${_mode}-only -c "${_test_short}"
                WORKING_DIRECTORY "${POLYFC_TEST_CACHE_DIR}")
        set(_labels "")
        if (_mode STREQUAL "offload")
            set_tests_properties(${_name} PROPERTIES LABELS "device")
            set(_labels "device")
        endif ()
        polyregion_test_dist_record(NAME ${_name}
                COMMAND :bin:polyfc-tests --${_mode}-only -c "${_test_short}"
                ENVIRONMENT "POLYTEST_FILES_DIR=\${CMAKE_CURRENT_LIST_DIR}/test/polyfc;POLYTEST_BINARY_DIR=\${CMAKE_CURRENT_LIST_DIR}/test-cache;POLYTEST_WORK_DIR=\${CMAKE_CURRENT_LIST_DIR}/test-cache"
                LABELS "${_labels}")
        # XXX LLD on Windows opens archives without FILE_SHARE_READ; serialise concurrent link
        # jobs against polydco-static.lib (~70 MB) to avoid LNK1104.
        if (WIN32)
            set_tests_properties(${_name} PROPERTIES RESOURCE_LOCK polyfc_polydco_static_lib)
        endif ()
    endforeach ()
endforeach ()
# Wipe the whole cache on `clean` so stale test binaries cannot survive a rebuild.
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${POLYFC_TEST_CACHE_DIR}")
# Targeted clean for the test-cache dir without disturbing the rest of the build.
add_custom_target(clean-polyfc-tests
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${POLYFC_TEST_CACHE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${POLYFC_TEST_CACHE_DIR}"
        COMMENT "Removing polyfc test-cache binaries from ${POLYFC_TEST_CACHE_DIR}")

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    #    add_custom_command(
    #            TARGET PolySTLPlugin POST_BUILD
    #            COMMAND strip
    #            ARGS -SXx $<TARGET_FILE:PolySTLPlugin>)
    add_custom_command(
            TARGET polyfc POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polyfc>)
endif ()

install(TARGETS polyfc RUNTIME DESTINATION bin)
# Binary is built as flang.exe on Windows; install a polyfc.exe copy as the user-facing name.
if (WIN32)
    install(FILES "$<TARGET_FILE_DIR:polyfc>/polyfc.exe" DESTINATION bin)
endif ()
install(TARGETS polydco polydco-static
        LIBRARY DESTINATION lib/polyfc/lib
        ARCHIVE DESTINATION lib/polyfc/lib
        RUNTIME DESTINATION lib/polyfc/lib
        FILE_SET HEADERS DESTINATION lib/polyfc/include)
if (NOT POLYREGION_FUSED_DRIVER)
    install(TARGETS polyfc-flang-plugin polyreflect-plugin
            LIBRARY DESTINATION lib/polyfc/lib
            RUNTIME DESTINATION lib/polyfc/lib)
endif ()
