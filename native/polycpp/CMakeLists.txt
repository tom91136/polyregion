add_subdirectory(polystl)
add_subdirectory(clang-plugin)

set(POLYCPP_HEADER_SET_DIR "lib/polycpp/include")
set(POLYCPP_LIB_DIR "lib/polycpp/lib")

# =========================== Generate header resource target ===========================
function(add_copy_header_set_targets target commands_var)
    get_target_property(header_set ${target} HEADER_SET)
    get_target_property(header_dirs ${target} HEADER_DIRS)
    list(LENGTH header_dirs header_dirs_len)
    if (header_dirs_len GREATER 1)
        message(FATAL_ERROR "More than one header dirs for dependent target ${target}, cannot create relative paths (dirs=${header_dirs})")
    endif ()
    foreach (absolute_file ${header_set})
        cmake_path(RELATIVE_PATH absolute_file BASE_DIRECTORY "${header_dirs}" OUTPUT_VARIABLE relative_file)
        cmake_path(GET relative_file EXTENSION LAST_ONLY header_extension)
        if (header_extension STREQUAL ".impl")
            cmake_path(GET relative_file STEM relative_file)
        endif ()
        list(APPEND ${commands_var} COMMAND ${CMAKE_COMMAND} -E copy "${absolute_file}" "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_HEADER_SET_DIR}/${relative_file}")
    endforeach ()
    set(${commands_var} ${${commands_var}} PARENT_SCOPE)
#    add_custom_target(${name} ${commands})
endfunction()

foreach (header_target polycommon polystl polyinvoke-obj polyrt-obj polyreflect-rt)
    add_copy_header_set_targets(${header_target} header_commands)
endforeach ()
add_custom_target(polycpp-header-resource ${header_commands})

# =======================================================================================


# =========================== Generate header resource target ===========================
function(add_copy_lib_targets target)
    foreach (target ${ARGN})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${target}>" "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_LIB_DIR}/")
    endforeach ()
    add_custom_target(${target}
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_LIB_DIR}/"
            ${commands})
endfunction()
if (POLYREGION_FUSED_DRIVER)
    add_copy_lib_targets(polycpp-lib-resource polystl-static polystl)
    # Fused mem=reflect uses polyld-link (polyreflect statically linked); install as the
    # platform's lld name (lld-link.exe on Windows, ld.lld on POSIX) next to polycpp so
    # `-fuse-ld=lld` resolves to it (clang's LTO check rejects anything not named lld).
    if (WIN32)
        set(POLYCPP_LLD_NAME "lld-link.exe")
    else ()
        set(POLYCPP_LLD_NAME "ld.lld")
    endif ()
    add_custom_target(polycpp-polyld-resource
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:polyld-link>" "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_LLD_NAME}")
else ()
    add_copy_lib_targets(polycpp-lib-resource polycpp-clang-plugin polyreflect-plugin polystl-static polystl)
endif ()
# =======================================================================================


if (POLYREGION_FUSED_DRIVER)
    set(POLYCPP_CLANG_DRIVER_DIR "${POLYREGION_LLVM_SRC_DIR}/clang/tools/driver")
    if (NOT EXISTS "${POLYCPP_CLANG_DRIVER_DIR}/driver.cpp")
        message(FATAL_ERROR "clang driver sources not found at ${POLYCPP_CLANG_DRIVER_DIR}; LLVM source tree required for POLYREGION_FUSED_DRIVER builds")
    endif ()
    add_executable(polycpp
            driver_main.cpp
            ${POLYCPP_CLANG_DRIVER_DIR}/driver.cpp
            ${POLYCPP_CLANG_DRIVER_DIR}/cc1_main.cpp
            ${POLYCPP_CLANG_DRIVER_DIR}/cc1as_main.cpp
            ${POLYCPP_CLANG_DRIVER_DIR}/cc1gen_reproducer_main.cpp
            clang-plugin/plugin.cpp
            clang-plugin/remapper.cpp
            clang-plugin/clang_utils.cpp
            clang-plugin/ast_visitors.cpp
            clang-plugin/codegen.cpp
            clang-plugin/rewriter.cpp
            # XXX Fused builds have no .dll for -fpass-plugin to load, so polyreflect's pass
            # plugin is statically linked and wired via CodeGenOpts.PassBuilderCallbacks from
            # the polycpp clang plugin. Without it InterposePass never runs and captured heap
            # allocations are invisible to polyreflect-rt on non-USM devices.
            ../polyreflect/plugin.cpp
            ../polyreflect/interpose.cpp
            ../polyreflect/reflect-mem.cpp
            ../polyreflect/reflect-stack.cpp)
    target_include_directories(polycpp PRIVATE
            clang-plugin
            ../polyreflect
            ${LLVM_INCLUDE_DIRS}
            ${CLANG_INCLUDE_DIRS})
    target_compile_definitions(polycpp PRIVATE
            ${LLVM_DEFINITIONS}
            POLYREGION_FUSED_DRIVER=1)
    # XXX Link the umbrella libLLVM + libclang-cpp under dylib, individual archives under
    # static; mixing the two double-registers MachineSchedRegistry entries (e.g. AMDGPU "si").
    # The standalone clangCodeGen/clangFrontendTool archives are absent in the dylib variant.
    if (POLYREGION_LLVM_DYLIB)
        set(POLYCPP_LLVM_LIBS LLVM)
        set(POLYCPP_CLANG_LIBS clang-cpp)
    else ()
        llvm_map_components_to_libnames(POLYCPP_LLVM_LIBS
                ${LLVM_TARGETS_TO_BUILD}
                Analysis CodeGen Core IPO AggressiveInstCombine InstCombine
                Instrumentation MC MCParser ObjCARCOpts Option ScalarOpts
                Support TargetParser TransformUtils Vectorize)
        set(POLYCPP_CLANG_LIBS clangBasic clangCodeGen clangDriver clangFrontend
                clangFrontendTool clangOptions clangSerialization)
    endif ()
    target_link_libraries(polycpp PRIVATE
            ${POLYCPP_CLANG_LIBS}
            ${POLYCPP_LLVM_LIBS}
            polycommon polyc-driver polyfront-options PolyAST
            Aspartame::Aspartame fmt::fmt)
    target_precompile_headers(polycpp PRIVATE
            <optional> <ostream> <unordered_map> <unordered_set> <utility> <vector>
            [["clang/AST/ASTConsumer.h"]]
            [["clang/AST/ASTContext.h"]]
            [["clang/AST/Decl.h"]]
            [["clang/AST/DeclCXX.h"]]
            [["clang/AST/Expr.h"]]
            [["clang/AST/ExprCXX.h"]]
            [["clang/AST/RecursiveASTVisitor.h"]]
            [["clang/ASTMatchers/ASTMatchers.h"]]
            [["clang/Frontend/CompilerInstance.h"]]
            [["clang/Frontend/CompilerInvocation.h"]]
            [["clang/Frontend/FrontendAction.h"]]
            [["clang/Lex/PreprocessorOptions.h"]]
            [["llvm/ADT/StringRef.h"]]
            [["llvm/Support/Casting.h"]]
            [["llvm/Support/FormatVariadic.h"]]
            [["llvm/Support/MemoryBuffer.h"]]
            [["fmt/format.h"]]
            [["aspartame/all.hpp"]]
            [["ast.h"]])
    add_dependencies(polycpp polycpp-lib-resource polycpp-header-resource polyld-link polycpp-polyld-resource)
else ()
    add_executable(polycpp driver_main.cpp)
    add_dependencies(polycpp polycpp-lib-resource polycpp-header-resource)
endif ()

add_executable(polycpp-tests test/test_all.cpp)
add_dependencies(polycpp-tests polycpp polystl)
target_include_directories(polycpp-tests PRIVATE
        ../include
        polystl
        ${LLVM_INCLUDE_DIRS}
        ${fire_SOURCE_DIR}
        "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(polycpp-tests PRIVATE
        POLYREGION_TEST_PROFILE_DIR="${CMAKE_SOURCE_DIR}/test-profiles")

# === test_all.h.in variables ===
get_filename_component(POLYCPP_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
set(POLYCPP_TEST_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-cache")
file(MAKE_DIRECTORY "${POLYCPP_TEST_CACHE_DIR}")
set(POLYCPP_TEST_INPUT_FILES "")
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/check_*.cpp")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYCPP_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/**/check_*.cpp")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYCPP_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()
set(POLYCPP_CLANG_DRIVER "${LLVM_DIST_DIR}/bin/clang++")
set(POLYCPP_POLYSTL_LIB "$<TARGET_FILE:polystl>")
set(POLYCPP_POLYSTL_LIBRARY_PATH "$<TARGET_FILE_DIR:polystl>")
set(POLYCPP_POLYSTL_INCLUDE "${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polycpp>/$<TARGET_NAME:polystl>/include:${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polyinvoke>/include:${CMAKE_SOURCE_DIR}/include:${CMAKE_SOURCE_DIR}/include")
# ===  end test_all.h.in variables ===
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test/test_all.h.in" "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in")

file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in"
)

target_include_directories(polycpp PRIVATE ../compiler ../include ${LLVM_INCLUDE_DIRS})
target_link_libraries(polycpp PRIVATE
        polyc-driver
        polyfront-options
        Aspartame::Aspartame
        fmt::fmt
)
# XXX /Zi emits a separate PDB so release polycpp crash dumps stay symbolicated.
if (MSVC)
    target_compile_options(polycpp PRIVATE /Zi)
    target_link_options(polycpp PRIVATE /DEBUG)
endif ()
target_link_libraries(polycpp-tests PRIVATE
        polytest
        Catch2::Catch2WithMain
)


target_compile_options(polycpp PRIVATE ${COMPILE_OPTIONS})
target_link_options(polycpp PRIVATE ${LINK_OPTIONS})

target_compile_options(polycpp-tests PRIVATE ${COMPILE_OPTIONS})
target_link_options(polycpp-tests PRIVATE ${LINK_OPTIONS})

# XXX one ctest entry per test file x {offload, passthrough}.
# Cross-process device serialisation lives in DeviceLock (polyinvoke/include/polyinvoke/device_lock.h).
foreach (_test_file IN LISTS TEST_FILE_PATHS)
    get_filename_component(_test_name "${_test_file}" NAME_WE)
    string(REGEX REPLACE "^check_" "" _test_short "${_test_name}")
    foreach (_mode IN ITEMS offload passthrough)
        set(_name "polycpp-${_test_short}-${_mode}")
        add_test(NAME ${_name}
                COMMAND polycpp-tests --${_mode}-only -c "${_test_short}"
                WORKING_DIRECTORY "${POLYCPP_TEST_CACHE_DIR}")
        set(_labels "")
        if (_mode STREQUAL "offload")
            set_tests_properties(${_name} PROPERTIES LABELS "device")
            set(_labels "device")
        endif ()
        polyregion_test_dist_record(NAME ${_name}
                COMMAND :bin:polycpp-tests --${_mode}-only -c "${_test_short}"
                ENVIRONMENT "POLYTEST_FILES_DIR=\${CMAKE_CURRENT_LIST_DIR}/test/polycpp;POLYTEST_PROFILE_DIR=\${CMAKE_CURRENT_LIST_DIR}/test/profiles;POLYTEST_BINARY_DIR=\${CMAKE_CURRENT_LIST_DIR}/test-cache"
                LABELS "${_labels}")
    endforeach ()
endforeach ()

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${POLYCPP_TEST_CACHE_DIR}")
add_custom_target(clean-polycpp-tests
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${POLYCPP_TEST_CACHE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${POLYCPP_TEST_CACHE_DIR}"
        COMMENT "Removing polycpp test-cache binaries from ${POLYCPP_TEST_CACHE_DIR}")

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    #    add_custom_command(
    #            TARGET PolySTLPlugin POST_BUILD
    #            COMMAND strip
    #            ARGS -SXx $<TARGET_FILE:PolySTLPlugin>)
    add_custom_command(
            TARGET polycpp POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polycpp>)
endif ()

install(TARGETS polycpp RUNTIME DESTINATION bin)
install(TARGETS polystl polystl-static
        LIBRARY DESTINATION lib/polycpp/lib
        ARCHIVE DESTINATION lib/polycpp/lib
        RUNTIME DESTINATION lib/polycpp/lib
        FILE_SET HEADERS DESTINATION lib/polycpp/include)
if (NOT POLYREGION_FUSED_DRIVER)
    install(TARGETS polycpp-clang-plugin polyreflect-plugin
            LIBRARY DESTINATION lib/polycpp/lib
            RUNTIME DESTINATION lib/polycpp/lib)
endif ()
foreach (HEADER_TARGET polycommon polyinvoke-obj polyrt-obj polyreflect-rt)
    install(TARGETS ${HEADER_TARGET} FILE_SET HEADERS DESTINATION lib/polycpp/include)
endforeach ()
