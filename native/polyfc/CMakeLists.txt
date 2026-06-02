add_subdirectory(polydco)
add_subdirectory(flang-plugin)

set(POLYFC_HEADER_SET_DIR "lib/polyfc/include")
set(POLYFC_LIB_DIR "lib/polyfc/lib")

if (POLYREGION_FUSED_DRIVER)
    polyregion_copy_lib_targets(polyfc-lib-resource polyfc ${POLYFC_LIB_DIR} polyreflect-rt polydco-static polydco)
    if (WIN32)
        set(POLYFC_LLD_NAME "lld-link.exe")
    else ()
        set(POLYFC_LLD_NAME "ld.lld")
    endif ()
    add_custom_target(polyfc-polyld-resource
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:polyld-link>" "$<TARGET_FILE_DIR:polyfc>/${POLYFC_LLD_NAME}")
else ()
    polyregion_copy_lib_targets(polyfc-lib-resource polyfc ${POLYFC_LIB_DIR} polyfc-flang-plugin polyreflect-plugin polyreflect-rt polydco-static polydco)
endif ()


if (POLYREGION_FUSED_DRIVER)
    # Prefer the unpacked LLVM tree; fall back to the dist's staged copy (cache-hit CI builds
    # restore only the dist, the source tree is gone). build_llvm.cmake mirrors the files.
    set(POLYFC_FLANG_DRIVER_DIR "")
    foreach (_root "${POLYREGION_LLVM_SRC_DIR}/flang/tools/flang-driver"
                   "${LLVM_DIST_DIR}/share/polyregion-fused-driver/flang-tools-flang-driver")
        if (EXISTS "${_root}/fc1_main.cpp")
            set(POLYFC_FLANG_DRIVER_DIR "${_root}")
            break ()
        endif ()
    endforeach ()
    if (NOT POLYFC_FLANG_DRIVER_DIR)
        message(FATAL_ERROR "flang driver sources not found in src tree or dist staging; LLVM source tree or staged dist required for POLYREGION_FUSED_DRIVER builds")
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
            <optional> <unordered_map> <unordered_set> <utility> <vector> <cstdio> <cstdlib>
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
    add_dependencies(polyfc polyfc-lib-resource polyld-link polyfc-polyld-resource)
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
polyregion_test_profile_dir(polyfc-tests)

target_include_directories(polyfc PRIVATE ../compiler ../include ${LLVM_INCLUDE_DIRS})
target_link_libraries(polyfc PRIVATE
        polycommon
        polyc-driver
        polyfront-options
        Aspartame::Aspartame
        fmt::fmt
)
target_link_libraries(polyfc-tests PRIVATE polytest)


polyregion_apply_common_options(polyfc)

polyregion_apply_common_options(polyfc-tests)

polytest_configure_suite(polyfc-tests
        NAME_PREFIX polyfc
        DRIVER "${LLVM_DIST_DIR}/bin/flang-new"
        LIB "$<TARGET_FILE:polydco>"
        LIBRARY_PATH "$<TARGET_FILE_DIR:polydco>"
        INCLUDE "${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polyfc>/$<TARGET_NAME:polydco>/include:${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polyinvoke>/include:${CMAKE_SOURCE_DIR}/include:${CMAKE_SOURCE_DIR}/include"
        GLOB "check_*.*"
        TEST_FILES_OUT _polyfc_test_files)

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
install(TARGETS polydco polydco-static polyreflect-rt
        LIBRARY DESTINATION lib/polyfc/lib
        ARCHIVE DESTINATION lib/polyfc/lib
        RUNTIME DESTINATION lib/polyfc/lib
        FILE_SET HEADERS DESTINATION lib/polyfc/include)
if (NOT POLYREGION_FUSED_DRIVER)
    install(TARGETS polyfc-flang-plugin polyreflect-plugin
            LIBRARY DESTINATION lib/polyfc/lib
            RUNTIME DESTINATION lib/polyfc/lib)
endif ()
