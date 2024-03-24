add_subdirectory(polystl)
add_subdirectory(interpose)

set(POLYCPP_HEADER_SET_DIR "lib/polycpp/include")
set(POLYCPP_LIB_DIR "lib/polycpp/lib")

add_executable(polycpp
        ../shared/cxa_thread_atexit.cpp
        options.cpp
        remapper.cpp
        clang_utils.cpp
        ast_visitors.cpp
        codegen.cpp
        frontend.cpp
        rewriter.cpp
        driver_clang.cpp
        driver_main.cpp
)
add_dependencies(polycpp polystl polystl-static polystl-interpose)


add_executable(polycpp-test test/test_all.cpp)
add_dependencies(polycpp-test polycpp)
target_include_directories(polycpp-test PRIVATE
        ../include
        ${LLVM_INCLUDE_DIRS}
        "${CMAKE_CURRENT_BINARY_DIR}"
)

# === test_all.h.in variables ===
get_filename_component(POLYCPP_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
set(POLYCPP_TEST_INPUT_FILES "")
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/check_*.cpp")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYCPP_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/**/check_*.cpp")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYCPP_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()
set(POLYCPP_POLYSTL_LIB "$<TARGET_FILE:polystl>")
set(POLYCPP_POLYSTL_LIBRARY_PATH "$<TARGET_FILE_DIR:polystl>")
set(POLYCPP_POLYSTL_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/${POLYCPP_HEADER_SET_DIR}:${CMAKE_SOURCE_DIR}/include")
# ===  end test_all.h.in variables ===
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test/test_all.h.in" "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in")

file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in"
)

target_include_directories(polycpp PRIVATE
        .
        ../compiler
        ../include
        ${CLANG_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        )

llvm_map_components_to_libnames(LLVM_LIBS
        ${LLVM_TARGETS_TO_BUILD}
        codegen
        option
        support
        frontendopenmp
        targetparser
        extensions
)

target_link_libraries(polycpp PRIVATE
        PolyAST
        polyc-driver
        Aspartame::Aspartame
        fmt::fmt-header-only


        ${LLVM_LIBS}

        clangBasic
        clangCodeGen
        clangDriver
        clangFrontend
        clangFrontendTool
        clangSerialization
        )
target_link_libraries(polycpp-test PRIVATE
        polycommon
        LLVMSupport
        fmt::fmt-header-only
        Aspartame::Aspartame
        Catch2::Catch2WithMain
)

if (NOT WIN32)
    target_precompile_headers(polycpp PRIVATE

            remapper.h
            ast_visitors.h
            rewriter.h
            codegen.h

            [["clang/AST/ASTConsumer.h"]]
            [["clang/AST/ASTContext.h"]]
            [["clang/AST/Decl.h"]]
            [["clang/AST/DeclCXX.h"]]
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

            <optional>
            <ostream>
            <unordered_map>
            <utility>
            <vector>
    )
endif ()

function(add_copy_header_set_targets TARGET DEPENDENT_TARGET DESTINATION)
    get_target_property(header_set ${DEPENDENT_TARGET} HEADER_SET)
    get_target_property(header_dirs ${DEPENDENT_TARGET} HEADER_DIRS)
    list(LENGTH header_dirs header_dirs_len)
    if (header_dirs_len GREATER 1)
        message(FATAL_ERROR "More than one header dirs for dependent target ${DEPENDENT_TARGET}, cannot create relative paths (dirs=${header_dirs})")
    endif ()
    foreach (absolute_file ${header_set})
        cmake_path(RELATIVE_PATH absolute_file BASE_DIRECTORY "${header_dirs}" OUTPUT_VARIABLE relative_file)
        add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND
                ${CMAKE_COMMAND} -E copy "${absolute_file}" "$<TARGET_FILE_DIR:${TARGET}>/${DESTINATION}/${relative_file}")
    endforeach ()
endfunction()

# === copy polycpp system libs and headers ===
foreach (header_target polycommon polyrt polystl)
    add_copy_header_set_targets(polycpp ${header_target} "${POLYCPP_HEADER_SET_DIR}")
endforeach ()
foreach (lib_target polystl polystl-static polystl-interpose)
    add_custom_command(TARGET polycpp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_LIB_DIR}/"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${lib_target}>" "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_LIB_DIR}/")
endforeach ()
get_filename_component(Clang_RESOURCE_DIR "${Clang_DIR}/../../clang" ABSOLUTE)
add_custom_command(TARGET polycpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${Clang_RESOURCE_DIR}" "$<TARGET_FILE_DIR:polycpp>/lib/clang")
# === end copy polycpp system libs and headers ===


target_compile_options(polycpp PRIVATE ${COMPILE_OPTIONS})
target_link_options(polycpp PRIVATE ${LINK_OPTIONS})

target_compile_options(polycpp-test PRIVATE ${COMPILE_OPTIONS})
target_link_options(polycpp-test PRIVATE ${LINK_OPTIONS})

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

