include(AddLLVM)

add_llvm_library(polycpp-clang-plugin
        MODULE
        plugin.cpp
        remapper.cpp
        clang_utils.cpp
        ast_visitors.cpp
        codegen.cpp
        rewriter.cpp

        PLUGIN_TOOL clang
        PARTIAL_SOURCES_INTENDED)

if (NOT WIN32)
    target_include_directories(polycpp-clang-plugin PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
    target_compile_definitions(polycpp-clang-plugin PRIVATE ${LLVM_DEFINITIONS})
    target_compile_options(polycpp-clang-plugin PRIVATE ${COMPILE_OPTIONS})
    target_link_options(polycpp-clang-plugin PRIVATE ${PLUGIN_LINK_OPTIONS})
    target_link_libraries(polycpp-clang-plugin
            PRIVATE
            polycommon
            polyfront-options
            PolyAST
            Aspartame::Aspartame
            fmt::fmt-header-only
    )
endif ()


if (NOT WIN32)
    target_precompile_headers(polycpp-clang-plugin PRIVATE

            remapper.h
            ast_visitors.h
            rewriter.h
            codegen.h

            [["aspartame/all.hpp"]]


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