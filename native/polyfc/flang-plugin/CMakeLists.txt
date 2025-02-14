include(AddLLVM)

add_llvm_library(polyfc-flang-plugin
        MODULE
        plugin.cpp
        rewriter.cpp
        remapper.cpp
        codegen.cpp
        mlir_utils.cpp
        parallel.cpp
        mirrors.cpp
        fexpr.cpp
        PLUGIN_TOOL flang
        PARTIAL_SOURCES_INTENDED)
set_property(TARGET polyfc-flang-plugin PROPERTY CXX_STANDARD 17)

if (NOT WIN32)
    target_include_directories(polyfc-flang-plugin PRIVATE
            ${LLVM_INCLUDE_DIRS}
            ${CLANG_INCLUDE_DIRS}
            ${FLANG_INCLUDE_DIRS}
            ${MLIR_INCLUDE_DIRS}
            ../polydco
    )
    target_compile_definitions(polyfc-flang-plugin PRIVATE ${LLVM_DEFINITIONS} FLANG_LITTLE_ENDIAN)
    target_compile_options(polyfc-flang-plugin PRIVATE ${COMPILE_OPTIONS}
            -Wno-deprecated-copy
            -Wno-c++20-designator
            -Wno-c++20-extensions
    )
    target_link_options(polyfc-flang-plugin PRIVATE ${PLUGIN_LINK_OPTIONS})
    target_link_libraries(polyfc-flang-plugin
            PRIVATE
            polycommon
            polyfront-options
            PolyAST
            Aspartame::Aspartame
            fmt::fmt-header-only
    )
endif ()


if (NOT WIN32)
    target_precompile_headers(polyfc-flang-plugin PRIVATE

            <optional>
            <ostream>
            <unordered_map>
            <utility>
            <vector>
            <cstdio>

            [["aspartame/all.hpp"]]


            [["flang/Optimizer/Dialect/FIROps.h"]]
            [["flang/Optimizer/Dialect/FIRType.h"]]
            [["flang/Optimizer/Dialect/Support/FIRContext.h"]]
            [["flang/Optimizer/Support/Utils.h"]]

            [["mlir/Dialect/LLVMIR/LLVMDialect.h"]]
            [["mlir/Dialect/LLVMIR/LLVMTypes.h"]]
            [["mlir/IR/BuiltinAttributes.h"]]
            [["mlir/IR/BuiltinTypes.h"]]
            [["mlir/IR/Dialect.h"]]
            [["mlir/IR/Value.h"]]

            [["clang/Basic/Diagnostic.h"]]

            [["llvm/ADT/DenseSet.h"]]
            [["llvm/IR/Constant.h"]]
            [["llvm/Support/Casting.h"]]


    )
endif ()