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


#target_include_directories(polycpp-clang-plugin PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
#target_compile_definitions(polycpp-clang-plugin PRIVATE ${LLVM_DEFINITIONS})
#target_compile_options(polycpp-clang-plugin PRIVATE ${COMPILE_OPTIONS})
#target_link_options(polycpp-clang-plugin PRIVATE ${PLUGIN_LINK_OPTIONS})
#target_link_libraries(polycpp-clang-plugin
#        PRIVATE
#        polycommon
#        PolyAST
#        Aspartame::Aspartame
#        fmt::fmt-header-only
#)
