include(AddLLVM)

add_llvm_pass_plugin(polystl-interpose-plugin
        interpose.cpp
        DEPENDS
        intrinsics_gen
        BUILDTREE_ONLY
)

if (NOT WIN32)
    target_include_directories(polystl-interpose-plugin PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(polystl-interpose-plugin PRIVATE ${LLVM_DEFINITIONS})
    target_compile_options(polystl-interpose-plugin PRIVATE ${COMPILE_OPTIONS})
    target_link_options(polystl-interpose-plugin PRIVATE ${PLUGIN_LINK_OPTIONS})
    target_link_libraries(polystl-interpose-plugin
            PUBLIC
            polycommon
            Aspartame::Aspartame)
endif ()
