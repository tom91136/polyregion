include(AddLLVM)

add_llvm_pass_plugin(polystl-interpose
        interpose.cpp
        DEPENDS
        intrinsics_gen
        BUILDTREE_ONLY
)

target_include_directories(polystl-interpose PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(polystl-interpose PUBLIC LLVMSupport LLVMPasses Aspartame::Aspartame)
