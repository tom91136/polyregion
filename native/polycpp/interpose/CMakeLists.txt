include(AddLLVM)

set(LLVM_BYE2_LINK_INTO_TOOLS ON)

add_llvm_pass_plugin(Bye2
        Bye.cpp
        DEPENDS
        intrinsics_gen
        BUILDTREE_ONLY
)

