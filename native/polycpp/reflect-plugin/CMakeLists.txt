include(AddLLVM)

add_library(ptr-reflect-rt INTERFACE)
target_sources(ptr-reflect-rt
        PRIVATE
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        ptr-reflect-rt/rt.hpp
        ptr-reflect-rt/rt_hashmap.hpp
        ptr-reflect-rt/rt_memory.hpp
        ptr-reflect-rt/rt_protected.hpp
        ptr-reflect-rt/rt_reflect.hpp
)


add_llvm_pass_plugin(polystl-reflect-plugin
        plugin.cpp
        DEPENDS
        intrinsics_gen
        BUILDTREE_ONLY
)

if (NOT WIN32)
    target_include_directories(polystl-reflect-plugin PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(polystl-reflect-plugin PRIVATE ${LLVM_DEFINITIONS})
    target_compile_options(polystl-reflect-plugin PRIVATE ${COMPILE_OPTIONS})
    target_link_options(polystl-reflect-plugin PRIVATE ${PLUGIN_LINK_OPTIONS})
    target_link_libraries(polystl-reflect-plugin
            PUBLIC
            polycommon
            Aspartame::Aspartame
            ptr-reflect-rt)
endif ()

