include(AddLLVM)

add_library(rt-reflect INTERFACE)
target_sources(rt-reflect
        PRIVATE
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        rt-reflect/rt.hpp
        rt-reflect/rt_hashmap.hpp
        rt-reflect/rt_memory.hpp
        rt-reflect/rt_protected.hpp
        rt-reflect/rt_reflect.hpp
)

add_llvm_pass_plugin(polystl-reflect-plugin
        plugin.cpp
        DEPENDS
        intrinsics_gen
        BUILDTREE_ONLY
)

add_executable(rt-reflect-tests test/map.cpp)
target_link_libraries(rt-reflect-tests PRIVATE Catch2::Catch2WithMain rt-reflect)
target_compile_options(rt-reflect-tests PRIVATE ${COMPILE_OPTIONS})
target_link_options(rt-reflect-tests PRIVATE ${LINK_OPTIONS})


if (NOT WIN32)
    target_include_directories(polystl-reflect-plugin PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(polystl-reflect-plugin PRIVATE ${LLVM_DEFINITIONS})
    target_compile_options(polystl-reflect-plugin PRIVATE ${COMPILE_OPTIONS})
    target_link_options(polystl-reflect-plugin PRIVATE ${PLUGIN_LINK_OPTIONS})
    target_link_libraries(polystl-reflect-plugin
            PUBLIC
            polycommon
            Aspartame::Aspartame
            rt-reflect)
endif ()

