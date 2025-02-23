include(AddLLVM)

add_library(polyreflect-rt INTERFACE)
target_sources(polyreflect-rt
        PRIVATE
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        reflect-rt/rt.hpp
        reflect-rt/rt_hashmap.hpp
        reflect-rt/rt_memory.hpp
        reflect-rt/rt_protected.hpp
        reflect-rt/rt_reflect.hpp
)

add_llvm_pass_plugin(polyreflect-plugin
        plugin.cpp
        reflect-stack.cpp
        reflect-mem.cpp
        interpose.cpp
        DEPENDS
        intrinsics_gen
        BUILDTREE_ONLY
)

add_executable(polyreflect-rt-tests test/map.cpp)
target_link_libraries(polyreflect-rt-tests PRIVATE Catch2::Catch2WithMain polyreflect-rt)
target_compile_options(polyreflect-rt-tests PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyreflect-rt-tests PRIVATE ${LINK_OPTIONS})
if (APPLE)
    target_link_options(polyreflect-rt-tests PRIVATE LINKER:-undefined,dynamic_lookup,-flat_namespace)
endif ()

if (NOT WIN32)
    target_include_directories(polyreflect-plugin PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(polyreflect-plugin PRIVATE ${LLVM_DEFINITIONS})
    target_compile_options(polyreflect-plugin PRIVATE ${COMPILE_OPTIONS})
    target_link_options(polyreflect-plugin PRIVATE ${PLUGIN_LINK_OPTIONS})
    target_link_libraries(polyreflect-plugin
            PUBLIC
            polycommon
            Aspartame::Aspartame
            magic_enum::magic_enum
            polyreflect-rt)
endif ()

