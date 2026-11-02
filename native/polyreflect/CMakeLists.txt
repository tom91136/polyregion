include(AddLLVM)

add_library(polyreflect-rt SHARED reflect-rt/rt_impl.cpp)
target_sources(polyreflect-rt
        PUBLIC FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
        reflect-rt/rt.hpp
        reflect-rt/rt_hashmap.hpp
        reflect-rt/rt_index.hpp
        reflect-rt/rt_protected.hpp
        reflect-rt/rt_reflect.hpp
)
target_include_directories(polyreflect-rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
polyregion_apply_common_options(polyreflect-rt)
target_link_libraries(polyreflect-rt PUBLIC ${CMAKE_DL_LIBS})

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
polyregion_apply_common_options(polyreflect-rt-tests)
if (APPLE)
    target_link_options(polyreflect-rt-tests PRIVATE LINKER:-undefined,dynamic_lookup,-flat_namespace)
endif ()
add_test(NAME polyreflect-rt-tests COMMAND polyreflect-rt-tests)
polyregion_test_dist_record(NAME polyreflect-rt-tests COMMAND :bin:polyreflect-rt-tests)

add_executable(polyreflect-rt-service-tests test/service.cpp)
target_include_directories(polyreflect-rt-service-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polyreflect-rt-service-tests PRIVATE Catch2::Catch2WithMain ${CMAKE_DL_LIBS})
polyregion_apply_common_options(polyreflect-rt-service-tests)
add_test(NAME polyreflect-rt-service-tests COMMAND polyreflect-rt-service-tests)
polyregion_test_dist_record(NAME polyreflect-rt-service-tests COMMAND :bin:polyreflect-rt-service-tests)

# Plugin source is statically linked into polycpp/polyfc under fused builds; no .so needed.
if (NOT POLYREGION_FUSED_DRIVER)
    target_include_directories(polyreflect-plugin SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(polyreflect-plugin PRIVATE ${LLVM_DEFINITIONS})
    target_compile_options(polyreflect-plugin PRIVATE ${COMPILE_OPTIONS})
    target_link_options(polyreflect-plugin PRIVATE ${PLUGIN_LINK_OPTIONS})
    polyregion_strip_asan_on_darwin(polyreflect-plugin)
    # XXX headers only (the pass uses rt_reflect::Type, a header enum) - linking the instrumented
    # rt dylib would drag asan into the dlopen'd plugin, which macOS asan forbids
    target_include_directories(polyreflect-plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(polyreflect-plugin
            PUBLIC
            polycommon
            Aspartame::Aspartame
            magic_enum::magic_enum
            fmt::fmt-header-only)

    if (POLYREGION_LLVM_DYLIB)
        set(POLYREFLECT_TEST_LLVM_LIBS LLVM)
    else ()
        llvm_map_components_to_libnames(POLYREFLECT_TEST_LLVM_LIBS
                ${LLVM_TARGETS_TO_BUILD}
                AsmParser Analysis Core Passes Support TransformUtils)
    endif ()

    add_executable(polyreflect-stack-tests test/stack_lifetime.cpp reflect-stack.cpp)
    target_include_directories(polyreflect-stack-tests SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    target_include_directories(polyreflect-stack-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(polyreflect-stack-tests PRIVATE ${LLVM_DEFINITIONS})
    target_compile_options(polyreflect-stack-tests PRIVATE ${COMPILE_OPTIONS})
    target_link_options(polyreflect-stack-tests PRIVATE ${PLUGIN_LINK_OPTIONS})
    target_link_libraries(polyreflect-stack-tests
            PRIVATE
            Catch2::Catch2WithMain
            polycommon
            Aspartame::Aspartame
            fmt::fmt-header-only
            ${POLYREFLECT_TEST_LLVM_LIBS})
    polyregion_apply_common_options(polyreflect-stack-tests)
    add_test(NAME polyreflect-stack-tests COMMAND polyreflect-stack-tests)
    polyregion_test_dist_record(NAME polyreflect-stack-tests COMMAND :bin:polyreflect-stack-tests)
endif ()
polyregion_stage_post_build(polyreflect-rt polyreflect-plugin)
