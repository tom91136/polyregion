add_subdirectory(polystl)


#add_library(PolySTLPlugin SHARED ../shared/cxa_thread_atexit.cpp
#        plugin.cpp
#        remapper.cpp clang_utils.cpp ast_visitors.cpp codegen.cpp frontend.cpp rewriter.cpp)
add_executable(polycpp ../shared/cxa_thread_atexit.cpp
        tool.cpp
        remapper.cpp clang_utils.cpp ast_visitors.cpp codegen.cpp frontend.cpp rewriter.cpp)

add_dependencies(polycpp polystl polystl-static polyc-bin)


add_executable(polycpp-test test/test_all.cpp)

add_dependencies(polycpp-test polycpp)


get_filename_component(POLYCPP_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/check_*.cpp")
set(POLYCPP_TEST_INPUT_FILES "")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYCPP_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()

target_include_directories(polycpp-test PRIVATE
        ../include
        ${LLVM_INCLUDE_DIRS}
        "${CMAKE_CURRENT_BINARY_DIR}"
)
configure_file(test/test_all.h.in "${CMAKE_CURRENT_BINARY_DIR}/test_all.h")


#target_include_directories(PolySTLPlugin PRIVATE
#        .
#        ../compiler
#        ../include
#        ${CLANG_INCLUDE_DIRS}
#        ${LLVM_INCLUDE_DIRS}
#        )
target_include_directories(polycpp PRIVATE
        .
        ../compiler
        ../include
        ${CLANG_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        )

#target_link_libraries(PolySTLPlugin PRIVATE PolyAST fmt::fmt-header-only)

llvm_map_components_to_libnames(LLVM_LIBS
        ${LLVM_TARGETS_TO_BUILD}
        codegen
        option
        support
        frontendopenmp
        targetparser
)


target_link_libraries(polycpp PRIVATE
        PolyAST
        fmt::fmt-header-only

        ${LLVM_LIBS}

        clangBasic
        clangCodeGen
        clangDriver
        clangFrontend
        clangFrontendTool
        clangSerialization
        )
target_link_libraries(polycpp-test PRIVATE
        LLVMSupport
        fmt::fmt-header-only
        Catch2::Catch2WithMain)


get_filename_component(Clang_RESOURCE_DIR "${Clang_DIR}/../../clang" ABSOLUTE)

add_custom_command(TARGET polycpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${Clang_RESOURCE_DIR}" "$<TARGET_FILE_DIR:polycpp>/lib/clang"
        )


#target_compile_options(PolySTLPlugin PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polycpp PRIVATE ${COMPILE_OPTIONS})

target_link_options(polycpp PRIVATE ${LINK_OPTIONS})
#target_link_options(PolySTLPlugin PRIVATE ${LINK_OPTIONS})

target_compile_options(polycpp-test PRIVATE ${COMPILE_OPTIONS})
target_link_options(polycpp-test PRIVATE ${LINK_OPTIONS})

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    #    add_custom_command(
    #            TARGET PolySTLPlugin POST_BUILD
    #            COMMAND strip
    #            ARGS -SXx $<TARGET_FILE:PolySTLPlugin>)
    add_custom_command(
            TARGET polycpp POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polycpp>)
endif ()

