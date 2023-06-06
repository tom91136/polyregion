

add_library(PolySTLPlugin SHARED ../shared/cxa_thread_atexit.cpp plugin.cpp remapper.cpp)


target_include_directories(PolySTLPlugin PRIVATE
        .
        ../compiler
        ../include
        ${CLANG_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        )
target_link_libraries(PolySTLPlugin PRIVATE PolyAST fmt::fmt-header-only)


set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>")

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>")

target_compile_options(PolySTLPlugin PRIVATE ${COMPILE_OPTIONS})

target_link_options(PolySTLPlugin PRIVATE ${LINK_OPTIONS})

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    add_custom_command(
            TARGET PolySTLPlugin POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:PolySTLPlugin>)
endif ()
