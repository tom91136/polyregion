find_package(JNI REQUIRED)

add_library(java-compiler SHARED java-compiler/impl.cpp)
target_include_directories(java-compiler PRIVATE
        ../compiler
        ../include
        java-compiler
        ${JNI_INCLUDE_DIRS})
target_link_libraries(java-compiler PRIVATE compiler-lib)

add_library(java-runtime SHARED java-runtime/impl.cpp)
target_include_directories(java-runtime PRIVATE
        ../runtime
        ../include
        java-runtime
        ${JNI_INCLUDE_DIRS})
target_link_libraries(java-runtime PRIVATE runtime-lib)

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

target_compile_options(java-compiler PRIVATE ${COMPILE_OPTIONS})
target_compile_options(java-runtime PRIVATE ${COMPILE_OPTIONS})


target_link_options(java-compiler PRIVATE ${LINK_OPTIONS})
target_link_options(java-runtime PRIVATE ${LINK_OPTIONS})
