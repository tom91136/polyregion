find_package(JNI REQUIRED)

add_library(java-codegen SHARED java-codegen/impl.cpp)
target_include_directories(java-codegen PRIVATE
        ../codegen
        java-codegen
        ${JNI_INCLUDE_DIRS})
target_link_libraries(java-codegen PRIVATE codegen-lib)

add_library(java-runtime SHARED java-runtime/impl.cpp)
target_include_directories(java-runtime PRIVATE
        ../runtime
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

target_compile_options(java-codegen PRIVATE ${COMPILE_OPTIONS})
target_compile_options(java-runtime PRIVATE ${COMPILE_OPTIONS})


target_link_options(java-codegen PRIVATE ${LINK_OPTIONS})
target_link_options(java-runtime PRIVATE ${LINK_OPTIONS})
