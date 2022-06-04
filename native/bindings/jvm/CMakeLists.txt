if (NOT JAVA_HOME)
    set(JAVA_HOME $ENV{JAVA_HOME})
endif ()

set(JNI_INCLUDE_DIRS "${JAVA_HOME}/include/")
if (NOT EXISTS "${JNI_INCLUDE_DIRS}")
    message(FATAL_ERROR "`${JNI_INCLUDE_DIRS}` is not a valid directory to JNI headers")
endif ()

set(JNI_INCLUDE_DIRS ${JNI_INCLUDE_DIRS};${JNI_INCLUDE_DIRS}/linux)


add_library(polyregion-compiler-jvm SHARED generated/mirror.cpp compiler.cpp)
target_include_directories(polyregion-compiler-jvm PRIVATE
        ../../compiler
        .
        generated
        ${JNI_INCLUDE_DIRS})
target_link_libraries(polyregion-compiler-jvm PRIVATE compiler-lib)

add_library(polyregion-runtime-jvm SHARED generated/mirror.cpp runtime.cpp)
target_include_directories(polyregion-runtime-jvm PRIVATE
        ../../runtime
        .
        generated
        ${JNI_INCLUDE_DIRS})
target_link_libraries(polyregion-runtime-jvm PRIVATE runtime-lib)

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>"
        )

target_compile_options(polyregion-compiler-jvm PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyregion-runtime-jvm PRIVATE ${COMPILE_OPTIONS})


target_link_options(polyregion-compiler-jvm PRIVATE ${LINK_OPTIONS})
target_link_options(polyregion-runtime-jvm PRIVATE ${LINK_OPTIONS})
