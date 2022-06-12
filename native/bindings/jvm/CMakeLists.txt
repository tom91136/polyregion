if (NOT JAVA_HOME)
    set(JAVA_HOME $ENV{JAVA_HOME})
endif ()

set(JNI_INCLUDE_DIRS "${JAVA_HOME}/include/")
if (NOT EXISTS "${JNI_INCLUDE_DIRS}")
    message(FATAL_ERROR "`${JNI_INCLUDE_DIRS}` is not a valid directory to JNI headers")
endif ()
if (WIN32)
    set(JNI_INCLUDE_DIRS ${JNI_INCLUDE_DIRS};${JNI_INCLUDE_DIRS}/win32)
elseif (UNIX)
    if (APPLE)
        set(JNI_INCLUDE_DIRS ${JNI_INCLUDE_DIRS};${JNI_INCLUDE_DIRS}/darwin)
    endif ()
    # JDK8 uses darwin, newer ones use linux in certain cases, so we append both
    set(JNI_INCLUDE_DIRS ${JNI_INCLUDE_DIRS};${JNI_INCLUDE_DIRS}/linux)
endif ()


add_library(polyregion-shim-jvm SHARED generated/mirror.cpp shim.cpp)
add_library(polyregion-compiler-jvm SHARED generated/mirror.cpp compiler.cpp)
add_library(polyregion-runtime-jvm SHARED generated/mirror.cpp runtime.cpp)


target_include_directories(polyregion-shim-jvm PRIVATE
        ../../include

        ${JNI_INCLUDE_DIRS})
target_include_directories(polyregion-compiler-jvm PRIVATE
        .
        ../../compiler

        ${JNI_INCLUDE_DIRS})
target_include_directories(polyregion-runtime-jvm PRIVATE
        .
        ../../runtime

        ${JNI_INCLUDE_DIRS})

target_link_libraries(polyregion-shim-jvm PRIVATE ${WINDOWS_LIBS})
target_link_libraries(polyregion-compiler-jvm PRIVATE compiler-lib)
target_link_libraries(polyregion-runtime-jvm PRIVATE runtime-lib)

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>")

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>")

target_compile_options(polyregion-shim-jvm PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyregion-compiler-jvm PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyregion-runtime-jvm PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyregion-shim-jvm PRIVATE ${LINK_OPTIONS})
target_link_options(polyregion-compiler-jvm PRIVATE ${LINK_OPTIONS})
target_link_options(polyregion-runtime-jvm PRIVATE ${LINK_OPTIONS})

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    add_custom_command(
            TARGET polyregion-runtime-jvm POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polyregion-runtime-jvm>)
    add_custom_command(
            TARGET polyregion-compiler-jvm POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polyregion-compiler-jvm>)
endif ()
