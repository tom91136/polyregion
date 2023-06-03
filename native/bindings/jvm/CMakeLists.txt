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


add_library(Polyc-JNIshim SHARED ../../shared/cxa_thread_atexit.cpp generated/mirror.cpp shim.cpp)
add_library(Polyc-JNI SHARED ../../shared/cxa_thread_atexit.cpp generated/mirror.cpp compiler.cpp)
add_library(PolyRT-JNI SHARED ../../shared/cxa_thread_atexit.cpp generated/mirror.cpp runtime.cpp)


target_include_directories(Polyc-JNIshim PRIVATE
        ../../include

        ${JNI_INCLUDE_DIRS})
target_include_directories(Polyc-JNI PRIVATE
        .
        ../../compiler

        ${JNI_INCLUDE_DIRS})
target_include_directories(PolyRT-JNI PRIVATE
        .
        ../../runtime

        ${JNI_INCLUDE_DIRS})

target_link_libraries(Polyc-JNIshim PRIVATE ${DL_LIB})
target_link_libraries(Polyc-JNI PRIVATE Polyc)
target_link_libraries(PolyRT-JNI PRIVATE PolyRT)

set(COMPILE_OPTIONS
        ${COMMON_COMPILE_OPTIONS}
        "$<$<CONFIG:RELEASE>:${COMMON_RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>")

set(LINK_OPTIONS
        ${COMMON_LINK_OPTIONS}
        ${STRICT_LINKER_OPTIONS}
        "$<$<CONFIG:DEBUG>:${COMMON_DEBUG_OPTIONS}>")

target_compile_options(Polyc-JNIshim PRIVATE ${COMPILE_OPTIONS})
target_compile_options(Polyc-JNI PRIVATE ${COMPILE_OPTIONS})
target_compile_options(PolyRT-JNI PRIVATE ${COMPILE_OPTIONS})

target_link_options(Polyc-JNIshim PRIVATE ${LINK_OPTIONS})
target_link_options(Polyc-JNI PRIVATE ${LINK_OPTIONS})
target_link_options(PolyRT-JNI PRIVATE ${LINK_OPTIONS})

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    add_custom_command(
            TARGET PolyRT-JNI POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:PolyRT-JNI>)
    add_custom_command(
            TARGET Polyc-JNI POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:Polyc-JNI>)
endif ()
