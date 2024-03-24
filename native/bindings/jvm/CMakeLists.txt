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


add_library(polyc-JNIshim SHARED ../../shared/cxa_thread_atexit.cpp generated/mirror.cpp shim.cpp)
add_library(polyc-JNI SHARED ../../shared/cxa_thread_atexit.cpp generated/mirror.cpp compiler.cpp)
add_library(polyrt-JNI SHARED ../../shared/cxa_thread_atexit.cpp generated/mirror.cpp runtime.cpp)


target_include_directories(polyc-JNIshim PRIVATE
        ../../include

        ${JNI_INCLUDE_DIRS})
target_include_directories(polyc-JNI PRIVATE
#        .
#        ../../compiler

        ${JNI_INCLUDE_DIRS})
target_include_directories(polyrt-JNI PRIVATE
#        .
#        ../../runtime

        ${JNI_INCLUDE_DIRS})

target_link_libraries(polyc-JNIshim PRIVATE ${DL_LIB} PUBLIC polycommon)
target_link_libraries(polyc-JNI PRIVATE polyc)
target_link_libraries(polyrt-JNI PRIVATE polyrt-static Aspartame::Aspartame)



target_compile_options(polyc-JNIshim PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyc-JNI PRIVATE ${COMPILE_OPTIONS})
target_compile_options(polyrt-JNI PRIVATE ${COMPILE_OPTIONS})

target_link_options(polyc-JNIshim PRIVATE ${LINK_OPTIONS})
target_link_options(polyc-JNI PRIVATE ${LINK_OPTIONS})
target_link_options(polyrt-JNI PRIVATE ${LINK_OPTIONS})

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    add_custom_command(
            TARGET polyrt-JNI POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polyrt-JNI>)
    add_custom_command(
            TARGET polyc-JNI POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polyc-JNI>)
endif ()
