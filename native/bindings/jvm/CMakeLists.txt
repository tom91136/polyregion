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


add_library(polyc-JNIshim SHARED generated/mirror.cpp shim.cpp)
add_library(polyc-JNI SHARED generated/mirror.cpp compiler.cpp)
add_library(polyinvoke-JNI SHARED generated/mirror.cpp runtime.cpp)
add_library(polyrt-JNI SHARED polyrt.cpp)


target_include_directories(polyc-JNIshim PRIVATE
        ../../include
        ../../polyfront/rt/include

        ${JNI_INCLUDE_DIRS})
target_include_directories(polyc-JNI PRIVATE
#        .
#        ../../compiler
        ../../polyfront
        ../../polyfront/rt/include

        ${JNI_INCLUDE_DIRS})
target_include_directories(polyinvoke-JNI PRIVATE
#        .
#        ../../runtime
        ../../polyfront/rt/include

        ${JNI_INCLUDE_DIRS})

target_include_directories(polyrt-JNI PRIVATE
        ${JNI_INCLUDE_DIRS})

target_link_libraries(polyc-JNIshim PRIVATE ${DL_LIB} PUBLIC polycommon)
target_link_libraries(polyc-JNI PRIVATE polyc Aspartame::Aspartame zstd::libzstd_static)
target_link_libraries(polyinvoke-JNI PRIVATE polyinvoke-static Aspartame::Aspartame)
target_link_libraries(polyrt-JNI PRIVATE polyinvoke-obj polyrt-obj polyreflect-rt Aspartame::Aspartame)



polyregion_apply_common_options(polyc-JNIshim)
polyregion_apply_common_options(polyc-JNI)
polyregion_apply_common_options(polyinvoke-JNI)
polyregion_apply_common_options(polyrt-JNI)
set(_polyrt_jni_exports JNI_ Java_ polyrt_)
if (APPLE)
    set(_polyrt_jni_exp "${CMAKE_CURRENT_BINARY_DIR}/polyrt-JNI.exp")
    file(WRITE "${_polyrt_jni_exp}" "")
    foreach (_prefix ${_polyrt_jni_exports})
        file(APPEND "${_polyrt_jni_exp}" "_${_prefix}*\n")
    endforeach ()
    target_link_options(polyrt-JNI PRIVATE LINKER:-undefined,dynamic_lookup LINKER:-exported_symbols_list,${_polyrt_jni_exp})
    set_target_properties(polyrt-JNI PROPERTIES INSTALL_RPATH "@loader_path/polycpp/lib" LINK_DEPENDS "${_polyrt_jni_exp}")
elseif (UNIX)
    set(_polyrt_jni_map "${CMAKE_CURRENT_BINARY_DIR}/polyrt-JNI.map")
    file(WRITE "${_polyrt_jni_map}" "{\n  global:\n")
    foreach (_prefix ${_polyrt_jni_exports})
        file(APPEND "${_polyrt_jni_map}" "    ${_prefix}*;\n")
    endforeach ()
    file(APPEND "${_polyrt_jni_map}" "  local:\n    *;\n};\n")
    target_link_options(polyrt-JNI PRIVATE LINKER:--exclude-libs,ALL LINKER:--version-script=${_polyrt_jni_map})
    set_target_properties(polyrt-JNI PROPERTIES INSTALL_RPATH "$ORIGIN/polycpp/lib" LINK_DEPENDS "${_polyrt_jni_map}")
endif ()

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    add_custom_command(
            TARGET polyinvoke-JNI POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polyinvoke-JNI>)
    add_custom_command(
            TARGET polyc-JNI POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polyc-JNI>)
endif ()

install(TARGETS polyc-JNI polyc-JNIshim polyinvoke-JNI polyrt-JNI
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib)
