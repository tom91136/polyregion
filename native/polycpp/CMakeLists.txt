add_subdirectory(polystl)
add_subdirectory(clang-plugin)

set(POLYCPP_HEADER_SET_DIR "lib/polycpp/include")
set(POLYCPP_LIB_DIR "lib/polycpp/lib")

# =========================== Generate header resource target ===========================
function(add_copy_header_set_targets target commands_var)
    get_target_property(header_set ${target} HEADER_SET)
    get_target_property(header_dirs ${target} HEADER_DIRS)
    list(LENGTH header_dirs header_dirs_len)
    if (header_dirs_len GREATER 1)
        message(FATAL_ERROR "More than one header dirs for dependent target ${target}, cannot create relative paths (dirs=${header_dirs})")
    endif ()
    foreach (absolute_file ${header_set})
        cmake_path(RELATIVE_PATH absolute_file BASE_DIRECTORY "${header_dirs}" OUTPUT_VARIABLE relative_file)
        cmake_path(GET relative_file EXTENSION LAST_ONLY header_extension)
        if (header_extension STREQUAL ".impl")
            cmake_path(GET relative_file STEM relative_file)
        endif ()
        list(APPEND ${commands_var} COMMAND ${CMAKE_COMMAND} -E copy "${absolute_file}" "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_HEADER_SET_DIR}/${relative_file}")
    endforeach ()
    set(${commands_var} ${${commands_var}} PARENT_SCOPE)
#    add_custom_target(${name} ${commands})
endfunction()

foreach (header_target polycommon polystl polyinvoke-obj polyrt-obj polyreflect-rt)
    add_copy_header_set_targets(${header_target} header_commands)
endforeach ()
add_custom_target(polycpp-header-resource ${header_commands})

# =======================================================================================


# =========================== Generate header resource target ===========================
function(add_copy_lib_targets target)
    foreach (target ${ARGN})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${target}>" "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_LIB_DIR}/")
    endforeach ()
    add_custom_target(${target}
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:polycpp>/${POLYCPP_LIB_DIR}/"
            ${commands})
endfunction()
if (WIN32)
    add_copy_lib_targets(polycpp-lib-resource polystl-static polystl)
else ()
    add_copy_lib_targets(polycpp-lib-resource polycpp-clang-plugin polyreflect-plugin polystl-static polystl)
endif ()
# =======================================================================================


add_executable(polycpp driver_main.cpp)
add_dependencies(polycpp polycpp-lib-resource polycpp-header-resource)

add_executable(polycpp-tests test/test_all.cpp)
add_dependencies(polycpp-tests polycpp polystl)
target_include_directories(polycpp-tests PRIVATE
        ../include
        polystl
        ${LLVM_INCLUDE_DIRS}
        ${fire_SOURCE_DIR}
        "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(polycpp-tests PRIVATE
        POLYREGION_TEST_PROFILE_DIR="${CMAKE_SOURCE_DIR}/test-profiles")

# === test_all.h.in variables ===
get_filename_component(POLYCPP_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
set(POLYCPP_TEST_INPUT_FILES "")
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/check_*.cpp")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYCPP_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/**/check_*.cpp")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYCPP_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()
set(POLYCPP_CLANG_DRIVER "${LLVM_DIST_DIR}/bin/clang++")
set(POLYCPP_POLYSTL_LIB "$<TARGET_FILE:polystl>")
set(POLYCPP_POLYSTL_LIBRARY_PATH "$<TARGET_FILE_DIR:polystl>")
set(POLYCPP_POLYSTL_INCLUDE "${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polycpp>/$<TARGET_NAME:polystl>/include:${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polyinvoke>/include:${CMAKE_SOURCE_DIR}/include:${CMAKE_SOURCE_DIR}/include")
# ===  end test_all.h.in variables ===
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test/test_all.h.in" "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in")

file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in"
)

target_include_directories(polycpp PRIVATE ../compiler ../include ${LLVM_INCLUDE_DIRS})
target_link_libraries(polycpp PRIVATE
        polyc-driver
        polyfront-options
        Aspartame::Aspartame
        fmt::fmt
)
target_link_libraries(polycpp-tests PRIVATE polytest)


target_compile_options(polycpp PRIVATE ${COMPILE_OPTIONS})
target_link_options(polycpp PRIVATE ${LINK_OPTIONS})

target_compile_options(polycpp-tests PRIVATE ${COMPILE_OPTIONS})
target_link_options(polycpp-tests PRIVATE ${LINK_OPTIONS})

add_test(NAME polycpp-offload COMMAND polycpp-tests --offload-only)
add_test(NAME polycpp-passthrough COMMAND polycpp-tests --passthrough-only)
set_tests_properties(polycpp-offload PROPERTIES RESOURCE_LOCK "gpu-cuda;gpu-hsa;gpu-spirv")

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

install(TARGETS polycpp RUNTIME DESTINATION bin)
install(TARGETS polystl polystl-static
        LIBRARY DESTINATION lib/polycpp/lib
        ARCHIVE DESTINATION lib/polycpp/lib
        RUNTIME DESTINATION lib/polycpp/lib
        FILE_SET HEADERS DESTINATION lib/polycpp/include)
if (NOT WIN32)
    install(TARGETS polycpp-clang-plugin polyreflect-plugin
            LIBRARY DESTINATION lib/polycpp/lib
            RUNTIME DESTINATION lib/polycpp/lib)
endif ()
foreach (HEADER_TARGET polycommon polyinvoke-obj polyrt-obj polyreflect-rt)
    install(TARGETS ${HEADER_TARGET} FILE_SET HEADERS DESTINATION lib/polycpp/include)
endforeach ()

