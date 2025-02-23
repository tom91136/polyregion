add_subdirectory(polydco)
add_subdirectory(flang-plugin)

set(POLYFC_HEADER_SET_DIR "lib/polyfc/include")
set(POLYFC_LIB_DIR "lib/polyfc/lib")

function(add_copy_lib_targets target)
    foreach (target ${ARGN})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${target}>" "$<TARGET_FILE_DIR:polyfc>/${POLYFC_LIB_DIR}/")
    endforeach ()
    add_custom_target(${target}
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:polyfc>/${POLYFC_LIB_DIR}/"
            ${commands})
endfunction()
if (WIN32)
    add_copy_lib_targets(polyfc-lib-resource polydco-static polydco)
else ()
    add_copy_lib_targets(polyfc-lib-resource polyfc-flang-plugin polyreflect-plugin polydco-static polydco)
endif ()


add_executable(polyfc driver_main.cpp)
add_dependencies(polyfc polyfc-lib-resource)

add_executable(polyfc-test test/test_all.cpp)
add_dependencies(polyfc-test polyfc polydco)
target_include_directories(polyfc-test PRIVATE
        ../include
        polydco
        ${LLVM_INCLUDE_DIRS}
        "${CMAKE_CURRENT_BINARY_DIR}")

# === test_all.h.in variables ===
get_filename_component(POLYFC_TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
set(POLYFC_TEST_INPUT_FILES "")
file(GLOB TEST_FILE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/test/check_*.*")
foreach (TEST_FILE ${TEST_FILE_PATHS})
    string(APPEND POLYFC_TEST_INPUT_FILES "\"${TEST_FILE}\", ")
endforeach ()
set(POLYFC_FLANG_DRIVER "${LLVM_DIST_DIR}/bin/flang-new")
set(POLYFC_POLYDCO_LIB "$<TARGET_FILE:polydco>")
set(POLYFC_POLYDCO_LIBRARY_PATH "$<TARGET_FILE_DIR:polydco>")
set(POLYFC_POLYDCO_INCLUDE "${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polyfc>/$<TARGET_NAME:polydco>/include:${CMAKE_SOURCE_DIR}/$<TARGET_NAME:polyinvoke>/include:${CMAKE_SOURCE_DIR}/include:${CMAKE_SOURCE_DIR}/include")
# ===  end test_all.h.in variables ===
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test/test_all.h.in" "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in")

file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in"
)

target_include_directories(polyfc PRIVATE ../compiler ../include ${LLVM_INCLUDE_DIRS})
target_link_libraries(polyfc PRIVATE
        polycommon
        polyc-driver
        polyfront-options
        Aspartame::Aspartame
        fmt::fmt
)
target_link_libraries(polyfc-test PRIVATE
        polyfront-lit
        LLVMSupport
        fmt::fmt
        Aspartame::Aspartame
        Catch2::Catch2WithMain
)


target_compile_options(polyfc PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyfc PRIVATE ${LINK_OPTIONS})

target_compile_options(polyfc-test PRIVATE ${COMPILE_OPTIONS})
target_link_options(polyfc-test PRIVATE ${LINK_OPTIONS})

# Running `strip` on macOS drops even more stuff than just compiling with -dead_strip.
if (APPLE)
    #    add_custom_command(
    #            TARGET PolySTLPlugin POST_BUILD
    #            COMMAND strip
    #            ARGS -SXx $<TARGET_FILE:PolySTLPlugin>)
    add_custom_command(
            TARGET polyfc POST_BUILD
            COMMAND strip
            ARGS -SXx $<TARGET_FILE:polyfc>)
endif ()

