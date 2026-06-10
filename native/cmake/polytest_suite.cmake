function(polytest_configure_suite target)
    cmake_parse_arguments(ARG "RECURSIVE" "NAME_PREFIX;DRIVER;LIB;LIBRARY_PATH;INCLUDE;GLOB;TEST_FILES_OUT" "" ${ARGN})
    foreach (req NAME_PREFIX DRIVER GLOB TEST_FILES_OUT)
        if (NOT ARG_${req})
            message(FATAL_ERROR "polytest_configure_suite: ${req} required")
        endif ()
    endforeach ()

    get_filename_component(POLYTEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
    # Isolated cache for generated test binaries so `cmake clean` removes them and a stale build
    # cannot shadow a fresh run.
    set(POLYTEST_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-cache")
    file(MAKE_DIRECTORY "${POLYTEST_CACHE_DIR}")

    if (ARG_RECURSIVE)
        file(GLOB _top CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test/${ARG_GLOB}")
        file(GLOB _sub CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test/**/${ARG_GLOB}")
        set(_paths ${_top} ${_sub})
    else ()
        file(GLOB _paths CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test/${ARG_GLOB}")
    endif ()
    list(LENGTH _paths _n)
    message(STATUS "${ARG_NAME_PREFIX} tests: ${_n} files registered (x 2 modes)")

    set(POLYTEST_INPUT_FILES "")
    foreach (_f IN LISTS _paths)
        string(APPEND POLYTEST_INPUT_FILES "\"${_f}\", ")
    endforeach ()

    set(POLYTEST_DRIVER "${ARG_DRIVER}")
    set(POLYTEST_LIB "${ARG_LIB}")
    set(POLYTEST_LIBRARY_PATH "${ARG_LIBRARY_PATH}")
    set(POLYTEST_INCLUDE "${ARG_INCLUDE}")

    configure_file("${CMAKE_SOURCE_DIR}/polytest/test_all.h.in" "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in")
    file(GENERATE
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h"
            INPUT "${CMAKE_CURRENT_BINARY_DIR}/test_all.h.in")

    # XXX one ctest entry per test file x {offload, passthrough}.
    # Cross-process device serialisation lives in DeviceLock (polyinvoke/include/polyinvoke/device_lock.h).
    polytest_discover_tests(${target}
            NAME_PREFIX ${ARG_NAME_PREFIX}
            WORKING_DIRECTORY "${POLYTEST_CACHE_DIR}"
            DIST_BIN ${target}
            TEST_FILES ${_paths})

    # Wipe the whole cache on `clean` so stale test binaries cannot survive a rebuild.
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${POLYTEST_CACHE_DIR}")
    add_custom_target(clean-${ARG_NAME_PREFIX}-tests
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${POLYTEST_CACHE_DIR}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${POLYTEST_CACHE_DIR}"
            COMMENT "Removing ${ARG_NAME_PREFIX} test-cache binaries from ${POLYTEST_CACHE_DIR}")

    set(${ARG_TEST_FILES_OUT} ${_paths} PARENT_SCOPE)
endfunction()
