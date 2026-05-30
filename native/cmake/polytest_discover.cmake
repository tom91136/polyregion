# polytest_discover_tests(<target> NAME_PREFIX <p> [WORKING_DIRECTORY <d>] [LABELS <l>]
#                         [DIST_BIN <name>] [ENVIRONMENT_MODIFICATION <pairs>] [TEST_FILES <files>])
function(polytest_discover_tests target)
    cmake_parse_arguments(ARG "" "NAME_PREFIX;WORKING_DIRECTORY;LABELS;DIST_BIN;ENVIRONMENT_MODIFICATION" "TEST_FILES" ${ARGN})
    if (NOT ARG_NAME_PREFIX)
        message(FATAL_ERROR "polytest_discover_tests: NAME_PREFIX required")
    endif ()

    set(_stamp_dir "${CMAKE_CURRENT_BINARY_DIR}/polytest-discover")
    set(_ids_file "${_stamp_dir}/${target}.ids")
    set(_tests_file "${_stamp_dir}/${target}-tests.cmake")
    set(_props "")
    if (ARG_WORKING_DIRECTORY)
        list(APPEND _props "-DWORKING_DIRECTORY=${ARG_WORKING_DIRECTORY}")
    endif ()
    if (ARG_LABELS)
        list(APPEND _props "-DLABELS=${ARG_LABELS}")
    endif ()
    if (ARG_ENVIRONMENT_MODIFICATION)
        list(APPEND _props "-DENVIRONMENT_MODIFICATION=${ARG_ENVIRONMENT_MODIFICATION}")
    endif ()
    set(_dist_outputs "")
    if (ARG_DIST_BIN AND NOT ARG_TEST_FILES)
        set(_dist_tests_file "${_stamp_dir}/${target}-dist-tests.cmake")
        list(APPEND _props
                "-DDIST_TESTS_FILE=${_dist_tests_file}"
                "-DDIST_BIN=${ARG_DIST_BIN}"
                "-DEXE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}")
        set(_dist_outputs "${_dist_tests_file}")
    endif ()
    add_custom_command(
            OUTPUT "${_ids_file}" "${_tests_file}" ${_dist_outputs}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_stamp_dir}"
            COMMAND $<TARGET_FILE:${target}> --list-ids > "${_ids_file}"
            COMMAND ${CMAKE_COMMAND}
                    -DBINARY=$<TARGET_FILE:${target}>
                    -DIDS_FILE=${_ids_file}
                    -DTESTS_FILE=${_tests_file}
                    -DNAME_PREFIX=${ARG_NAME_PREFIX}
                    ${_props}
                    -P "${CMAKE_SOURCE_DIR}/cmake/polytest_discover_emit.cmake"
            DEPENDS ${target} ${ARG_TEST_FILES}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Discovering ${target} tasks"
            VERBATIM)
    add_custom_target(${target}-discover ALL DEPENDS "${_ids_file}" "${_tests_file}" ${_dist_outputs})

    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_tests_file}")

    if (ARG_DIST_BIN)
        install(FILES "${_ids_file}" DESTINATION test-meta RENAME "${ARG_DIST_BIN}.ids" OPTIONAL)
    endif ()
    if (_dist_outputs)
        install(FILES "${_dist_tests_file}" DESTINATION "${target}" RENAME CTestTestfile.cmake
                COMPONENT test-dist EXCLUDE_FROM_ALL OPTIONAL)
        set_property(GLOBAL APPEND_STRING PROPERTY POLYREGION_TEST_DIST_ENTRIES
                "subdirs(\"${target}\")\n")
        set_property(GLOBAL APPEND PROPERTY POLYREGION_TEST_DISCOVER_TARGETS "${target}-discover")
    endif ()
endfunction()
