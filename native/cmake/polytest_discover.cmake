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
    set(_dist_args "")
    set(_dist_outputs "")
    if (ARG_DIST_BIN)
        set(_dist_tests_file "${_stamp_dir}/${target}-dist-tests.cmake")
        set(_dist_args
                --emit-dist-ctest "${_dist_tests_file}"
                --emit-dist-binary "\${CMAKE_CURRENT_LIST_DIR}/../bin/${ARG_DIST_BIN}${CMAKE_EXECUTABLE_SUFFIX}")
        if (ARG_TEST_FILES)
            list(APPEND _dist_args --emit-dist-subdir "test/${ARG_NAME_PREFIX}")
        endif ()
        set(_dist_outputs "${_dist_tests_file}")
    endif ()
    set(_workdir_args "")
    if (ARG_WORKING_DIRECTORY)
        set(_workdir_args --emit-workdir "${ARG_WORKING_DIRECTORY}")
    endif ()
    add_custom_command(
            OUTPUT "${_ids_file}" "${_tests_file}" ${_dist_outputs}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_stamp_dir}"
            COMMAND $<TARGET_FILE:${target}> --list-ids > "${_ids_file}"
            COMMAND $<TARGET_FILE:${target}>
                    --emit-ctest "${_tests_file}"
                    --emit-prefix "${ARG_NAME_PREFIX}"
                    --emit-binary "$<TARGET_FILE:${target}>"
                    ${_workdir_args}
                    ${_dist_args}
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
