# polyregion_add_polytest_suite(<name> SUITE <file> [DEPENDS <deps>...])
# Discovers the suite's tasks through the dist polytest at build time and registers one ctest entry per task.
function(polyregion_add_polytest_suite name)
  cmake_parse_arguments(ARG "" "SUITE" "DEPENDS" ${ARGN})
  if(NOT ARG_SUITE)
    message(FATAL_ERROR "polyregion_add_polytest_suite: SUITE required")
  endif()
  set(_dir "${CMAKE_CURRENT_BINARY_DIR}/polytest-discover")
  set(_ids "${_dir}/${name}.ids")
  set(_tests "${_dir}/${name}-tests.cmake")
  set(_polytest "$<TARGET_FILE:Polyregion::polytest>")
  add_custom_command(
      OUTPUT "${_ids}" "${_tests}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_dir}"
      COMMAND ${CMAKE_COMMAND} -E env "POLYTEST_SUITE=${ARG_SUITE}" "${_polytest}" --list-ids > "${_ids}"
      COMMAND ${CMAKE_COMMAND} -E env "POLYTEST_SUITE=${ARG_SUITE}" "${_polytest}"
              --emit-ctest "${_tests}" --emit-prefix "${name}" --emit-binary "${_polytest}"
              --emit-env "POLYTEST_SUITE=${ARG_SUITE}"
      DEPENDS "${ARG_SUITE}" "${_polytest}" ${ARG_DEPENDS}
      COMMENT "Discovering ${name} tasks"
      VERBATIM)
  add_custom_target(${name}-discover ALL DEPENDS "${_ids}" "${_tests}")
  set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_tests}")
endfunction()
