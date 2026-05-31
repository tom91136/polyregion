function(polyregion_apply_common_options target)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "EXTRA_LINK")
  target_compile_options(${target} PRIVATE ${COMPILE_OPTIONS})
  target_link_options(${target} PRIVATE ${LINK_OPTIONS} ${ARG_EXTRA_LINK})
endfunction()

function(polyregion_test_profile_dir target)
  target_compile_definitions(${target} PRIVATE POLYREGION_TEST_PROFILE_DIR="${CMAKE_SOURCE_DIR}/test-profiles")
endfunction()
