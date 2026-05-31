function(polyregion_apply_common_options target)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "EXTRA_LINK")
  target_compile_options(${target} PRIVATE ${COMPILE_OPTIONS})
  target_link_options(${target} PRIVATE ${LINK_OPTIONS} ${ARG_EXTRA_LINK})
endfunction()

function(polyregion_test_profile_dir target)
  target_compile_definitions(${target} PRIVATE POLYREGION_TEST_PROFILE_DIR="${CMAKE_SOURCE_DIR}/test-profiles")
endfunction()

function(polyregion_copy_lib_targets out_target driver lib_dir)
  set(commands "")
  foreach (lib ${ARGN})
    list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${lib}>" "$<TARGET_FILE_DIR:${driver}>/${lib_dir}/")
    get_target_property(_t ${lib} TYPE)
    if (WIN32 AND _t STREQUAL "SHARED_LIBRARY")
      list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_LINKER_FILE:${lib}>" "$<TARGET_FILE_DIR:${driver}>/${lib_dir}/")
    endif ()
  endforeach ()
  add_custom_target(${out_target}
          COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${driver}>/${lib_dir}/"
          ${commands})
endfunction()
