function(polyregion_apply_common_options target)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "DYNAMIC_CXX_RUNTIME" "" "EXTRA_LINK")
  set(_link_options ${LINK_OPTIONS})
  if (ARG_DYNAMIC_CXX_RUNTIME AND UNIX AND NOT APPLE)
    # A DSO loaded into an arbitrary C++ process must use that process's C++ and
    # exception runtimes. Statically embedding either creates duplicate locale /
    # exception state and can mix libgcc's static unwinder with libgcc_s personalities.
    list(REMOVE_ITEM _link_options
            -static-libstdc++
            -static-libgcc
            "LINKER:--exclude-libs=libstdc++.a")
  endif ()
  target_compile_options(${target} PRIVATE ${COMPILE_OPTIONS})
  target_link_options(${target} PRIVATE ${_link_options} ${ARG_EXTRA_LINK})
endfunction()

# XXX macOS asan cannot be dlopen'd ("interceptors loaded too late"), so code loaded into the
# uninstrumented dist clang/flang (the plugins + the PolyAST they embed) must carry no __asan_*;
# strip asan there. Appended last so it wins over the toolchain's global -fsanitize.
function(polyregion_strip_asan_on_darwin target)
  if (APPLE AND POLYREGION_ASAN)
    target_compile_options(${target} PRIVATE -fno-sanitize=address,undefined)
    target_link_options(${target} PRIVATE -fno-sanitize=address,undefined)
  endif ()
endfunction()

function(polyregion_test_profile_dir target)
  target_compile_definitions(${target} PRIVATE POLYREGION_TEST_PROFILE_DIR="${CMAKE_SOURCE_DIR}/test-profiles")
endfunction()

function(polyregion_stage_post_build)
  foreach (lib ${ARGN})
    foreach (driver IN ITEMS polycpp polyfc)
      if (lib IN_LIST POLYREGION_STAGED_LIBS_${driver})
        set(_dest "$<TARGET_FILE_DIR:${driver}>/${POLYREGION_STAGE_DIR_${driver}}/")
        add_custom_command(TARGET ${lib} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_dest}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${lib}>" "${_dest}"
                VERBATIM)
        get_target_property(_t ${lib} TYPE)
        if (WIN32 AND _t STREQUAL "SHARED_LIBRARY")
          add_custom_command(TARGET ${lib} POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_LINKER_FILE:${lib}>" "${_dest}"
                  VERBATIM)
        endif ()
      endif ()
    endforeach ()
  endforeach ()
endfunction()

function(polyregion_copy_lib_targets out_target driver lib_dir)
  set(keep "")
  set(commands "")
  foreach (lib ${ARGN})
    list(APPEND keep "$<TARGET_FILE_NAME:${lib}>")
    list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${lib}>" "$<TARGET_FILE_DIR:${driver}>/${lib_dir}/")
    get_target_property(_t ${lib} TYPE)
    if (WIN32 AND _t STREQUAL "SHARED_LIBRARY")
      list(APPEND keep "$<TARGET_LINKER_FILE_NAME:${lib}>")
      list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_LINKER_FILE:${lib}>" "$<TARGET_FILE_DIR:${driver}>/${lib_dir}/")
    endif ()
  endforeach ()
  add_custom_target(${out_target}
          COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${driver}>/${lib_dir}/"
          ${commands}
          COMMAND ${CMAKE_COMMAND} "-DDIR=$<TARGET_FILE_DIR:${driver}>/${lib_dir}" "-DKEEP=$<JOIN:${keep},$<SEMICOLON>>"
                  -P "${CMAKE_SOURCE_DIR}/cmake/prune_staged.cmake"
          VERBATIM)
  add_dependencies(${out_target} ${ARGN})
endfunction()
