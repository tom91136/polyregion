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
