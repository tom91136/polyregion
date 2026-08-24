if(NOT DEFINED FIXTURE OR NOT DEFINED EMIT OR NOT DEFINED WORK)
    message(FATAL_ERROR "FIXTURE, EMIT and WORK are required")
endif()
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")
execute_process(
        COMMAND "${FIXTURE}" --write-package-inputs "${WORK}/inputs"
        RESULT_VARIABLE fixture_result
        OUTPUT_VARIABLE fixture_out
        ERROR_VARIABLE fixture_error)
if(NOT fixture_result EQUAL 0)
    message(FATAL_ERROR "fixture failed (${fixture_result}): ${fixture_out}${fixture_error}")
endif()

execute_process(
        COMMAND "${EMIT}" "${WORK}/inputs/interface.polyast" "${WORK}/packages" "${WORK}/inputs/program.polyast"
        RESULT_VARIABLE emit_result
        OUTPUT_VARIABLE emit_out
        ERROR_VARIABLE emit_error)
if(NOT emit_result EQUAL 0)
    message(FATAL_ERROR "polypackage-emit failed (${emit_result}): ${emit_out}${emit_error}")
endif()
if(NOT EXISTS "${WORK}/packages/foo/lib.polyast")
    message(FATAL_ERROR "polypackage-emit did not atomically publish foo/lib.polyast")
endif()
