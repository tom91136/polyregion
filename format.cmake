# Usage:
#   cmake -P format.cmake          # format in place
#   cmake -P format.cmake check    # CI gate (non-zero exit on diff)

cmake_minimum_required(VERSION 3.20)

set(REPO_ROOT ${CMAKE_CURRENT_LIST_DIR})

set(NATIVE_TARGET format)
set(SBT_TASKS scalafmtAll scalafmtSbt)
if ("${CMAKE_ARGV3}" STREQUAL "check")
    set(NATIVE_TARGET format-check)
    set(SBT_TASKS scalafmtCheckAll scalafmtSbtCheck)
endif ()

if (NOT POLYREGION_NATIVE_BUILD)
    file(GLOB _candidates "${REPO_ROOT}/native/cmake-build-*" "${REPO_ROOT}/native/build-*")
    set(_newest_mtime 0)
    foreach (c ${_candidates})
        if (EXISTS "${c}/CMakeCache.txt" AND EXISTS "${c}/build.ninja" AND EXISTS "${c}/CMakeFiles/rules.ninja")
            file(TIMESTAMP "${c}/CMakeCache.txt" _ts "%s")
            if (_ts GREATER _newest_mtime)
                set(_newest_mtime ${_ts})
                set(POLYREGION_NATIVE_BUILD ${c})
            endif ()
        endif ()
    endforeach ()
endif ()
if (NOT POLYREGION_NATIVE_BUILD)
    message(FATAL_ERROR
            "No configured native build dir under native/{cmake-build,build}-* with usable ninja "
            "files - configure one first, or pass -DPOLYREGION_NATIVE_BUILD=<path>.")
endif ()

### native

message(STATUS "Native: ${NATIVE_TARGET} via ${POLYREGION_NATIVE_BUILD}")
execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${POLYREGION_NATIVE_BUILD} --target ${NATIVE_TARGET}
        RESULT_VARIABLE native_rc)

### scala

find_program(SBT NAMES sbt)
set(sbt_rc 0)
if (SBT)
    message(STATUS "Scala: sbt ${SBT_TASKS}")
    execute_process(
            COMMAND ${SBT} ${SBT_TASKS}
            WORKING_DIRECTORY ${REPO_ROOT}/frontend
            RESULT_VARIABLE sbt_rc)
else ()
    message(WARNING "sbt not found on PATH — skipping Scala format")
endif ()

if (NOT native_rc EQUAL 0 OR NOT sbt_rc EQUAL 0)
    message(FATAL_ERROR "Format ${NATIVE_TARGET} failed (native=${native_rc} sbt=${sbt_rc})")
endif ()
