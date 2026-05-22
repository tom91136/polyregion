# Usage:
#   cmake -P lint-ci.cmake          # lint workflows, auto-fetch actionlint if missing

cmake_minimum_required(VERSION 3.20)

set(REPO_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(ACTIONLINT_VERSION 1.7.7)
set(ACTIONLINT_DIR ${REPO_ROOT}/.cache/actionlint)
set(ACTIONLINT ${ACTIONLINT_DIR}/actionlint)

if (NOT EXISTS ${ACTIONLINT})
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        set(_os linux)
    elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        set(_os darwin)
    elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        set(_os windows)
        set(ACTIONLINT ${ACTIONLINT_DIR}/actionlint.exe)
    else ()
        message(FATAL_ERROR "Unsupported host: ${CMAKE_HOST_SYSTEM_NAME}")
    endif ()
    execute_process(COMMAND uname -m OUTPUT_VARIABLE _machine OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (_machine STREQUAL "x86_64" OR _machine STREQUAL "AMD64")
        set(_arch amd64)
    elseif (_machine STREQUAL "aarch64" OR _machine STREQUAL "arm64")
        set(_arch arm64)
    else ()
        message(FATAL_ERROR "Unsupported arch: ${_machine}")
    endif ()
    set(_ext tar.gz)
    if (_os STREQUAL "windows")
        set(_ext zip)
    endif ()
    set(_url "https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/actionlint_${ACTIONLINT_VERSION}_${_os}_${_arch}.${_ext}")
    message(STATUS "Fetching actionlint ${ACTIONLINT_VERSION} from ${_url}")
    file(MAKE_DIRECTORY ${ACTIONLINT_DIR})
    file(DOWNLOAD ${_url} ${ACTIONLINT_DIR}/actionlint.${_ext} STATUS _dl_status)
    list(GET _dl_status 0 _dl_rc)
    if (NOT _dl_rc EQUAL 0)
        message(FATAL_ERROR "Download failed: ${_dl_status}")
    endif ()
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf ${ACTIONLINT_DIR}/actionlint.${_ext}
            WORKING_DIRECTORY ${ACTIONLINT_DIR})
endif ()

file(GLOB _workflows ${REPO_ROOT}/.github/workflows/*.yaml ${REPO_ROOT}/.github/workflows/*.yml)
execute_process(
        COMMAND ${ACTIONLINT} -color ${_workflows}
        RESULT_VARIABLE _rc)
if (NOT _rc EQUAL 0)
    message(FATAL_ERROR "actionlint reported issues (rc=${_rc})")
endif ()
message(STATUS "actionlint clean")
