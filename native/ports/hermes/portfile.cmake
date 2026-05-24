set(HERMES_REF "hermes-v250829098.0.13")

vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO facebook/hermes
    REF ${HERMES_REF}
    SHA512 f50d0a574f7aa5b511e7a06c16ac59f0e36d4b6b957fed36a0b5c9b6fc2b6ca5313db1dce824f34de35189a2e7c046d014bfdfa995d9d4f84d7c7b0a1b39c430
    HEAD_REF main
    PATCHES
        boost-context-asm-marmasm.patch
        msvc-mp-cxx-only.patch
)

# Stub out llvh's PerfectShuffle utility: it registers as a real target and
# collides with LLVM's BuildTreeOnlyTargets export set.
file(WRITE "${SOURCE_PATH}/external/llvh/utils/PerfectShuffle/CMakeLists.txt" "")

# boost_context keys its arch/ABI on CMAKE_SYSTEM_PROCESSOR. Some triplet
# toolchains leave it empty, defaulting boost to x86_64+sysv even on arm.
# Derive both from VCPKG_TARGET_ARCHITECTURE directly.
if (VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    set(_boost_arch arm64)
    set(_boost_abi  aapcs)
elseif (VCPKG_TARGET_ARCHITECTURE STREQUAL "arm")
    set(_boost_arch arm)
    set(_boost_abi  aapcs)
elseif (VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    set(_boost_arch x86_64)
    if (VCPKG_TARGET_IS_WINDOWS)
        set(_boost_abi ms)
    else ()
        set(_boost_abi sysv)
    endif ()
elseif (VCPKG_TARGET_ARCHITECTURE STREQUAL "x86")
    set(_boost_arch i386)
    if (VCPKG_TARGET_IS_WINDOWS)
        set(_boost_abi ms)
    else ()
        set(_boost_abi sysv)
    endif ()
else ()
    message(FATAL_ERROR "hermes port: unhandled VCPKG_TARGET_ARCHITECTURE='${VCPKG_TARGET_ARCHITECTURE}'")
endif ()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DHERMES_ENABLE_TOOLS=ON
        -DHERMES_ENABLE_CONTRIB_EXTENSIONS=OFF
        -DHERMES_ENABLE_TEST_SUITE=OFF
        -DHERMES_ENABLE_DEBUGGER=OFF
        -DHERMES_BUILD_APPLE_FRAMEWORK=OFF
        -DHERMES_UNICODE_LITE=ON
        -DHERMES_BUILD_SHARED_JSI=OFF
        -DBOOST_CONTEXT_ARCHITECTURE=${_boost_arch}
        -DBOOST_CONTEXT_ABI=${_boost_abi}
)

vcpkg_cmake_build(TARGET hermesvm_a LOGFILE_BASE build-hermesvm_a)
vcpkg_cmake_build(TARGET jsi        LOGFILE_BASE build-jsi)
vcpkg_cmake_build(TARGET compileJS  LOGFILE_BASE build-compileJS)

file(INSTALL "${SOURCE_PATH}/API/jsi/jsi"           DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/API/hermes"            DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/public/hermes"         DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/include/hermes"        DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# hermesvm_a transitively needs every static archive under the build tree.
set(_lib_pfx "${CMAKE_STATIC_LIBRARY_PREFIX}")
set(_lib_ext "${CMAKE_STATIC_LIBRARY_SUFFIX}")
file(GLOB_RECURSE _archives
    "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/${_lib_pfx}*${_lib_ext}")
foreach (_a IN LISTS _archives)
    file(INSTALL "${_a}" DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
endforeach ()

file(GLOB_RECURSE _archives_dbg
    "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-dbg/${_lib_pfx}*${_lib_ext}")
foreach (_a IN LISTS _archives_dbg)
    file(INSTALL "${_a}" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib")
endforeach ()

configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/hermes-config.cmake.in"
    "${CURRENT_PACKAGES_DIR}/share/${PORT}/hermes-config.cmake"
    @ONLY)

file(INSTALL "${SOURCE_PATH}/LICENSE"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    RENAME copyright)

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
