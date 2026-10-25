vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO bdwgc/bdwgc
    REF "v${VERSION}"
    SHA512 514824d4171374bb0f3211c79732dd84cb3af44f515b2fb51b7fa339a5194ba653cc93f1d56d9da202c959f2f9ee451f0d0c67da42d0b0531a9c749da91a30a0
    HEAD_REF master
)

set(BDWGC_CFLAGS_EXTRA "-I${CURRENT_INSTALLED_DIR}/include") # for libatomic_ops
if(VCPKG_TARGET_IS_WINDOWS)
    string(APPEND BDWGC_CFLAGS_EXTRA " -DMAX_HEAP_SECTS=16384 -DNO_MSGBOX_ON_ERROR")
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -Denable_cplusplus=ON
        -Denable_docs=OFF
        "-DCFLAGS_EXTRA=${BDWGC_CFLAGS_EXTRA}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/bdwgc)
vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_fixup_pkgconfig()

file(INSTALL "${SOURCE_PATH}/README.QUICK" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)
