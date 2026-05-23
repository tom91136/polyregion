# Pin clang's GCC discovery to the sysroot. Without this, host gcc may bring in
# symbols newer than the sysroot's libstdc++/glibc supports.
if (DEFINED ENV{CMAKE_SYSROOT} AND NOT "$ENV{CMAKE_SYSROOT}" STREQUAL "/")
    set(_sysroot "$ENV{CMAKE_SYSROOT}")
    set(POLYREGION_SYSROOT_GCC_INSTALL_DIR "")
    foreach (_d
            "${_sysroot}/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12"
            "${_sysroot}/usr/lib/gcc/x86_64-redhat-linux/8"
            "${_sysroot}/usr/lib/gcc/x86_64-linux-gnu/10"
            "${_sysroot}/usr/lib/gcc/x86_64-linux-gnu/12"
            "${_sysroot}/usr/lib/gcc/x86_64-linux-gnu/13")
        if (EXISTS "${_d}/libstdc++.so" OR EXISTS "${_d}/libstdc++.a")
            set(POLYREGION_SYSROOT_GCC_INSTALL_DIR "${_d}")
            break()
        endif ()
    endforeach ()
    if (POLYREGION_SYSROOT_GCC_INSTALL_DIR)
        message(STATUS "Sysroot gcc-install-dir = ${POLYREGION_SYSROOT_GCC_INSTALL_DIR}")
    endif ()
endif ()
