# Pin clang's GCC discovery to the sysroot. Without this, host gcc may bring in
# symbols newer than the sysroot's libstdc++/glibc supports.
if (DEFINED ENV{CMAKE_SYSROOT} AND NOT "$ENV{CMAKE_SYSROOT}" STREQUAL "/")
    set(_sysroot "$ENV{CMAKE_SYSROOT}")
    set(POLYREGION_SYSROOT_GCC_INSTALL_DIR "")
    # Prefer gcc-toolset-N over stock, then highest version within each set.
    file(GLOB _toolset "${_sysroot}/opt/rh/gcc-toolset-*/root/usr/lib/gcc/*-redhat-linux*/[0-9]*")
    file(GLOB _stock
            "${_sysroot}/usr/lib/gcc/*-redhat-linux*/[0-9]*"
            "${_sysroot}/usr/lib/gcc/*-linux-gnu/[0-9]*"
            "${_sysroot}/usr/lib/gcc/*-linux-gnueabi*/[0-9]*")
    list(SORT _toolset ORDER DESCENDING)
    list(SORT _stock ORDER DESCENDING)
    foreach (_d ${_toolset} ${_stock})
        if (EXISTS "${_d}/libstdc++.so" OR EXISTS "${_d}/libstdc++.a")
            set(POLYREGION_SYSROOT_GCC_INSTALL_DIR "${_d}")
            break()
        endif ()
    endforeach ()
    if (POLYREGION_SYSROOT_GCC_INSTALL_DIR)
        message(STATUS "Sysroot gcc-install-dir = ${POLYREGION_SYSROOT_GCC_INSTALL_DIR}")
    endif ()
endif ()
