set(POLYREGION_SYSTEM_PROCESSOR ppc64le)
set(POLYREGION_HOST_MATCH_REGEX "^ppc64le$")
set(POLYREGION_TARGET_TRIPLE    powerpc64le-linux-gnu)
# clang defaults to IEEE 128-bit long double on ppc64le, which pulls in __*ieee128 libm symbols
# (nexttowardieee128 etc.) not shipped by pre-2.32 glibc (AL8 = 2.28).
set(POLYREGION_EXTRA_CFLAGS "-mabi=ibmlongdouble")
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/linux-clang-cross-common.cmake")
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_FILE}")
