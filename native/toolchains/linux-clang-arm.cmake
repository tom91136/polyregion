set(POLYREGION_SYSTEM_PROCESSOR arm)
set(POLYREGION_HOST_MATCH_REGEX "^arm")
set(POLYREGION_TARGET_TRIPLE    arm-linux-gnueabihf)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/linux-clang-cross-common.cmake")
# VCPKG_CHAINLOAD_TOOLCHAIN_FILE must point at THIS file so vcpkg re-enters here per-port.
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_FILE}")
