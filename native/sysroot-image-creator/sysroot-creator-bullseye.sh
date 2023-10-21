#!/bin/bash
# Copyright 2022 The Chromium Authors
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DISTRO=debian
DIST=bullseye

# This number is appended to the sysroot key to cause full rebuilds.  It
# should be incremented when removing packages or patching existing packages.
# It should not be incremented when adding packages.
SYSROOT_RELEASE=0

ARCHIVE_TIMESTAMP=20221215T151605Z
ARCHIVE_URL="https://snapshot.debian.org/archive/debian/$ARCHIVE_TIMESTAMP/"
APT_SOURCES_LIST=(
  # This mimicks a sources.list from bullseye.
  "${ARCHIVE_URL} bullseye main"
)

# gpg keyring file generated using generate_keyring.sh
KEYRING_FILE="${SCRIPT_DIR}/keyring.gpg"

# Sysroot packages: these are the packages needed to build chrome.
DEBIAN_PACKAGES="\
libatomic1
libc6
libc6-dev
 
libgcc-s1
libgcc-10-dev
libgomp1
 
libstdc++6
libstdc++-10-dev

linux-libc-dev
libpthread-stubs0-dev

libcrypt-dev
libcrypt1
"
DEBIAN_PACKAGES_X86_64="\
libtsan0
liblsan0
libc6-dev-i386
"
DEBIAN_PACKAGES_X86="\
libasan6
libitm1
libquadmath0
libubsan1
"
DEBIAN_PACKAGES_ARM="\
libasan6
libubsan1
"
DEBIAN_PACKAGES_AARCH64="\
libasan6
libitm1
liblsan0
libtsan0
libubsan1
"

. "${SCRIPT_DIR}/sysroot-creator.sh"
