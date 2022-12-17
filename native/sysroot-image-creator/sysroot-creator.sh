# Copyright 2014 The Chromium Authors
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# This script should not be run directly but sourced by the other
# scripts (e.g. sysroot-creator-bullseye.sh).  Its up to the parent scripts
# to define certain environment variables: e.g.
#  DISTRO=debian
#  DIST=bullseye
#  # Similar in syntax to /etc/apt/sources.list
#  APT_SOURCES_LIST=( "http://ftp.us.debian.org/debian/ bullseye main" )
#  KEYRING_FILE=debian-archive-bullseye-stable.gpg
#  DEBIAN_PACKAGES="gcc libz libssl"

#@ This script builds Debian/Ubuntu sysroot images for building Google Chrome.
#@
#@  Generally this script is invoked as:
#@  sysroot-creator-<flavour>.sh <mode> <args>*
#@  Available modes are shown below.
#@
#@ List of modes:

######################################################################
# Config
######################################################################

set -o nounset
set -o errexit

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ -z "${DIST:-}" ]; then
  echo "error: DIST not defined"
  exit 1
fi

if [ -z "${KEYRING_FILE:-}" ]; then
  echo "error: KEYRING_FILE not defined"
  exit 1
fi

if [ -z "${DEBIAN_PACKAGES:-}" ]; then
  echo "error: DEBIAN_PACKAGES not defined"
  exit 1
fi

readonly REQUIRED_TOOLS="curl xzcat dpkg"

######################################################################
# Package Config
######################################################################

readonly PACKAGES_EXT=xz
readonly RELEASE_FILE="Release"
readonly RELEASE_FILE_GPG="Release.gpg"

readonly DEBIAN_DEP_LIST_X86_64="generated_package_lists/${DIST}.x86_64"
readonly DEBIAN_DEP_LIST_I386="generated_package_lists/${DIST}.i386"
readonly DEBIAN_DEP_LIST_ARM="generated_package_lists/${DIST}.arm"
readonly DEBIAN_DEP_LIST_AARCH64="generated_package_lists/${DIST}.aarch64"

Banner() { echo "[$*]"; }

SubBanner() { echo " ->$*"; }

Usage() {
  grep -E "^#@" "${BASH_SOURCE[0]}" | cut --bytes=3-
}

DownloadOrCopyNonUniqueFilename() {
  # Use this function instead of DownloadOrCopy when the url uniquely
  # identifies the file, but the filename (excluding the directory)
  # does not.
  local url="$1"
  local dest="$2"

  local hash
  hash="$(echo "$url" | sha256sum | cut -d' ' -f1)"

  DownloadOrCopy "${url}" "${dest}.${hash}"
  # cp the file to prevent having to redownload it, but mv it to the
  # final location so that it's atomic.
  cp "${dest}.${hash}" "${dest}.$$"
  mv "${dest}.$$" "${dest}"
}

DownloadOrCopy() {
  if [ -f "$2" ]; then
    echo "Cache hit: $2"
    return
  fi

  HTTP=0
  echo "$1" | grep -Eqs '^https?://' && HTTP=1
  if [ "$HTTP" = "1" ]; then
    SubBanner "downloading from $1 -> $2"
    # Appending the "$$" shell pid is necessary here to prevent concurrent
    # instances of sysroot-creator.sh from trying to write to the same file.
    local temp_file="${2}.partial.$$"
    # curl --retry doesn't retry when the page gives a 4XX error, so we need to
    # manually rerun.
    for i in {1..10}; do
      # --create-dirs is added in case there are slashes in the filename, as can
      # happen with the "debian/security" release class.
      local http_code
      http_code=$(curl -L "$1" --create-dirs -o "${temp_file}" \
        -w "%{http_code}")
      if [ "${http_code}" -eq 200 ]; then
        break
      fi
      echo "Bad HTTP code ${http_code} when downloading $1"
      rm -f "${temp_file}"
      sleep "$i"
    done
    if [ ! -f "${temp_file}" ]; then
      exit 1
    fi
    mv "${temp_file}" "$2"
  else
    SubBanner "copying from $1"
    cp "$1" "$2"
  fi
}

ClearInstallDir() {
  Banner "Clearing dirs in ${INSTALL_ROOT}"
  rm -rf "${INSTALL_ROOT:?}"/*
}

# some sanity checks to make sure this script is run from the right place
# with the right tools
SanityCheck() {
  Banner "Sanity Checks"

  BUILD_DIR="${SCRIPT_DIR}/out/${DIST}"
  mkdir -p "${BUILD_DIR}"
  echo "Using build directory: ${BUILD_DIR}"

  for tool in ${REQUIRED_TOOLS}; do
    if ! which "${tool}" >/dev/null; then
      echo "Required binary $tool not found."
      echo "Exiting." && exit 1
    fi
  done

  # This is where the staging sysroot is.
  INSTALL_ROOT="${BUILD_DIR}/${DIST}_${ARCH_LOWER}_staging"
  TARBALL="${BUILD_DIR}/${DISTRO}_${DIST}_${ARCH_LOWER}_sysroot.tar.xz"

  if ! mkdir -p "${INSTALL_ROOT}"; then
    echo "ERROR: ${INSTALL_ROOT} can't be created." && exit 1
  fi
}

CreateTarBall() {
  Banner "Creating tarball ${TARBALL}"
  tar -I "xz -2 -T0" -cf ${TARBALL} -C ${INSTALL_ROOT} .
}

ExtractPackageXz() {
  local src_file="$1"
  local dst_file="$2"
  local repo="$3"
  xzcat "${src_file}" | grep -E '^(Package:|Filename:|SHA256:) ' |
    sed "s|Filename: |Filename: ${repo}|" >"${dst_file}"
}
GeneratePackageListDistRepo() {
  local arch="$1"
  local repo="$2"
  local dist="$3"
  local repo_name="$4"

  local tmp_package_list="${BUILD_DIR}/Packages.${dist}_${repo_name}_${arch}"
  local repo_basedir="${repo}/dists/${dist}"
  local package_list="${BUILD_DIR}/Packages.${dist}_${repo_name}_${arch}.${PACKAGES_EXT}"
  local package_file_arch="${repo_name}/binary-${arch}/Packages.${PACKAGES_EXT}"
  local package_list_arch="${repo_basedir}/${package_file_arch}"

  DownloadOrCopyNonUniqueFilename "${package_list_arch}" "${package_list}"
  VerifyPackageListing "${package_file_arch}" "${package_list}" ${repo} ${dist}
  ExtractPackageXz "${package_list}" "${tmp_package_list}" ${repo}
  ./merge-package-lists.py "${list_base}" <"${tmp_package_list}"
}

GeneratePackageListDist() {
  local arch="$1"
  set -- $2
  local repo="$1"
  local dist="$2"
  shift 2
  while (("$#")); do
    GeneratePackageListDistRepo "$arch" "$repo" "$dist" "$1"
    shift
  done
}

GeneratePackageListCommon() {
  local output_file="$1"
  local arch="$2"
  local packages="$3"

  local list_base="${BUILD_DIR}/Packages.${DIST}_${arch}"
  >"${list_base}" # Create (or truncate) a zero-length file.
  printf '%s\n' "${APT_SOURCES_LIST[@]}" | while read source; do
    GeneratePackageListDist "${arch}" "${source}"
  done

  GeneratePackageList "${list_base}" "${output_file}" "${packages}"
}

StripChecksumsFromPackageList() {
  local package_file="$1"
  sed -i 's/ [a-f0-9]\{64\}$//' "$package_file"
}

######################################################################
#
######################################################################

HacksAndPatchesCommon() {
  local arch=$1
  local os=$2
  Banner "Misc Hacks & Patches"

  # fcntl64() was introduced in glibc 2.28.  Make sure to use fcntl() instead.
  local fcntl_h="${INSTALL_ROOT}/usr/include/fcntl.h"
  sed -i '{N; s/#ifndef __USE_FILE_OFFSET64\(\nextern int fcntl\)/#if 1\1/}' \
    "${fcntl_h}"

  # __GLIBC_MINOR__ is used as a feature test macro.  Replace it with the
  # earliest supported version of glibc (2.17, https://crbug.com/376567).
  local usr_include="${INSTALL_ROOT}/usr/include"
  local features_h="${usr_include}/features.h"
  sed -i 's|\(#define\s\+__GLIBC_MINOR__\)|\1 17 //|' "${features_h}"
  # Do not use pthread_cond_clockwait as it was introduced in glibc 2.30.
  local cppconfig_h="${usr_include}/${arch}-${os}/c++/10/bits/c++config.h"
  sed -i 's|\(#define\s\+_GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT\)|// \1|' \
    "${cppconfig_h}"

}

ReversionGlibc() {
  local arch=$1
  local os=$2
  # Avoid requiring unsupported glibc versions.
  "${SCRIPT_DIR}/reversion_glibc.py" "${INSTALL_ROOT}/lib/${arch}-${os}/libc.so.6"
  "${SCRIPT_DIR}/reversion_glibc.py" "${INSTALL_ROOT}/lib/${arch}-${os}/libm.so.6"
}

InstallIntoSysroot() {
  Banner "Install Libs And Headers Into Jail"

  mkdir -p ${BUILD_DIR}/debian-packages
  # The /debian directory is an implementation detail that's used to cd into
  # when running dpkg-shlibdeps.
  mkdir -p ${INSTALL_ROOT}/debian
  # An empty control file is necessary to run dpkg-shlibdeps.
  touch ${INSTALL_ROOT}/debian/control
  while (("$#")); do
    local file="$1"
    local package="${BUILD_DIR}/debian-packages/${file##*/}"
    shift
    local sha256sum="$1"
    shift
    if [ "${#sha256sum}" -ne "64" ]; then
      echo "Bad sha256sum from package list"
      exit 1
    fi

    Banner "Installing $(basename ${file})"
    DownloadOrCopy ${file} ${package}
    if [ ! -s "${package}" ]; then
      echo
      echo "ERROR: bad package ${package}"
      exit 1
    fi
    echo "${sha256sum}  ${package}" | sha256sum --quiet -c

    SubBanner "Extracting to ${INSTALL_ROOT}"
    dpkg-deb -x ${package} ${INSTALL_ROOT}

    base_package=$(dpkg-deb --field ${package} Package)
    mkdir -p ${INSTALL_ROOT}/debian/${base_package}/DEBIAN
    dpkg-deb -e ${package} ${INSTALL_ROOT}/debian/${base_package}/DEBIAN
  done

  ls -d ${INSTALL_ROOT}/usr/share/* |
    grep -v "/\(pkgconfig\)$" | xargs rm -r

}

CleanupJailSymlinks() {
  Banner "Jail symlink cleanup"

  SAVEDPWD=$(pwd)
  cd ${INSTALL_ROOT}
  local libdirs="lib usr/lib"

  find $libdirs -type l -printf '%p %l\n' | while read link target; do
    # skip links with non-absolute paths
    echo "${target}" | grep -qs ^/ || continue
    echo "${link}: ${target}"
    # Relativize the symlink.
    prefix=$(echo "${link}" | sed -e 's/[^/]//g' | sed -e 's|/|../|g')
    ln -snfv "${prefix}${target}" "${link}"
  done

  failed=0
  while read link target; do
    # Make sure we catch new bad links.
    if [ ! -r "${link}" ]; then
      echo "ERROR: FOUND BAD LINK ${link}"
      ls -l ${link}
      failed=1
    fi
  done < <(find $libdirs -type l -printf '%p %l\n')
  if [ $failed -eq 1 ]; then
    exit 1
  fi
  cd "$SAVEDPWD"
}

VerifyLibraryDepsCommon() {
  local arch=$1
  local os=$2
  local find_dirs=(
    "${INSTALL_ROOT}/lib/"
    "${INSTALL_ROOT}/lib/${arch}-${os}/"
    "${INSTALL_ROOT}/usr/lib/${arch}-${os}/"
  )
  local needed_libs="$(
    find ${find_dirs[*]} -name "*\.so*" -type f -exec file {} \; |
      grep ': ELF' | sed 's/^\(.*\): .*$/\1/' | xargs readelf -d |
      grep NEEDED | sort | uniq | sed 's/^.*Shared library: \[\(.*\)\]$/\1/g'
  )"
  local all_libs="$(find ${find_dirs[*]} -printf '%f\n')"
  local missing_libs="$(grep -vFxf <(echo "${all_libs}") \
    <(echo "${needed_libs}"))"
  if [ ! -z "${missing_libs}" ]; then
    echo "Missing libraries:"
    echo "${missing_libs}"
    exit 1
  fi
}

#
# CheckForDebianGPGKeyring
#
#     Make sure the Debian GPG keys exist. Otherwise print a helpful message.
#
CheckForDebianGPGKeyring() {
  if [ ! -e "$KEYRING_FILE" ]; then
    echo "KEYRING_FILE not found: ${KEYRING_FILE}"
    echo "Debian GPG keys missing. Install the debian-archive-keyring package."
    exit 1
  fi
}

#
# VerifyPackageListing
#
#     Verifies the downloaded Packages.xz file has the right checksums.
#
VerifyPackageListing() {
  local file_path="$1"
  local output_file="$2"
  local repo="$3"
  local dist="$4"

  local repo_basedir="${repo}/dists/${dist}"
  local release_list="${repo_basedir}/${RELEASE_FILE}"
  local release_list_gpg="${repo_basedir}/${RELEASE_FILE_GPG}"

  local release_file="${BUILD_DIR}/${dist}-${RELEASE_FILE}"
  local release_file_gpg="${BUILD_DIR}/${dist}-${RELEASE_FILE_GPG}"

  CheckForDebianGPGKeyring

  DownloadOrCopyNonUniqueFilename ${release_list} ${release_file}
  DownloadOrCopyNonUniqueFilename ${release_list_gpg} ${release_file_gpg}
  echo "Verifying: ${release_file} with ${release_file_gpg}"
  set -x
  gpgv --keyring "${KEYRING_FILE}" "${release_file_gpg}" "${release_file}"
  set +x

  echo "Verifying: ${output_file}"
  local sha256sum
  sha256sum=$(grep -E "${file_path}\$|:\$" "${release_file}" | grep "SHA256:" -A 1 | xargs echo | awk '{print $2;}')
  if [ "${#sha256sum}" -ne "64" ]; then
    echo "Bad sha256sum from ${release_list}" && exit 1
  fi
  echo "${sha256sum}  ${output_file}" | sha256sum --quiet -c
}

#
# GeneratePackageList
#
#     Looks up package names in ${BUILD_DIR}/Packages and write list of URLs
#     to output file.
#
GeneratePackageList() {
  local input_file="$1"
  local output_file="$2"
  echo "Updating: ${output_file} from ${input_file}"
  /bin/rm -f "${output_file}"
  shift
  shift
  local failed=0
  for pkg in $@; do
    local pkg_full=$(grep -A 1 " ${pkg}\$" "$input_file" |
      egrep "pool/.*" | sed 's/.*Filename: //')
    if [ -z "${pkg_full}" ]; then
      echo "ERROR: missing package: $pkg"
      local failed=1
    else
      local sha256sum=$(grep -A 4 " ${pkg}\$" "$input_file" |
        grep ^SHA256: | sed 's/^SHA256: //')
      if [ "${#sha256sum}" -ne "64" ]; then
        echo "Bad sha256sum from Packages"
        local failed=1
      fi
      echo $pkg_full $sha256sum >>"$output_file"
    fi
  done
  if [ $failed -eq 1 ]; then exit 1; fi
  # sort -o does an in-place sort of this file
  sort "$output_file" -o "$output_file"
}

cd "${SCRIPT_DIR}"

case "$1" in
x86_64)
  ARCH_LOWER=x86_64
  SanityCheck
  ClearInstallDir

  package_file="${DEBIAN_DEP_LIST_X86_64}"

  GeneratePackageListCommon "$package_file" amd64 "${DEBIAN_PACKAGES}
        ${DEBIAN_PACKAGES_X86:=} ${DEBIAN_PACKAGES_X86_64:=}"

  files_and_sha256sums="$(cat ${package_file})"
  StripChecksumsFromPackageList "$package_file"
  InstallIntoSysroot ${files_and_sha256sums}
  HacksAndPatchesCommon x86_64 linux-gnu
  ReversionGlibc x86_64 linux-gnu
  CleanupJailSymlinks
  VerifyLibraryDepsCommon x86_64 linux-gnu
  CreateTarBall
  ;;
i386)
  ARCH_LOWER=i386
  SanityCheck
  ClearInstallDir

  package_file="${DEBIAN_DEP_LIST_I386}"

  GeneratePackageListCommon "$package_file" i386 "${DEBIAN_PACKAGES}
        ${DEBIAN_PACKAGES_X86:=}"

  files_and_sha256sums="$(cat ${package_file})"
  StripChecksumsFromPackageList "$package_file"
  InstallIntoSysroot ${files_and_sha256sums}
  HacksAndPatchesCommon i386 linux-gnu
  ReversionGlibc i386 linux-gnu
  CleanupJailSymlinks
  VerifyLibraryDepsCommon i386 linux-gnu
  CreateTarBall
  ;;
arm)
  ARCH_LOWER=arm
  SanityCheck
  ClearInstallDir

  package_file="${DEBIAN_DEP_LIST_ARM}"

  GeneratePackageListCommon "$package_file" armhf "${DEBIAN_PACKAGES}
            ${DEBIAN_PACKAGES_ARM:=}"

  files_and_sha256sums="$(cat ${package_file})"
  StripChecksumsFromPackageList "$package_file"
  InstallIntoSysroot ${files_and_sha256sums}
  HacksAndPatchesCommon arm linux-gnueabihf
  ReversionGlibc arm linux-gnueabihf
  CleanupJailSymlinks
  VerifyLibraryDepsCommon arm linux-gnueabihf
  CreateTarBall
  ;;
aarch64)
  ARCH_LOWER=aarch64
  SanityCheck
  ClearInstallDir

  package_file="${DEBIAN_DEP_LIST_AARCH64}"

  GeneratePackageListCommon "$package_file" arm64 "${DEBIAN_PACKAGES}
              ${DEBIAN_PACKAGES_AARCH64:=}"

  files_and_sha256sums="$(cat ${package_file})"
  StripChecksumsFromPackageList "$package_file"
  InstallIntoSysroot ${files_and_sha256sums}
  # Use the unstripped libdbus for arm64 to prevent linker errors.
  # https://bugs.chromium.org/p/webrtc/issues/detail?id=8535
  HacksAndPatchesCommon aarch64 linux-gnu
  # Skip reversion_glibc.py. Glibc is compiled in a way where many libm math
  # functions do not have compatibility symbols for versions <= 2.17.
  ReversionGlibc aarch64 linux-gnu
  CleanupJailSymlinks
  VerifyLibraryDepsCommon aarch64 linux-gnu
  CreateTarBall
  ;;
*) echo "ERROR: unknown arch $2" && exit 1 ;;
esac
