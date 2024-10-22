#!/usr/bin/env python3

import sys
import os
import requests
import zipfile

COMMIT = "34290ccd3eedf493b5b09497b32e7a2749e5b18c"
OWNER = "electron"
REPO = "debian-sysroot-image-creator"
URL = f"https://github.com/{OWNER}/{REPO}/archive/{COMMIT}.zip"

if not os.path.exists(REPO):
    response = requests.get(URL)
    zip_file_path = "debian-sysroot-image-creator.zip"
    with open(zip_file_path, "wb") as file:
        file.write(response.content)
    try:
        with zipfile.ZipFile(zip_file_path, "r") as ref:
            ref.extractall()
        os.rename(f"{REPO}-{COMMIT}", REPO)
        os.remove(zip_file_path)
        print("Download and extraction complete.")
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
else:
    print(f"The repository '{REPO}' already exists. Skipping download.")

sys.path.append(os.path.join(os.path.dirname(__file__), f"{REPO}/build/linux/sysroot_scripts"))
import sysroot_creator

sysroot_creator.TRIPLES = {
    "amd64": "x86_64-linux-gnu",
    "i386": "i386-linux-gnu",
    "armhf": "arm-linux-gnueabihf",
    "arm64": "aarch64-linux-gnu",
}

sysroot_creator.DEBIAN_PACKAGES = [
    "libatomic1",
    "libc6",
    "libc6-dev",

    "libgcc-s1",
    "libgcc-10-dev",
    "libgomp1",

    "libstdc++6",
    "libstdc++-10-dev",

    "linux-libc-dev",
    "libpthread-stubs0-dev",

    "libcrypt-dev",
    "libcrypt1"
]

sysroot_creator.DEBIAN_PACKAGES_ELECTRON = []

sysroot_creator.DEBIAN_PACKAGES_ARCH = {
    "amd64": [
        "libasan6",
        "libtsan0",
        "liblsan0",
        "libquadmath0",
        "libubsan1"
    ],
    "i386": [
        "libasan6",
        "libitm1",
        "libquadmath0",
        "libubsan1",
    ],
    "armhf": [
        "libasan6",
        "libubsan1"
    ],
    "arm64": [
        "libasan6",
        "libitm1",
        "liblsan0",
        "libtsan0",
        "libubsan1"
    ]
}


def hacks_and_patches(install_root: str, script_dir: str, arch: str) -> None:
    sysroot_creator.banner("Misc Hacks & Patches")

    # Include limits.h in stdlib.h to fix an ODR issue.
    stdlib_h = os.path.join(install_root, "usr", "include", "stdlib.h")
    sysroot_creator.replace_in_file(stdlib_h, r"(#include <stddef.h>)",
                                    r"\1\n#include <limits.h>")

    # Move pkgconfig scripts.
    pkgconfig_dir = os.path.join(install_root, "usr", "lib", "pkgconfig")
    os.makedirs(pkgconfig_dir, exist_ok=True)
    triple_pkgconfig_dir = os.path.join(install_root, "usr", "lib",
                                        sysroot_creator.TRIPLES[arch], "pkgconfig")
    if os.path.exists(triple_pkgconfig_dir):
        for file in os.listdir(triple_pkgconfig_dir):
            sysroot_creator.shutil.move(os.path.join(triple_pkgconfig_dir, file),
                                        pkgconfig_dir)


sysroot_creator.hacks_and_patches = hacks_and_patches

if __name__ == "__main__":
    sysroot_creator.main()
