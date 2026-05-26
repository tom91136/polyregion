# polyregion task runner. `just` lists recipes; `just <recipe>` runs one.

# Recipes use bash array/brace expansion. Windows-shell expects Git Bash on PATH.
set shell := ["bash", "-cu"]
set windows-shell := ["bash", "-cu"]

native_build := env_var_or_default('POLYREGION_NATIVE_BUILD', '')
arch         := env_var_or_default('POLYREGION_ARCH', env_var_or_default('ARCH', `uname -m`))
build_type   := env_var_or_default('BUILD_TYPE', 'Release')

# `just --set dylib OFF llvm` builds a static dist (no libLLVM.so / libMLIR.so / libclang-cpp.so).
dylib := env_var_or_default('POLYREGION_LLVM_DYLIB', 'ON')
export POLYREGION_LLVM_DYLIB := dylib

# Auto-export VCPKG_ROOT to ./vcpkg (populated by `just vcpkg`) if the env doesn't set it.
export VCPKG_ROOT := env_var_or_default('RUNVCPKG_VCPKG_ROOT', env_var_or_default('VCPKG_ROOT', justfile_directory() / "vcpkg"))

# Shared vcpkg install root so `just vcpkg-deps` and `just configure` agree on a location.
export VCPKG_INSTALLED_DIR := env_var_or_default('VCPKG_INSTALLED_DIR', justfile_directory() / "native" / ".vcpkg" / "vcpkg_installed")

actionlint_version  := '1.7.7'
clang_format_version := '20.1.0'
# See https://github.com/muttleyxd/clang-tools-static-binaries/releases
clang_format_release := 'master-796e77c'

# CI sets SYSROOT_PATH; locally defaults to native/sysroot-{arch}.
sysroot_path := env_var_or_default('SYSROOT_PATH', justfile_directory() / "native" / "sysroot-" + arch)

# === Default ===

# List all recipes.
default:
    @just --list

# Print the resolved build settings.
env:
    @printf '%-12s = %s\n' arch         '{{ arch }}'
    @printf '%-12s = %s\n' build_type   '{{ build_type }}'
    @printf '%-12s = %s\n' dylib        '{{ dylib }}'
    @printf '%-12s = %s\n' sysroot_path '{{ sysroot_path }}'
    @printf '%-12s = %s\n' vcpkg_root   "${VCPKG_ROOT}"
    @printf '%-12s = %s\n' java_home    "${JAVA_HOME:-(unset)}"

# === Format ===

# clang-format + scalafmt, in place, in parallel.
format:       (_format "format"       "scalafmtAll"      "scalafmtSbt")

# clang-format + scalafmt, dry-run, in parallel; non-zero exit on diff.
format-check: (_format "format-check" "scalafmtCheckAll" "scalafmtSbtCheck")

_format native_target sbt_task_a sbt_task_b:
    #!/usr/bin/env bash
    # XXX no `-e`; we collect both native + sbt rc's after `wait` and report jointly.
    set -uo pipefail
    BUILD=$(just _native-build)
    [ -z "$BUILD" ] && { echo "no configured native build dir" >&2; exit 1; }
    echo "Native:  {{ native_target }} via $BUILD"
    cmake --build "$BUILD" --target {{ native_target }} --parallel &
    pid_n=$!
    if command -v sbt >/dev/null 2>&1; then
        echo "Scala:   sbt {{ sbt_task_a }} {{ sbt_task_b }}"
        (cd frontend && sbt -no-colors {{ sbt_task_a }} {{ sbt_task_b }}) &
        pid_s=$!
    else
        echo "sbt not found on PATH - skipping Scala format" >&2
        pid_s=
    fi
    wait $pid_n; rc_n=$?
    rc_s=0
    [ -n "$pid_s" ] && { wait $pid_s; rc_s=$?; }
    if [ "$rc_n" -ne 0 ] || [ "$rc_s" -ne 0 ]; then
        echo "{{ native_target }} failed (native=$rc_n sbt=$rc_s)" >&2
        exit 1
    fi

# === Codegen ===

# Regenerate Scala-derived C++/JNI sources, then clang-format the output.
codegen:       _codegen-sbt _codegen-format

# codegen + git diff against committed state, fails if regenerated sources drift.
codegen-check: codegen _codegen-diff

_codegen-sbt:
    cd frontend && sbt -no-colors 'codegen/genCodegen'

_codegen-format:
    "$(just _clang-format)" -i native/polyast/generated/*.{h,cpp} native/bindings/jvm/generated/*.{h,cpp}

_codegen-diff:
    git diff --exit-code -- native/polyast/generated native/bindings/jvm/generated native/polyc/generated native/polyc/include/polyregion/polypass.h

# === Pass bundles ===

# Build the Scala.js pass bundle (closure-compiled) and stage at native/polyc/polypass.js.
pass-js:
    cd frontend && sbt -no-colors 'passJS/exportPassBundle'

# Build the Scala Native pass DSO and stage at native/polyc/libpolypass.{so,dylib,dll}.
# Export CMAKE_SYSROOT so sbt's nativeConfig propagates --sysroot to clang
pass-native:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d "{{ sysroot_path }}" ]; then export CMAKE_SYSROOT="{{ sysroot_path }}"; fi
    cd frontend && sbt -no-colors 'passNative/exportPassDso'

# === Lint ===

# Validate .github/workflows/*.yaml against actionlint; auto-fetches the binary on first run.
lint-ci:
    #!/usr/bin/env bash
    set -euo pipefail
    "$(just _actionlint)" -color .github/workflows/*.yaml

_actionlint:
    #!/usr/bin/env bash
    set -euo pipefail
    DIR=".cache/actionlint"
    BIN="$DIR/actionlint"
    [[ "{{ os() }}" == "windows" ]] && BIN="$BIN.exe"
    if [ ! -x "$BIN" ]; then
        mkdir -p "$DIR"
        case "{{ os() }}" in linux|macos) ext=tar.gz ;; windows) ext=zip ;; *) echo "unsupported OS: {{ os() }}" >&2; exit 1 ;; esac
        os_token="{{ os() }}"; [ "$os_token" = "macos" ] && os_token=darwin
        case "{{ arch() }}" in x86_64|amd64) a=amd64 ;; aarch64|arm64) a=arm64 ;; *) echo "unsupported arch: {{ arch() }}" >&2; exit 1 ;; esac
        url="https://github.com/rhysd/actionlint/releases/download/v{{ actionlint_version }}/actionlint_{{ actionlint_version }}_${os_token}_${a}.${ext}"
        echo "Fetching actionlint {{ actionlint_version }} from $url" >&2
        curl -fsSL "$url" -o "$DIR/actionlint.$ext"
        (cd "$DIR" && tar xf "actionlint.$ext")
    fi
    echo "$BIN"

_clang-format:
    #!/usr/bin/env bash
    set -euo pipefail
    DIR=".cache/clang-format-{{ clang_format_version }}"
    BIN="$DIR/clang-format"
    [[ "{{ os() }}" == "windows" ]] && BIN="$BIN.exe"
    if [ ! -x "$BIN" ]; then
        mkdir -p "$DIR"
        case "{{ os() }}-{{ arch() }}" in
            linux-x86_64|linux-amd64)      asset=clang-format-20_linux-amd64 ;;
            linux-aarch64|linux-arm64)     asset=clang-format-20_linux-arm64 ;;
            macos-aarch64|macos-arm64)     asset=clang-format-20_macosx-arm64 ;;
            macos-x86_64|macos-amd64)      asset=clang-format-20_macosx-amd64 ;;
            windows-x86_64|windows-amd64)  asset=clang-format-20_windows-amd64.exe ;;
            *) echo "unsupported {{ os() }}-{{ arch() }}" >&2; exit 1 ;;
        esac
        url="https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/{{ clang_format_release }}/${asset}"
        echo "Fetching clang-format {{ clang_format_version }} from $url" >&2
        curl -fsSL "$url" -o "$BIN"
        chmod +x "$BIN"
    fi
    echo "$BIN"

# === Test ===

# Run the full scalalang munit suite.
scala-tests *args='':
    cd frontend && sbt -no-colors 'compiler-testsuite-scala/test' {{ args }}

# Run the native ctest suite, excluding the `device` label (GPU dispatch), extra ctest flags via *args.
native-tests *args='':
    #!/usr/bin/env bash
    set -euo pipefail
    BUILD=$(just _native-build)
    ctest --test-dir "$BUILD" -LE device {{ args }}

# === Build wrappers ===

# Clone vcpkg into ./vcpkg at the commit pinned in native/.env. Idempotent.
vcpkg:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d vcpkg/.git ]; then git clone https://github.com/microsoft/vcpkg.git; fi
    COMMIT=$(awk -F= '/^VCPKG_COMMIT=/{print $2}' native/.env)
    [ -n "$COMMIT" ] && git -C vcpkg fetch --depth=1 origin "$COMMIT" && git -C vcpkg checkout "$COMMIT"
    echo "vcpkg ready at $PWD/vcpkg"

# Install vcpkg manifest deps to $VCPKG_INSTALLED_DIR for the current host/arch.
vcpkg-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{ os() }}-{{ arch }}" in
        linux-x86_64|linux-amd64)       TRIPLET=linux-clang-amd64 ;;
        linux-aarch64|linux-arm64)      TRIPLET=linux-clang-aarch64 ;;
        macos-x86_64|macos-amd64)       TRIPLET=darwin-clang-amd64 ;;
        macos-aarch64|macos-arm64)      TRIPLET=darwin-clang-arm64 ;;
        windows-x86_64|windows-amd64)   TRIPLET=x64-windows-static ;;
        windows-aarch64|windows-arm64)  TRIPLET=arm64-windows-static ;;
        *) echo "unsupported {{ os() }}-{{ arch }}" >&2; exit 1 ;;
    esac
    VCPKG_BIN="$VCPKG_ROOT/vcpkg"
    [[ "{{ os() }}" == "windows" ]] && VCPKG_BIN="$VCPKG_BIN.exe"
    JS_ENGINE="${POLYC_JS_ENGINE:-hermes}"
    # Match polyregion's sysroot pin so vcpkg ports compile against the same glibc/libstdc++.
    if [ -d "{{ sysroot_path }}" ]; then export CMAKE_SYSROOT="{{ sysroot_path }}"; fi
    echo "Installing vcpkg deps (triplet=$TRIPLET, feature=$JS_ENGINE, sysroot=${CMAKE_SYSROOT:-none}) -> $VCPKG_INSTALLED_DIR" >&2
    mkdir -p "$VCPKG_INSTALLED_DIR"
    "$VCPKG_BIN" install \
        --x-manifest-root=native \
        --x-install-root="$VCPKG_INSTALLED_DIR" \
        --x-feature="$JS_ENGINE" \
        --triplet="$TRIPLET"

# Build the AL8 sysroot (podman/docker required) and extract to native/sysroot-{arch}
sysroot:
    #!/usr/bin/env bash
    set -euo pipefail
    cd native
    ./make_sysroot.sh {{ arch }}
    rm -rf "sysroot-{{ arch }}"
    mkdir -p "sysroot-{{ arch }}"
    tar xf "out/sysroot-build/al8/sysroot-al8-{{ arch }}.tar.xz" -C "sysroot-{{ arch }}"
    echo "sysroot ready at native/sysroot-{{ arch }}"

# Build the bundled LLVM/Clang/LLD/Flang/MLIR dist; cached on rerun.
llvm        extra='': (_native "LLVM"        extra)

# Build the AMDGPU/NVPTX device bitcode libs.
device-libs extra='': (_native "DEVICE_LIBS" extra)

# cmake-configure the polyregion native build.
configure   extra='': (_native "CONFIGURE"   extra)

# Build + stage the relocatable polyregion dist (bin/ + lib/).
dist        extra='': (_native "DIST"        extra)

# Build + stage the test-only dist (test binaries + check_* sources).
test-dist   extra='': (_native "DIST_TEST"   extra)

# Smoke-check a built dist: compile hello/offload programs through clang/flang/polycpp/polyfc.
dist-check  extra='': (_native "CHECK"       extra)

# Incremental build of a single ninja target (default: all); pass extra cmake flags via second arg.
build target='all' extra='': (_native "BUILD" ("-DTARGET=" + target + " " + extra))

# Run `cmake -DACTION=<action> -P native/build.cmake`. Auto-passes -DCMAKE_SYSROOT if it exists.
_native action extra='':
    #!/usr/bin/env bash
    set -euo pipefail
    SYSROOT_FLAG=()
    [ -d "{{ sysroot_path }}" ] && SYSROOT_FLAG=(-DCMAKE_SYSROOT="{{ sysroot_path }}")
    cd native && cmake -DCMAKE_BUILD_TYPE={{ build_type }} -DARCH={{ arch }} -DACTION={{ action }} ${SYSROOT_FLAG[@]+"${SYSROOT_FLAG[@]}"} {{ extra }} -P build.cmake

# === Aggregate ===

# Local mirror of the CI checks.
ci: codegen-check format-check lint-ci

# === Internal ===

# Locate the newest cmake-build-*/build-* dir under native/. Honours $POLYREGION_NATIVE_BUILD.
_native-build:
    #!/usr/bin/env bash
    set -uo pipefail
    if [ -n "{{ native_build }}" ]; then echo "{{ native_build }}"; exit 0; fi
    shopt -s nullglob
    candidates=(native/cmake-build-*/CMakeCache.txt native/build-*/CMakeCache.txt)
    [ ${#candidates[@]} -eq 0 ] && exit 0
    ls -td "${candidates[@]}" 2>/dev/null | head -1 | xargs -r dirname
