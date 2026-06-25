set shell := ["bash", "-cu"]
set windows-shell := ["bash", "-cu"]
set dotenv-path := "native/.env"

native_build := env_var_or_default('POLYREGION_NATIVE_BUILD', '')
arch         := env_var_or_default('POLYREGION_ARCH', env_var_or_default('ARCH', `uname -m`))
build_type   := env_var_or_default('BUILD_TYPE', 'Release')
container    := env_var_or_default('CONTAINER', 'podman')
sbt          := 'sbt -no-colors'

# `just --set dylib OFF build-llvm` builds a static dist (no libLLVM.so / libMLIR.so / libclang-cpp.so).
dylib := env_var_or_default('POLYREGION_LLVM_DYLIB', 'ON')
export POLYREGION_LLVM_DYLIB := dylib

# Auto-export VCPKG_ROOT to ./vcpkg (populated by `just build-vcpkg`) if the env doesn't set it.
export VCPKG_ROOT := env_var_or_default('RUNVCPKG_VCPKG_ROOT', env_var_or_default('VCPKG_ROOT', justfile_directory() / "vcpkg"))

# Shared vcpkg install root so `just build-vcpkg-deps` and `just configure` agree on a location.
export VCPKG_INSTALLED_DIR := env_var_or_default('VCPKG_INSTALLED_DIR', justfile_directory() / "native" / ".vcpkg" / "vcpkg_installed")

actionlint_version  := '1.7.12'
clang_format_version := '20.1.0'
# See https://github.com/muttleyxd/clang-tools-static-binaries/releases
clang_format_release := 'master-796e77c'

# CI sets SYSROOT_PATH; locally defaults to sysroot/out/{arch}.
sysroot_path := env_var_or_default('SYSROOT_PATH', justfile_directory() / "sysroot" / "out" / arch)

# === Default ===

# List all recipes.
default: install-git-hooks
    @just --list

# Point git at .githooks/ for this clone (idempotent; runs on first `just`).
install-git-hooks:
    #!/usr/bin/env bash
    set -euo pipefail
    git rev-parse --git-dir >/dev/null 2>&1 || exit 0
    current=$(git config --local --default '' core.hooksPath)
    [ "$current" = ".githooks" ] && exit 0
    git config --local core.hooksPath .githooks
    echo "git hooks: pointed at .githooks/"

# Print the resolved build settings.
env:
    #!/usr/bin/env bash
    set -uo pipefail
    note() { [ -d "$1" ] || printf ' (missing)'; }
    printf '%-12s = %s\n' arch         '{{ arch }}'
    printf '%-12s = %s\n' build_type   '{{ build_type }}'
    printf '%-12s = %s\n' dylib        '{{ dylib }}'
    printf '%-12s = %s%s\n' sysroot_path '{{ sysroot_path }}' "$(note '{{ sysroot_path }}')"
    printf '%-12s = %s%s\n' vcpkg_root   "${VCPKG_ROOT:-(unset)}" "$(note "${VCPKG_ROOT:-/}")"
    printf '%-12s = %s\n' vcpkg_commit "${VCPKG_COMMIT:-(unset)}"
    printf '%-12s = %s\n' java_home    "${JAVA_HOME:-(unset)}"

# === Format ===

# clang-format + scalafmt, in place, in parallel.
format:       (_format "format"       "scalafmtAll"      "scalafmtSbt")

# clang-format + scalafmt, dry-run, in parallel; non-zero exit on diff.
check-format: check-header (_format "format-check" "scalafmtCheckAll" "scalafmtSbtCheck")

_format mode sbt_task_a sbt_task_b:
    #!/usr/bin/env bash
    # XXX no `-e`; we collect both native + sbt rc's after `wait` and report jointly.
    set -uo pipefail
    CF="$(just _clang-format)"
    case "{{ mode }}" in
        format)       cf_args=(--style=file -i) ;;
        format-check) cf_args=(--style=file --dry-run --Werror) ;;
        *) echo "unknown format mode: {{ mode }}" >&2; exit 2 ;;
    esac
    echo "Native:  clang-format {{ mode }} via $CF"
    git ls-files -z -- '*.cpp' '*.cc' '*.cu' '*.h' '*.hpp' \
        | grep -zvE '^native/(polyinvoke/thirdparty/|polyinvoke/test/kernels/generated_|polyc/generated/|polyc/include/polyregion/polypass\.h$)' \
        | xargs -0 -r -P "$(nproc 2>/dev/null || echo 4)" -n 32 "$CF" "${cf_args[@]}" &
    pid_n=$!
    if command -v sbt >/dev/null 2>&1; then
        echo "Scala:   sbt {{ sbt_task_a }} {{ sbt_task_b }}"
        (cd frontend && {{ sbt }} {{ sbt_task_a }} {{ sbt_task_b }}) &
        pid_s=$!
    else
        echo "sbt not found on PATH - skipping Scala format" >&2
        pid_s=
    fi
    wait $pid_n; rc_n=$?
    rc_s=0
    [ -n "$pid_s" ] && { wait $pid_s; rc_s=$?; }
    if [ "$rc_n" -ne 0 ] || [ "$rc_s" -ne 0 ]; then
        echo "{{ mode }} failed (native=$rc_n sbt=$rc_s)" >&2
        exit 1
    fi

# === Codegen ===

# Regenerate Scala-derived C++/JNI sources, then clang-format the output.
codegen:       _codegen-sbt _codegen-format

# codegen + git diff against committed state, fails if regenerated sources drift.
check-codegen: check-header codegen _codegen-diff

# Regen test/kernels/generated_*.hpp; needs `just build-llvm` plus spirv-link and glslangValidator on PATH.
codegen-kernels:
    #!/usr/bin/env bash
    set -euo pipefail
    BUILD=$(just _native-build)
    [ -z "$BUILD" ] && { echo "no configured native build dir; run 'just configure' first" >&2; exit 1; }
    cmake --build "$BUILD" --target polyinvoke-regen-kernels

_codegen-sbt:
    cd frontend && {{ sbt }} 'codegen/genCodegen'

_codegen-format:
    "$(just _clang-format)" -i native/polyast/generated/*.{h,cpp} native/bindings/jvm/generated/*.{h,cpp}

_codegen-diff:
    git diff --exit-code -- native/polyast/generated native/bindings/jvm/generated native/polyc/generated native/polyc/include/polyregion/polypass.h

# === Pass bundles ===

# Build the Scala.js pass bundle (closure-compiled) and stage at native/polyc/polypass.js.
build-pass-js:
    cd frontend && {{ sbt }} 'passJS/exportPassBundle'

# Build the Scala Native pass DSO -> native/polyc/libpolypass.{so,dylib,dll} (sysroot propagated to clang).
build-pass-native:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -d "{{ sysroot_path }}" ]; then export CMAKE_SYSROOT="{{ sysroot_path }}"; fi
    cd frontend && {{ sbt }} 'passNative/exportPassDso'

# === Lint ===

# Validate .github/workflows/*.yaml against actionlint; auto-fetches the binary on first run.
check-ci:
    #!/usr/bin/env bash
    set -euo pipefail
    "$(just _actionlint)" -color .github/workflows/*.yaml

# Forbid <filesystem|regex|codecvt|iostream|sstream|ostream|iomanip> in tracked C/C++ sources (use LLVM sys::/fmt).
check-header:
    #!/usr/bin/env bash
    set -u
    pat='^[[:space:]]*#[[:space:]]*include[[:space:]]*<(filesystem|regex|codecvt|iostream|sstream|ostream|iomanip)>'
    hits=$(git ls-files -z -- '*.cpp' '*.cc' '*.cu' '*.h' '*.hpp' '*.h.in' '*.hpp.in' | xargs -0 grep -nE "$pat" 2>/dev/null)
    [ -z "$hits" ] || { echo "banned headers (use fmt/LLVM alternatives):"; echo "$hits"; exit 1; }

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
test-scala *args='':
    #!/usr/bin/env bash
    set -uo pipefail
    BUILD=$(just _native-build)
    if [ -z "${POLYREGION_NATIVE_LIB_DIR:-}" ] && [ -n "$BUILD" ]; then
        export POLYREGION_NATIVE_LIB_DIR="$PWD/$BUILD/bindings/jvm"
    fi
    cd frontend && {{ sbt }} 'compiler-testsuite-scala/test' {{ args }}

# Run the native ctest suite; device targets skip without a GPU/emulator. Extra ctest flags via *args.
# Skips are not ctest SKIP entries (that property is O(n^2) at parse); the runner logs them per workdir.
test-native *args='':
    #!/usr/bin/env bash
    set -euo pipefail
    BUILD=$(just _native-build)
    rm -f "$BUILD"/*/polytest-discover/*.ids
    find "$BUILD" -name polytest-skips.log -delete 2>/dev/null || true
    ninja -C "$BUILD" polyinvoke-tests-discover polycpp-tests-discover polyfc-tests-discover
    rc=0; ctest --test-dir "$BUILD" --timeout 600 {{ args }} || rc=$?
    skips=$(find "$BUILD" -name polytest-skips.log -exec cat {} + 2>/dev/null | wc -l || true)
    [ "${skips:-0}" -gt 0 ] && echo "polytest: ${skips} skipped (find $BUILD -name polytest-skips.log | xargs cat)" || true
    exit $rc

# Build the software emulator bundle -> emulators/out. linux: container (CONTAINER=docker overrides podman); macos/windows: native on the host via macos.sh / windows.bat
build-emulators:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ replace(justfile_directory(), "\\", "/") }}/emulators"
    case "{{ os() }}" in
      macos)   exec bash macos.sh "$PWD/out" ;;
      windows) exec cmd //c "windows.bat $(cygpath -w "$PWD" 2>/dev/null || echo "$PWD")\\out" ;;
      linux)   : ;;
      *) echo "build-emulators: unsupported OS {{ os() }}" >&2; exit 1 ;;
    esac
    TMP="${EMU_TMPDIR:-$HOME/.podman-tmp}"; mkdir -p "$TMP"
    TMPDIR="$TMP" {{ container }} build -f Dockerfile --target check -t polyemu .
    rm -rf out
    cid=$({{ container }} create polyemu)
    {{ container }} cp "$cid:/out" ./out
    {{ container }} rm "$cid" >/dev/null

# Run the native ctest suite against the software emulators; extra ctest flags via *args.
test-native-with-emulators *args='':
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ justfile_directory() }}
    [ -f emulators/out/env.sh ] || { echo "no emulators/out -- run 'just build-emulators' first"; exit 1; }
    source emulators/out/env.sh
    BUILD=$(just _native-build)
    export POLYREGION_TEST_PROFILE=emulators
    export POLYINVOKE_TEST_LOCK="${POLYINVOKE_TEST_LOCK:-0}"
    # XXX ctest bakes target names at discovery; re-list under this profile first.
    rm -f "$BUILD"/*/polytest-discover/*.ids
    ninja -C "$BUILD" polyinvoke-tests-discover polycpp-tests-discover polyfc-tests-discover
    ctest --test-dir "$BUILD" --timeout 600 {{ args }}

# Smoke test of the emulator bundle (Vulkan on lavapipe, OpenCL on pocl[+rusticl], CUDA on gpuocelot/linux).
check-emulators bundle='':
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ replace(justfile_directory(), "\\", "/") }}/emulators"
    case "{{ os() }}" in
      macos)   exec bash macos.sh --check "${1:-$PWD/out}" ;;
      windows) exec cmd //c "windows.bat --check" ;;
      *)       exec bash run.sh {{ bundle }} ;;
    esac

# === Build wrappers ===

# Clone + bootstrap vcpkg at the native/.env pin and sync vcpkg.json's builtin-baseline. Idempotent.
build-vcpkg: _vcpkg-sync-baseline
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d vcpkg/.git ]; then git clone https://github.com/microsoft/vcpkg.git; fi
    git -C vcpkg fetch --depth=1 origin "$VCPKG_COMMIT"
    git -C vcpkg checkout "$VCPKG_COMMIT"
    BOOTSTRAP=bootstrap-vcpkg.sh
    [[ "{{ os() }}" == "windows" ]] && BOOTSTRAP=bootstrap-vcpkg.bat
    if [ ! -x vcpkg/vcpkg ] && [ ! -x vcpkg/vcpkg.exe ]; then
        (cd vcpkg && "./$BOOTSTRAP" -disableMetrics)
    fi
    echo "vcpkg ready at $PWD/vcpkg"

# Install vcpkg manifest deps to $VCPKG_INSTALLED_DIR for the current host/arch.
build-vcpkg-deps:
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
    if [ -z "${VCPKG_ROOT:-}" ]; then
        echo "VCPKG_ROOT not set; run \`just build-vcpkg\` first or export VCPKG_ROOT" >&2; exit 1
    fi
    VCPKG_BIN="$VCPKG_ROOT/vcpkg"
    BOOTSTRAP="bootstrap-vcpkg.sh"
    [[ "{{ os() }}" == "windows" ]] && { VCPKG_BIN="$VCPKG_BIN.exe"; BOOTSTRAP="bootstrap-vcpkg.bat"; }
    if [ ! -x "$VCPKG_BIN" ]; then
        if [ -x "$VCPKG_ROOT/$BOOTSTRAP" ]; then
            (cd "$VCPKG_ROOT" && "./$BOOTSTRAP" -disableMetrics)
        else
            echo "vcpkg checkout incomplete at $VCPKG_ROOT (no $BOOTSTRAP); run \`just build-vcpkg\`" >&2; exit 1
        fi
    fi
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

# Rewrite native/vcpkg.json's `builtin-baseline` from $VCPKG_COMMIT. Cheap; safe to depend on.
_vcpkg-sync-baseline:
    #!/usr/bin/env bash
    set -euo pipefail
    BASELINE=$(jq -r '."builtin-baseline" // empty' native/vcpkg.json)
    if [ "$VCPKG_COMMIT" != "$BASELINE" ]; then
        echo "vcpkg.json builtin-baseline: $BASELINE -> $VCPKG_COMMIT (from native/.env)"
        jq --indent 2 --arg c "$VCPKG_COMMIT" '."builtin-baseline" = $c' native/vcpkg.json > native/vcpkg.json.tmp
        mv native/vcpkg.json.tmp native/vcpkg.json
    fi

# Build the AL8 sysroot (sysroot/Dockerfile) into sysroot/out/<arch> (CONTAINER=docker overrides podman).
build-sysroot:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ justfile_directory() }}/sysroot
    case "{{ arch }}" in
      x86_64)  rhel=x86_64-redhat-linux;  gnu=x86_64-linux-gnu;  plat=linux/amd64; extra=libquadmath ;;
      aarch64) rhel=aarch64-redhat-linux; gnu=aarch64-linux-gnu; plat=linux/arm64; extra= ;;
      *) echo "unsupported arch: {{ arch }} (x86_64/aarch64 only)" >&2; exit 1 ;;
    esac
    flags=()  # cross-arch flag only when target != host
    if [ "{{ arch }}" != "$(uname -m)" ]; then
      case "{{ container }}" in podman) flags=(--arch="{{ arch }}");; docker) flags=(--platform="$plat");; esac
    fi
    mkdir -p out
    img=polyregion-sysroot-al8:{{ arch }}
    {{ container }} build --pull ${flags[@]+"${flags[@]}"} \
      --build-arg RHEL_TRIPLE="$rhel" --build-arg GNU_TRIPLE="$gnu" --build-arg EXTRA_PKGS="$extra" \
      -t "$img" .
    cid=$({{ container }} create "$img")
    {{ container }} export "$cid" | xz -T0 > "out/sysroot-al8-{{ arch }}.tar.xz"
    {{ container }} rm "$cid" >/dev/null
    rm -rf "out/{{ arch }}"; mkdir -p "out/{{ arch }}"
    tar xf "out/sysroot-al8-{{ arch }}.tar.xz" -C "out/{{ arch }}"
    echo "sysroot ready at sysroot/out/{{ arch }}"

# Build the bundled LLVM/Clang/LLD/Flang/MLIR dist; cached on rerun.
build-llvm        extra='': (_native "LLVM"        extra)

# Build the AMDGPU/NVPTX device bitcode libs.
build-device-libs extra='': (_native "DEVICE_LIBS" extra)

# cmake-configure the polyregion native build.
configure         extra='': (_native "CONFIGURE"   extra)

# Build + stage the relocatable polyregion dist (bin/ + lib/).
build-dist        extra='': (_native "DIST"        extra)

# Build + stage the test-only dist (test binaries + check_* sources).
build-test-dist   extra='': (_native "DIST_TEST"   extra)

# Smoke-check a built dist: compile hello/offload programs through clang/flang/polycpp/polyfc.
check-dist        extra='': (_native "CHECK"       extra)

# Run the mini-app matrix against a built dist + optional emulators.
check-miniapps:
    ./ci/check_miniapps.sh

# Incremental build of all ninja targets.
build-all     extra='': (_build "all"     extra) check-fused

# Incremental build of the polyc driver.
build-polyc   extra='': (_build "polyc"   extra)

# Incremental build of the polycpp driver.
build-polycpp extra='': (_build "polycpp" extra)

# Incremental build of the polyfc driver.
build-polyfc  extra='': (_build "polyfc"  extra)

# Incremental build of an arbitrary ninja target (escape hatch for non-default names).
build target  extra='': (_build target    extra)

# BUILD a single ninja target; interpolation here keeps callers free of nested-paren deps.
_build target extra='':
    @just _native BUILD "-DTARGET={{ target }} {{ extra }}"

# Build + verify the fused driver (configure + polycpp + polyfc with POLYREGION_FUSED_DRIVER=ON); non-Windows only.
check-fused:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "{{ os() }}" == "windows" ]]; then
        echo "check-fused: skipped on windows"
        exit 0
    fi
    export POLYREGION_FUSED_DRIVER=ON
    just configure
    just build-polycpp
    just build-polyfc

# === ASan variants ===

# cmake-configure the ASan+UBSan build.
configure-san extra='':
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{ os() }}-{{ arch }}" in
        linux-x86_64)               tc=linux-clang-amd64-asan ;;
        linux-aarch64)              tc=linux-clang-aarch64-asan ;;
        macos-arm64)                tc=darwin-clang-arm64-asan ;;
        macos-x86_64|macos-amd64)   tc=darwin-clang-amd64-asan ;;
        *) echo "configure-san: unsupported {{ os() }}-{{ arch }}" >&2; exit 1 ;;
    esac
    POLYREGION_ASAN=ON just configure "-DCMAKE_TOOLCHAIN_FILE={{ justfile_directory() }}/native/toolchains/${tc}.cmake {{ extra }}"

# Incremental build of a ninja target in the -asan tree.
build-san target='all' extra='':
    POLYREGION_ASAN=ON just build {{ target }} "{{ extra }}"

# Build + stage the relocatable -asan dist.
build-dist-san extra='':
    POLYREGION_ASAN=ON just build-dist "{{ extra }}"

# Build + stage the -asan test dist.
build-test-dist-san extra='':
    POLYREGION_ASAN=ON just build-test-dist "{{ extra }}"

# Run the native ctest suite against the -asan tree; extra ctest flags via *args.
test-native-san *args='':
    #!/usr/bin/env bash
    set -euo pipefail
    export POLYREGION_ASAN=ON
    BUILD=$(just _native-build)
    ctest --test-dir "$BUILD" {{ args }}

# Run `cmake -DACTION=<action> -P native/build.cmake`. Auto-passes -DCMAKE_SYSROOT if it exists.
_native action extra='':
    #!/usr/bin/env bash
    set -euo pipefail
    SYSROOT_FLAG=()
    [ -d "{{ sysroot_path }}" ] && SYSROOT_FLAG=(-DCMAKE_SYSROOT="{{ sysroot_path }}")
    cd native && cmake -DCMAKE_BUILD_TYPE={{ build_type }} -DARCH={{ arch }} -DACTION={{ action }} ${SYSROOT_FLAG[@]+"${SYSROOT_FLAG[@]}"} {{ extra }} -P build.cmake

# === Clean ===

# Remove the LLVM build trees (preserves llvm-patches* sources).
clean-llvm:
    #!/usr/bin/env bash
    set -uo pipefail
    shopt -s nullglob
    paths=(native/out/llvm-Release-* native/out/llvm-Debug-* native/out/llvm-RelWithDebInfo-*)
    [ ${#paths[@]} -eq 0 ] && { echo "clean-llvm: nothing to remove"; exit 0; }
    echo "clean-llvm: ${paths[*]}"
    rm -rf "${paths[@]}"

# Remove the staged polyregion dist + test dist + dist-check build.
clean-dist:
    #!/usr/bin/env bash
    set -uo pipefail
    shopt -s nullglob
    paths=(native/out/polyregion-*-dist native/out/build-dist-check-*)
    [ ${#paths[@]} -eq 0 ] && { echo "clean-dist: nothing to remove"; exit 0; }
    echo "clean-dist: ${paths[*]}"
    rm -rf "${paths[@]}"

# Remove the built sysroots (sysroot/out: tarballs + extracted trees).
clean-sysroot:
    #!/usr/bin/env bash
    set -uo pipefail
    [ -d sysroot/out ] || { echo "clean-sysroot: nothing to remove"; exit 0; }
    echo "clean-sysroot: sysroot/out"
    rm -rf sysroot/out

# Remove the repo-local vcpkg clone from `just build-vcpkg`; leaves $VCPKG_ROOT alone if it points elsewhere.
clean-vcpkg:
    #!/usr/bin/env bash
    set -uo pipefail
    if [ -d vcpkg/.git ]; then
        echo "clean-vcpkg: ./vcpkg"
        rm -rf vcpkg
    else
        echo "clean-vcpkg: nothing to remove"
    fi

# Remove polyregion CMake build trees + cached vcpkg install (forces a fresh configure).
clean-build:
    #!/usr/bin/env bash
    set -uo pipefail
    shopt -s nullglob
    paths=(native/out/build-* native/cmake-build-* native/.vcpkg)
    [ ${#paths[@]} -eq 0 ] && { echo "clean-build: nothing to remove"; exit 0; }
    echo "clean-build: ${paths[*]}"
    rm -rf "${paths[@]}"

# Run all clean recipes in parallel.
[parallel]
clean-all: clean-llvm clean-dist clean-sysroot clean-vcpkg clean-build

# === Aggregate ===

# Local mirror of the CI checks.
ci: check-codegen check-format check-ci

# === Internal ===

# Locate the newest configured build dir (native/out/build-* or native/cmake-build-*); honours $POLYREGION_NATIVE_BUILD.
_native-build:
    #!/usr/bin/env bash
    set -uo pipefail
    shopt -s nullglob
    if [ -n "{{ native_build }}" ]; then echo "{{ native_build }}"; exit 0; fi
    caches=(native/out/build-*/CMakeCache.txt native/cmake-build-*/CMakeCache.txt)
    # variant trees (-fused, -asan) are only selected when the matching env asks for them
    kept=()
    for c in "${caches[@]}"; do
        case "$c" in
            */build-*-asan/CMakeCache.txt)  [ "${POLYREGION_ASAN:-}" = "ON" ] && kept+=("$c") ;;
            */build-*-fused/CMakeCache.txt) [ "${POLYREGION_FUSED_DRIVER:-}" = "ON" ] && kept+=("$c") ;;
            *) [ "${POLYREGION_ASAN:-}" != "ON" ] && [ "${POLYREGION_FUSED_DRIVER:-}" != "ON" ] && kept+=("$c") ;;
        esac
    done
    caches=(${kept[@]+"${kept[@]}"})
    [ ${#caches[@]} -eq 0 ] && exit 0
    ls -td "${caches[@]}" | head -1 | xargs -r dirname
