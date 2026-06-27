# Polyregion

Polyregion compiles normal/idiomatic source-level code fragments to machine code targeting CPUs and GPUs from the host language directly. 
Supported frontends are:
* `polycpp` - a clang frontend that intercepts `-fstdpar` regions
* `polyfc`  - a flang frontend that intercepts Fortran `do concurrent`
* A Scala macro that delegates to the `polyc` AST/IR pipeline

## Build & debug

The native side (compilers, runtime, plugins) lives under `native/` and is built with CMake + vcpkg. The Scala frontend lives under `frontend/` and is built with sbt.
Top-level orchestration is via [`just`](https://github.com/casey/just); `just` from the repo root lists every recipe and points git at the tracked `.githooks/` directory on first run (manual: `just install-git-hooks`).

### Quick start (fresh clone)

Prerequisites:
 * `JAVA_HOME` (JDK 21+)
   * Fedora: `sudo dnf install java-latest-openjdk-devel` 
   * Windows: `winget install -e --id EclipseAdoptium.Temurin.21.JDK` 
 * `sbt` - https://www.scala-sbt.org/download/
 * bash
   * Windows: use Git bash 

***Important: Use Git Bash on Windows; just recipies are bash scripts***

```shell
export JAVA_HOME=/usr/lib/jvm/java  # any JDK 21+ install root

just build-vcpkg               # clone + bootstrap vcpkg at the pinned commit
just build-sysroot             # AL8 sysroot for portable binaries (optional; podman or docker required)
                               # skip for host-native; absence is detected as "no sysroot"
just build-llvm                # bundled LLVM/Clang/LLD/Flang/MLIR (matches sysroot if present)
just build-vcpkg-deps          # manifest deps (Catch2, fmt, libffi, JS engine, ...) into native/.vcpkg
just build-pass-native         # Scala-Native pass DSO (libpolypass) that polyc embeds (needs sbt)
just configure                 # CMake-configure the native build
just build-dist                # polyregion dist (bin/ + lib/)
just check-dist                # smoke test
```

`build-pass-native` builds the default `dso` pass bundle that `configure` then requires; to use the
JavaScript bundle instead, run `just build-pass-js` and pass `--set POLYC_PASS_BUNDLE js` to `configure`.

`just env` prints the resolved settings (`arch`, `build_type`, `dylib`, `sysroot_path`, `vcpkg_root`, `vcpkg_commit`, `java_home`).

### Native - incremental

```shell
just configure              # one-time CMake configure of native/

just build polycpp          # build a single component
just build all              # everything

# Useful targets:
#   polyc       — AST → LLVM IR backend (subprocess invoked by frontends)
#   polycpp     — C++ frontend (clang plugin + driver)
#   polyfc      — Fortran frontend (flang plugin + driver)
#   polystl     — runtime library linked into user binaries
#   polyc-test  — polyc unit tests
```

For IDE-driven CMake configurations or anything fancier, see `native/README.md` for the raw cmake invocation.

The `polycpp` and `polyfc` targets transitively rebuild everything they depend on (clang plugin, polystl, polyreflect plugin, polyc, etc.).

### Compiling and running an offload program

`polycpp` is a clang wrapper. It forwards to a real clang via `POLYCPP_DRIVER` and adds the polycpp clang plugin, polyreflect plugin, and runtime link.

```shell
export POLYCPP_DRIVER=$PWD/native/out/polyregion-Release-x86_64-dylib-dist/bin/clang++
POLYCPP=$PWD/native/out/build-linux-x86_64-dylib/polycpp/polycpp

# -fstdpar enables the polyreflect+polystl wiring; -fstdpar-arch picks the GPU target.
$POLYCPP -O1 -fstdpar -fstdpar-arch=cuda@sm_89 -fstdpar-mem=reflect -fstdpar-rt=dynamic \
    -o /tmp/check_struct native/polycpp/test/check_struct.cpp -DCHECK_CAPTURE==

POLYRT_PLATFORM=cuda@sm_89 POLYRT_HOST_FALLBACK=0 /tmp/check_struct
```

Runtime platform selectors (`POLYRT_PLATFORM`):
- `host@native` — host CPU via libffi (default if unset)
- `cuda@sm_<arch>` — NVIDIA via cuda runtime (e.g. `sm_89`)
- `hsa@gfx<arch>` — AMD via ROCm/HSA (e.g. `gfx1036`)
- `cl@<vendor>` — OpenCL
- `vulkan@<vendor>` — Vulkan/SPIRV

### Debug options

| env var | values | effect |
|---|---|---|
| `POLYRT_PLATFORM` | see above | select GPU/CPU backend |
| `POLYRT_HOST_FALLBACK` | `0`/`1` | when `1`, fall back to host CPU if GPU init fails (default `1`) |
| `POLYRT_DEBUG` | `0`-`3` | runtime log level; `2` (Debug) prints SMA mirror/sync trace |
| `POLYSTL_NO_OFFLOAD` | any non-empty | skip GPU/host dispatch entirely; run the lambda inline on the calling thread |
| `POLYFRONT_VERBOSE` | `0`/`1` | when `1`, dump polyAST IR before invoking polyc |

### Native test suite

The native suite is driven by ctest:

```sh
just build all       # build the drivers, runtime and test binaries first
just test-native     # run the ctest suite (extra ctest flags pass through, e.g. `just test-native -j 8`)
```

Tests are `#pragma region`-annotated `.cpp` files under `native/polycpp/test/`. 
Each file declares one or more cases, the compile flags, and the expected stdout via `#pragma region requires:`.

### Frontend (Scala)

```sh
cd frontend
sbt compile
sbt test
```

The Scala build picks up native artifacts from `native/cmake-build-debug-clang/bindings/jvm/`; build that target before running JVM tests.

#### Running the Scala test suite

The test suite (`compiler-testsuite-scala`) is a munit suite that compiles each kernel via the Scala 3 macro pipeline, runs both a JDK reference and the offloaded LLVM-compiled version, and asserts they match. It's heavyweight (~6100 cases, 13–17 min on a typical desktop) because every `testExpr` triggers a full macro expansion + LLVM codegen.

Prerequisites: the native JNI shared library must be built first — the test runner `dlopen`s it from `native/cmake-build-release-clang/bindings/jvm/libpolyc-JNI.so`.

```sh
# 1. Build the native JNI bindings the test runner depends on.
cmake --build native/cmake-build-release-clang --target polyc-JNI -j

# 2. Run the full suite (~20 min).
just test-scala

#  sbt for specific runs:
cd frontend
sbt 'compiler-testsuite-scala/testOnly polyregion.GivenSuite'
sbt 'compiler-testsuite-scala/testOnly polyregion.ControlFlowSuite -- *while-le-inc*'
```

If you change anything in `compiler/` (macros) or `prism/StdLib.scala`, force a clean macro re-expansion as sbt's incremental compiler doesn't always track macro changes correctly:

```sh
rm -rf compiler-testsuite-scala/target/scala-3.7.4 compiler/target/scala-3.7.4/classes
sbt 'compiler-testsuite-scala/test'
```

Likewise after editing C++ in `native/polyc/backend/`, rebuild `polyc-JNI` (the JVM caches the `.so` per process, so a stale build will silently mask backend changes).

Individual suites are gated by booleans in `compiler-testsuite-scala/src/test/scala/polyregion/Toggles.scala`.
