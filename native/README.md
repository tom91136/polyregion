# Native build

Top-level task runner is `just` (see `../justfile` at the repo root); each recipe is a thin
wrapper around `cmake -P build.cmake -DACTION=<action>`. Run `just` from the repo root to list
recipes. The underlying cmake form still works if you prefer it.

## Actions

| Action        | `just` recipe   | What it does                                                                                                                                                                                                         |
|---------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LLVM`        | `just llvm`         | Configure + build + install LLVM/Clang/LLD/Flang/MLIR into `llvm-${BUILD_TYPE}-${ARCH}-${VARIANT}/`. Slow (hours, first time).                                                                                       |
| `DEVICE_LIBS` | `just device-libs`  | Build AMDGPU/NVPTX device libs (offload bitcode).                                                                                                                                                                    |
| `CONFIGURE`   | `just configure`    | Configure polyregion build into `build-${platform}-${ARCH}-${VARIANT}/`.                                                                                                                                             |
| `BUILD`       | `just build [tgt]`  | Build a target. `-DTARGET=<name>` selects (default `all`).                                                                                                                                                           |
| `DIST`        | `just dist`         | Install polyregion into `polyregion-${BUILD_TYPE}-${ARCH}-${VARIANT}-dist/`. Builds install deps as needed.                                                                                                          |
| `DIST_TEST`   | `just test-dist`    | Bundle the test binaries + sources into `polyregion-test-${BUILD_TYPE}-${ARCH}-${VARIANT}-dist/`.                                                                                                                    |
| `CHECK`       | `just check-dist`   | Run the dist sanity check (compiles hello/offload programs through `clang`/`flang-new`/`polycpp`/`polyfc`, verifies output binaries don't depend on shipped DSOs, compile-only checks for cuda/hsa/spirv/metal/c11). |

## Common options

| Option                | Default    | Notes                                                                          |
|-----------------------|------------|--------------------------------------------------------------------------------|
| `-DCMAKE_BUILD_TYPE=` | (required) | `Release` / `Debug`                                                            |
| `-DARCH=`             | host arch  | `x86_64`, `aarch64`, ...                                                       |
| `-DTARGET=`           | —          | for `BUILD` only                                                               |
| `-DCMAKE_SYSROOT=`    | —          | required for cross builds; pass `/` when configuring locally without a sysroot |

## Environment

| Variable                | Default                     | Effect                                                                                                                                                                                          |
|-------------------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `POLYREGION_LLVM_DYLIB` | `ON`                        | `OFF` builds a static dist (no `libLLVM.so` / `libMLIR.so` / `libclang-cpp.so`). Affects both `LLVM` and polyregion configure/build/dist; output dir gets `-static` suffix instead of `-dylib`. |
| `VCPKG_ROOT`            | (required)                  | vcpkg checkout                                                                                                                                                                                  |
| `JAVA_HOME`             | (required for JNI bindings) | JDK install root                                                                                                                                                                                |

If `ccache` is on `PATH` it is used automatically (set by the toolchain files).

## Examples

Full dylib release dist + smoke check (from the repo root):

```sh
just llvm        # one-time, slow
just configure
just dist
just check-dist
```

Static dist (no shipped LLVM dylibs):

```sh
just --set dylib OFF llvm
just --set dylib OFF configure
just --set dylib OFF dist
just --set dylib OFF check-dist
```

Iterate on a single target without re-installing:

```sh
just build polycpp
```

Cross-arch (sysroot is auto-passed when `native/sysroot-${ARCH}` exists):

```sh
just sysroot ARCH=aarch64    # one-time, downloads + extracts Debian sysroot
just llvm    ARCH=aarch64
```

`just env` prints the resolved settings (`arch`, `build_type`, `dylib`, `sysroot_path`).

## CI release artefacts

Each platform workflow (`linux.yaml`, `macos.yaml`, `windows.yaml`) runs `check-codegen`, then
matrix-driven `build` (`LLVM` + `DEVICE_LIBS` + `CONFIGURE` + `BUILD` + `DIST` + `DIST_TEST`),
then in parallel `check-dist`, `test-native`, and (linux only) `test-scala`. The packaged
dist is uploaded as a GitHub Actions artefact named `polyregion-${platform}-${arch}-${build_type}`
(90-day retention, not a release).

### Running CI locally with `act`

[`act`](https://github.com/nektos/act) executes the workflow on a local podman/docker runner.
`.actrc` at the repo root configures it: x86_64-only matrix, host network (needed because
rootless podman's default bridge has no IPv6 egress), workspace bind-mount so artefacts
survive between runs, and `/tmp/act-ccache` for compiler cache.

```sh
act -W .github/workflows/linux.yaml -j check-codegen   # fast: sbt codegen + diff
act -W .github/workflows/linux.yaml -j build           # full LLVM build (~30 min cold)
act -W .github/workflows/linux.yaml -j check-dist      # uses the artefact from `build`
```

Notes:
- Requires `act >= 0.2.86` for upload-artifact@v7 protocol compatibility.
- macOS and Windows workflows aren't tested locally (`act` doesn't have parity for those runners).
- First run is slow (sysroot ~40s, LLVM ~30min); subsequent runs reuse the build dir on disk
  thanks to `--bind`.

## Testing

### To run everything

```bash
cmake --build <build> --target test_all
```

Builds the seven test binaries (`polycommon-tests`, `polyc-tests`,
`polyinvoke-tests`, `polyrt-tests`, `polyreflect-rt-tests`, `polycpp-tests`,
`polyfc-tests`) and runs them under `ctest -j$NPROC --output-junit
test-results.xml`. The xml drop lands in `<build>/test-results.xml` and is what
CI consumes.

### To package a relocatable test bundle

```bash
cmake --install <build> --component test-dist --prefix <path>
cd <path> && ctest
```

Bundles the test binaries, the `check_*.cpp` / `check_*.f95` sources, the
arch profiles, and a generated `CTestTestfile.cmake` under `<path>`. Run from
any machine with matching toolchains.

### To run one suite

```bash
build-*/polyfc/polyfc-tests        # all polyfc cases, arch matrix expanded
build-*/polycpp/polycpp-tests      # all polycpp cases, arch matrix expanded
```

`polycpp-tests` / `polyfc-tests` enumerate `check_*.cpp` / `check_*.f95` under
the project test directory, expand them across the arch matrix in
`test-profiles/$(hostname).env` (falls back to `default.env`), then compile in
parallel and run serially. The other five `*-tests` binaries are leaf unit
tests with no arch matrix - run them directly.

### To focus on one case

```bash
polyfc-tests -c reduce                                  # match by file shortname
polyfc-tests -c reduce -c reduce_many                   # multiple
polycpp-tests -c "app_microbude/microbude ppwi=1"       # full case name
```

`-c|--case` is repeatable. Match is on file shortname or the rendered case name
that appears in `--list` output.

### To list tasks without running

```bash
polyfc-tests -l               # full matrix
polyfc-tests -c reduce -l     # filtered
```

### To run only one mode

```bash
polyfc-tests --offload-only           # skip passthrough (host-only) runs
polyfc-tests --passthrough-only       # skip offload runs
```

### To debug a single failure

```bash
polyfc-tests -c reduce -v 2>&1 | tee /tmp/reduce.log
grep -B2 -A20 FAIL /tmp/reduce.log
```

`-v` dumps stdout, stderr, and the exact compile command for every failing
task. Tee long runs and grep the log; do not re-run for follow-up questions.

### To skip a flaky backend

```bash
POLYINVOKE_DISABLE_BACKENDS=LevelZero polyfc-tests   # drop Intel Level Zero
POLYINVOKE_DISABLE_BACKENDS=HIP        polycpp-tests # drop ROCm/HIP
```

Token must match the `magic_enum::enum_name` of `polyregion::invoke::Backend` (e.g.
`LevelZero`, `HIP`, `CUDA`, `HSA`, `OpenCL`, `Vulkan`, `Metal`).

Drops the listed backend from the arch matrix without editing the host
profile.

### To run a test binary by hand

```bash
POLYFC_DRIVER=<dist>/bin/flang \
POLYRT_PLATFORM=hsa@gfx1036 \
POLYRT_HOST_FALLBACK=0 \
<test-binary> <args>
```

Useful when attaching a debugger or inspecting generated IR.
`POLYRT_HOST_FALLBACK=0` surfaces backend faults instead of silently falling
back to the host CPU path.

### To rebuild for a single test

```bash
ninja polyfc-tests          # driver + plugins + runtime
ninja polycpp-tests         # driver + polystl + plugins + runtime
```

Pulls only what the test binary depends on - no extra targets needed for
per-test iteration.

### To debug a kernel-side fault

For `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` and similar:

1. Carve the embedded GPU ELF out of the host executable - scan for ELF magic
   with `e_machine=224` (AMDGPU) or `e_machine=190` (NVPTX).
2. `llvm-objdump -d --mcpu=<gfx>` the carved binary.
3. Check `polyc/vendor-bitcode/*.bc` for stray `target-cpu` / `target-features`
   attrs that don't match the kernel target; rebuild after scrubbing if the
   dispatch crashes on launch.
