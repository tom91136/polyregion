# Native build

All build steps go through `cmake -P build.cmake -DACTION=<action>`.

## Actions

| Action      | What it does                                                                                                                                                                                                         |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LLVM`      | Configure + build + install LLVM/Clang/LLD/Flang/MLIR into `llvm-${BUILD_TYPE}-${ARCH}-${VARIANT}/`. Slow (hours, first time).                                                                                       |
| `CONFIGURE` | Configure polyregion build into `build-${platform}-${ARCH}-${VARIANT}/`.                                                                                                                                             |
| `BUILD`     | Build a target. Requires `-DTARGET=<name>` (use `all` for everything).                                                                                                                                               |
| `DIST`      | Install polyregion into `polyregion-${BUILD_TYPE}-${ARCH}-${VARIANT}-dist/`. Builds install deps as needed.                                                                                                          |
| `CHECK`     | Run the dist sanity check (compiles hello/offload programs through `clang`/`flang-new`/`polycpp`/`polyfc`, verifies output binaries don't depend on shipped DSOs, compile-only checks for cuda/hsa/spirv/metal/c11). |

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

Full dylib release dist + smoke check:

```sh
cd native
cmake -DACTION=LLVM      -DCMAKE_BUILD_TYPE=Release -P build.cmake     # one-time
cmake -DACTION=CONFIGURE -DCMAKE_BUILD_TYPE=Release -P build.cmake
cmake -DACTION=DIST      -DCMAKE_BUILD_TYPE=Release -P build.cmake
cmake -DACTION=CHECK     -DCMAKE_BUILD_TYPE=Release -P build.cmake
```

Static dist (everything statically linked, no shipped LLVM dylibs):

```sh
POLYREGION_LLVM_DYLIB=OFF cmake -DACTION=LLVM      -DCMAKE_BUILD_TYPE=Release -P build.cmake
POLYREGION_LLVM_DYLIB=OFF cmake -DACTION=CONFIGURE -DCMAKE_BUILD_TYPE=Release -P build.cmake
POLYREGION_LLVM_DYLIB=OFF cmake -DACTION=DIST      -DCMAKE_BUILD_TYPE=Release -P build.cmake
POLYREGION_LLVM_DYLIB=OFF cmake -DACTION=CHECK     -DCMAKE_BUILD_TYPE=Release -P build.cmake
```

Iterate on a single target without re-installing:

```sh
cmake -DACTION=BUILD -DTARGET=polycpp -DCMAKE_BUILD_TYPE=Release -P build.cmake
```

## CI release artefacts

Each platform workflow (`linux-shared.yaml`, `macos-shared.yaml`, `windows-shared.yaml`) runs
`LLVM` (cached), `CONFIGURE`, per-target `BUILD`, then `DIST` → `Package dist` → `Upload dist` →
`CHECK`. The packaged dist is uploaded as a GitHub Actions artefact named
`polyregion-${platform}-${arch}-${build_type}` (90-day retention, not a release).

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
POLYINVOKE_DISABLE_BACKENDS=ZE polyfc-tests          # drop Intel Level Zero
POLYINVOKE_DISABLE_BACKENDS=HIP polycpp-tests        # drop ROCm/HIP
```

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
