# Native build

All build steps go through `cmake -P build.cmake -DACTION=<action>`.

## Actions

| Action | What it does |
|---|---|
| `LLVM` | Configure + build + install LLVM/Clang/LLD/Flang/MLIR into `llvm-${BUILD_TYPE}-${ARCH}-${VARIANT}/`. Slow (hours, first time). |
| `CONFIGURE` | Configure polyregion build into `build-${platform}-${ARCH}-${VARIANT}/`. |
| `BUILD` | Build a target. Requires `-DTARGET=<name>` (use `all` for everything). |
| `DIST` | Install polyregion into `polyregion-${BUILD_TYPE}-${ARCH}-${VARIANT}-dist/`. Builds install deps as needed. |
| `CHECK` | Run the dist sanity check (compiles hello/offload programs through `clang`/`flang-new`/`polycpp`/`polyfc`, verifies output binaries don't depend on shipped DSOs, compile-only checks for cuda/hsa/spirv/metal/c11). |

## Common options

| Option | Default | Notes |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=` | (required) | `Release` / `Debug` |
| `-DARCH=` | host arch | `x86_64`, `aarch64`, ... |
| `-DTARGET=` | — | for `BUILD` only |
| `-DCMAKE_SYSROOT=` | — | required for cross builds; pass `/` when configuring locally without a sysroot |

## Environment

| Variable | Default | Effect |
|---|---|---|
| `POLYREGION_LLVM_DYLIB` | `ON` | `OFF` builds a static dist (no `libLLVM.so` / `libMLIR.so` / `libclang-cpp.so`). Affects both `LLVM` and polyregion configure/build/dist; output dir gets `-static` suffix instead of `-dylib`. |
| `VCPKG_ROOT` | (required) | vcpkg checkout |
| `JAVA_HOME` | (required for JNI bindings) | JDK install root |

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

Each platform workflow (`linux-shared.yaml`, `macos-shared.yaml`, `windows-shared.yaml`) runs `LLVM` (cached), `CONFIGURE`, per-target `BUILD`, then `DIST` → `Package dist` → `Upload dist` → `CHECK`. The packaged dist is uploaded as a GitHub Actions artefact named `polyregion-${platform}-${arch}-${build_type}` (90-day retention, not a release).
