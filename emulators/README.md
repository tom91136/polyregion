# Emulators

Fully relocatable CPU implementations of offload runtimes:

- **gpuocelot** - CUDA / PTX (`libcuda.so.1`, `nvcuda.dll` on Windows)
- **rusticl** (Mesa 26.1.1, llvmpipe) + **PoCL** - OpenCL source + SPIRV, behind an `ocl-icd` loader (`libOpenCL.so.1`)
- **lavapipe** (Mesa 26.1.1) + Vulkan-Loader - Vulkan (`libvulkan_lvp.so`, `libvulkan.so.1`)

## Recipes

```bash
just build-emulators             # build the Dockerfile -> emulators/out (gated on the vecadd smoke)
just check-emulators             # rerun the vecadd smoke against emulators/out
just test-native-with-emulators  # run the native suite on the emulators (extra ctest args pass through)
source emulators/out/env.sh      # use the bundle directly (LD_LIBRARY_PATH + VK_DRIVER_FILES)
```

`gpuocelot.patch` is the polyregion fork's diff vs upstream `b16039dc`. `gpuocelot.windows.patch`
layers MSVC/clang-cl support on top (built by `windows.bat`, staged as `nvcuda.dll`).
