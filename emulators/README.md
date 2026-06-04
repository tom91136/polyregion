# Emulators

Fully relocatable CPU implementations of offload runtimes:

- **gpuocelot** - CUDA / PTX (`libcuda.so.1`)
- **PoCL**  - OpenCL source + SPIRV (`libOpenCL.so.1`)
- **lavapipe** + Vulkan-Loader - Vulkan (`libvulkan_lvp.so`, `libvulkan.so.1`)

## Recipes

```bash
just build-emulators             # build the Dockerfile -> emulators/out
just test-native-with-emulators  # run the native suite on the emulators (extra ctest args pass through)
source emulators/out/env.sh      # use the bundle directly (LD_LIBRARY_PATH + VK_DRIVER_FILES)
```

`gpuocelot.patch` is the polyregion fork's diff vs upstream `b16039dc`.
