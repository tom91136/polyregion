#pragma once

#include "polyrt/runtime.h"

#include <cassert>
#include <cstddef>
#include <cstring>

using polyregion::runtime::Access;
using polyregion::runtime::Device;
using polyregion::runtime::DeviceQueue;
using polyregion::runtime::ModuleFormat;
using polyregion::runtime::Platform;
using polyregion::runtime::PlatformKind;

extern "C" struct RuntimeStructMember {
  const char *name;
  size_t offsetInBytes, sizeInBytes;
  bool isPtr;
  int64_t ptrTypeIdx; // -1 = no ptr
};

extern "C" struct RuntimeStruct {
  const char *name;
  bool exported;
  size_t memberCount;
  RuntimeStructMember *members;
};

extern "C" struct RuntimeKernelObject {
  PlatformKind kind;
  ModuleFormat format;
  const char **features;
  size_t imageLength;
  const unsigned char *image;
};

extern "C" struct RuntimeKernelBundle {
  const char *moduleName;

  size_t objectCount;
  RuntimeKernelObject *objects;

  size_t structCount;
  RuntimeStruct *structs;

  const char *metadata;
};

static constexpr const char *__polyregion_prefix = "PolySTL"; // NOLINT(*-reserved-identifier)

#ifdef POLYSTL_LOG
  #error Trace already defined
#else

  #define POLYSTL_LOG(fmt, ...) std::fprintf(stderr, "[%s] " fmt "\n", __polyregion_prefix, __VA_ARGS__)
//  #define POLYSTL_LOG(fmt, ...)

#endif

POLYREGION_EXPORT extern std::unique_ptr<Platform> __polyregion_selected_platform; // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern std::unique_ptr<Device> __polyregion_selected_device;     // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern std::unique_ptr<DeviceQueue> __polyregion_selected_queue; // NOLINT(*-reserved-identifier)

POLYREGION_EXPORT extern "C" void __polyregion_initialise_runtime();              // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern "C" bool __polyregion_platform_kind(PlatformKind &kind); // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern "C" bool __polyregion_load_kernel_object(const char *moduleName,
                                                                  const RuntimeKernelObject &object); // NOLINT(*-reserved-identifier)
POLYREGION_EXPORT extern "C" void __polyregion_dispatch_hostthreaded(                                 // NOLINT(*-reserved-identifier)
    size_t global, void *functorData, const char *moduleId);
POLYREGION_EXPORT extern "C" void __polyregion_dispatch_managed( // NOLINT(*-reserved-identifier)
    size_t global, size_t local,                                 //
    size_t localMemBytes,                                        //
    size_t functorDataSize,                                      //
    const void *functorData,                                     //
    const char *moduleId);

[[nodiscard]] uint32_t __polyregion_builtin_gpu_global_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_global_size(uint32_t); // NOLINT(*-reserved-identifier)

[[nodiscard]] uint32_t __polyregion_builtin_gpu_group_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_group_size(uint32_t); // NOLINT(*-reserved-identifier)

[[nodiscard]] uint32_t __polyregion_builtin_gpu_local_idx(uint32_t);  // NOLINT(*-reserved-identifier)
[[nodiscard]] uint32_t __polyregion_builtin_gpu_local_size(uint32_t); // NOLINT(*-reserved-identifier)

void __polyregion_builtin_gpu_barrier_global(); // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_barrier_local();  // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_barrier_all();    // NOLINT(*-reserved-identifier)

void __polyregion_builtin_gpu_fence_global(); // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_fence_local();  // NOLINT(*-reserved-identifier)
void __polyregion_builtin_gpu_fence_all();    // NOLINT(*-reserved-identifier)

template <typename T> class __polyregion_local {
  T *ptr;
  __polyregion_local(T *ptr) : ptr(ptr) { static_assert(sizeof(__polyregion_local<void *>) == sizeof(void *)); }

public:
  T &operator[](size_t idx) const { return ptr[idx]; }
};

template <PlatformKind Kind, typename F>
const RuntimeKernelBundle &__polyregion_offload__([[maybe_unused]] F) { // NOLINT(*-reserved-identifier)

  //   [[maybe_unused]] const char *__moduleName;       // NOLINT(*-reserved-identifier)
  //   [[maybe_unused]] const char *__metadata;         // NOLINT(*-reserved-identifier)
  //   [[maybe_unused]] size_t __objectSize;            // NOLINT(*-reserved-identifier)
  //   [[maybe_unused]] uint8_t *__formats;             // NOLINT(*-reserved-identifier)
  //   [[maybe_unused]] uint8_t *__kinds;               // NOLINT(*-reserved-identifier)
  //   [[maybe_unused]] const char ***__features;       // NOLINT(*-reserved-identifier)
  //   [[maybe_unused]] size_t *__imageSizes;           // NOLINT(*-reserved-identifier)
  //   [[maybe_unused]] const unsigned char **__images; // NOLINT(*-reserved-identifier)
  //
  // __insert_point:;
  //
  //   const static RuntimeKernelBundle __bundle = // NOLINT(*-reserved-identifier)
  //       {
  //           __moduleName, //
  //           __metadata,   //
  //           __objectSize, //
  //           __formats,    //
  //           __kinds,      //
  //           __features,   //
  //           __imageSizes,
  //           __images //
  //       };
  //   return __bundle;
  assert(false && "impl not replaced");
}

extern "C" inline __attribute__((used)) void *__polyregion_malloc(size_t size) { // NOLINT(*-reserved-identifier)
  __polyregion_initialise_runtime();
  if (!__polyregion_selected_platform || !__polyregion_selected_device || !__polyregion_selected_queue) {
    POLYSTL_LOG("No device/queue in %s", __func__);
    return nullptr;
  }
  if (auto ptr = __polyregion_selected_device->mallocShared(size, Access::RW); ptr) {
    return *ptr;
  } else {
    POLYSTL_LOG("No USM support in %s", __func__);
    return nullptr;
  }
}

extern "C" inline __attribute__((used)) void __polyregion_free(void *ptr) { // NOLINT(*-reserved-identifier)
  __polyregion_initialise_runtime();
  if (!__polyregion_selected_platform || !__polyregion_selected_device || !__polyregion_selected_queue) {
    POLYSTL_LOG("No device/queue in %s", __func__);
  }
  __polyregion_selected_device->freeShared(ptr);
}
extern "C" inline __attribute__((used)) void *__polyregion_operator_new(size_t size) { // NOLINT(*-reserved-identifier)
  return __polyregion_malloc(size);
}
extern "C" inline __attribute__((used)) void __polyregion_operator_delete(void *ptr) { // NOLINT(*-reserved-identifier)
  __polyregion_free(ptr);
}
extern "C" inline __attribute__((used)) void __polyregion_operator_delete_sized(void *ptr, size_t size) { // NOLINT(*-reserved-identifier)
  __polyregion_free(ptr);
}
extern "C" inline __attribute__((used)) void *__polyregion_aligned_alloc(size_t alignment, size_t size) { // NOLINT(*-reserved-identifier)
  // TODO actually align it
  return __polyregion_malloc(size);
}