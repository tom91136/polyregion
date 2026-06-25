#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
  #define _DARWIN_C_SOURCE
#endif

#include "test_utils.h"

#include <cctype>

#if !defined(_WIN32)
  #include <fcntl.h>
  #include <unistd.h>

  #include <sys/file.h>
#endif

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/env_keys.h"
#include "polyregion/llvm_utils.hpp"

#include "polytest/profile.hpp"

#ifndef POLYREGION_TEST_PROFILE_DIR
  #define POLYREGION_TEST_PROFILE_DIR ""
#endif

using namespace aspartame;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::RequireFailed;
using polyregion::polytest::cases::SkipRequested;
using polyregion::polytest::cases::Task;

namespace polyregion::test_utils {

const DeviceSkip skipHasSpirv = [](invoke::Backend, invoke::Device &d) { return d.features() | contains("spirv_kernel"); };
const DeviceSkip skipNoSpirv = [](invoke::Backend, invoke::Device &d) { return !(d.features() | contains("spirv_kernel")); };
const DeviceSkip skipNoFp64 = [](invoke::Backend, invoke::Device &d) { return !(d.features() | contains("fp64")); };

ImageGroup findTestImage(const ImageGroups &archToImage, const invoke::Backend &backend, const std::vector<std::string> &features) {
  const auto sortedFeatures = features ^ sort();

  // XXX vulkan: every entry is a separate kernel module that must be loaded together (singleEntryPerModule).
  if (backend == invoke::Backend::Vulkan) {
    ImageGroup out;
    out.reserve(archToImage.size());
    for (auto &[module_, image] : archToImage)
      out.emplace_back(module_, std::string(image.begin(), image.end()));
    return out;
  }

  if (auto direct = sortedFeatures ^ collect_first([&](auto &feature) {
                      return archToImage ^ get_maybe(feature) //
                             ^ map([&](auto &x) { return ImageGroup{{feature, std::string(x.begin(), x.end())}}; });
                    }))
    return *direct;

  for (auto &[arch, image] : archToImage) {
#if defined(__x86_64__) || defined(_M_X64)
    auto archType = llvm::Triple::ArchType::x86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
    auto archType = llvm::Triple::ArchType::aarch64;
#elif defined(__arm__) || defined(_M_ARM)
    auto archType = llvm::Triple::ArchType::arm;
#else
  #error "Unsupported architecture"
#endif
    std::vector<std::string> required;
    llvm_shared::collectCPUFeatures(arch, archType, required);
    if (required ^ forall([&](auto &r) { return sortedFeatures ^ contains(r); })) return {{arch, std::string(image.begin(), image.end())}};
  }

  if (const auto x = archToImage ^ get_maybe(std::string{})) return {{"", std::string(x->begin(), x->end())}};

  return {};
}

int runOnTarget(invoke::Backend backend, std::string_view arch, const std::vector<std::string> &requiredFeatures, const ImageGroups &images,
                const DeviceRunner &runner, const DeviceSkip &skip) {
  Context ctx;

#if !defined(_WIN32)
  // The gfx1036 (2-CU Raphael iGPU) silently miscomputes a contiguous region of its output when any
  // other GPU work runs concurrently. The wrong data is produced device-side during the compute kernels
  // and is stably resident in device memory (verified: the input H2D lands intact, two back-to-back
  // readbacks agree, and the driver logs no fault) - it is not a dropped copy or a host-side readback
  // race, and an explicit per-kernel flush does not prevent it. The iGPU shares the system-memory
  // subsystem with everything else, and two concurrent stressors reproduce the fault:
  //   1. a concurrent KFD op - device enumeration, runtime init, or teardown (the /dev/kfd close that
  //      frees its 2 SDMA queues). Foreign OpenCL/Vulkan tasks reach this through their ICD loader, which
  //      dlopens the AMD ICD (amdocl64, RADV) and probes /dev/kfd even when targeting another vendor.
  //   2. raw memory-subsystem contention - a discrete GPU (e.g. CUDA) running heavy host<->device DMA,
  //      with no KFD op involved at all.
  // The fragility is in AMD hardware/firmware and cannot be fixed from here; the only lever we hold is
  // the concurrency that triggers it, so the iGPU dispatch must run exclusively against all other GPU
  // work. A discrete GPU with its own VRAM and memory controller does not share this path, which is why
  // no other vendor is affected.
  //
  // A reader-writer flock, taken before any GPU access and held past /dev/kfd close (fd relocated above
  // it, leaked, so it releases after exit_files frees the SDMA queues). The fragile AMD GPU task is the
  // exclusive WRITER (one iGPU dispatch at a time, with nothing else on the GPU). Every other GPU task -
  // CUDA, NVIDIA/Intel OpenCL/Vulkan, Level Zero - is a shared READER: readers run concurrently with
  // each other (full per-device parallelism across the discrete GPUs) but never alongside an AMD
  // dispatch. Host/object backends touch no GPU and take no lock.
  {
    std::string a(arch);
    for (auto &ch : a)
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    const bool amdArch = a == "amd" || a.find("gfx") != std::string::npos || a.find("radeon") != std::string::npos;
    const bool icdBackend = backend == invoke::Backend::OpenCL || backend == invoke::Backend::Vulkan;
    const bool gpuBackend = backend != invoke::Backend::RelocatableObject && backend != invoke::Backend::SharedObject;
    const bool writer = backend == invoke::Backend::HSA || backend == invoke::Backend::HIP || (icdBackend && amdArch);
    const bool reader = gpuBackend && !writer;
    if (writer || reader) {
      [[maybe_unused]] static const int coarseLock = [&] {
        int fd = ::open("/tmp/polyinvoke-gpu-coarse-amd.lock", O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) return -1;
        if (::flock(fd, writer ? LOCK_EX : LOCK_SH) != 0) {
          ::close(fd);
          return -1;
        }
        if (const int hi = ::fcntl(fd, F_DUPFD_CLOEXEC, 900); hi >= 0) {
          ::close(fd);
          fd = hi;
        }
        return fd;
      }();
    }
  }
#endif

  // leaked so the flock is never released before process exit: a scope release runs before exit_files
  // closes /dev/kfd, re-opening the teardown-handoff window on gfx1036 (see device_lock.cpp)
  auto &lock = *new std::optional<invoke::DeviceLock>();
  try {
    auto platform = invoke::Platform::maybe(backend);
    if (!platform) return 77;
    const bool archAsFeature = backend != invoke::Backend::RelocatableObject && backend != invoke::Backend::SharedObject;
    std::unique_ptr<invoke::Device> device;
    for (auto &d : platform->enumerate()) {
      const auto features = d->features();
      const auto hasFeature = [&](const std::string_view f) { return features ^ exists([&](auto &x) { return x == f; }); };
      if (!(requiredFeatures ^ forall(hasFeature))) continue;
      if (archAsFeature && !arch.empty() && arch != "*" && !hasFeature(arch)) continue;
      if (skip(backend, *d)) continue;
      device = std::move(d);
      break;
    }
    if (!device) return 77;
    lock.emplace(device->physicalDevice());
    auto imageGroup = findTestImage(images, backend, device->features());
    if (imageGroup.empty()) return 77;
    runner(ctx, backend, *platform, *device, imageGroup);
  } catch (const SkipRequested &) {
    return 77;
  } catch (const RequireFailed &) {
    return 1;
  }
  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverMatrix(const std::vector<Suite> &suites) {
  const auto targets = polytest::resolveTestTargets(POLYREGION_TEST_PROFILE_DIR, env::PolyinvokeTestTargets);
  return suites ^ flat_map([&](const Suite &s) {
           return s.backends ^ flat_map([&](const invoke::Backend b) {
                    return targets ^ collect([&](auto &t) -> std::optional<Task> {
                             if (t.spec.runtime != b) return {};
                             const std::string arch = t.arch;
                             const std::vector<std::string> required(t.spec.requiredDeviceFeatures.begin(),
                                                                     t.spec.requiredDeviceFeatures.end());
                             // XXX Key on the canonical spec name, not the Backend enum: opencl1_1 and spirv64_kernel both map
                             // to Backend::OpenCL and would collide on the same arch (e.g. both `@intel`).
                             return Task{fmt::format("{}-{}-{}", s.name, std::string(t.spec.canonical),
                                                     polytest::cases::sanitiseId(arch.empty() ? "any" : arch)),
                                         (b == invoke::Backend::RelocatableObject || b == invoke::Backend::SharedObject) ? "" : "device",
                                         [b, arch, required, images = s.images, runner = s.runner, skip = s.skip]() -> int {
                                           return runOnTarget(b, arch, required, *images, runner, skip);
                                         }};
                           });
                  });
         });
}

} // namespace polyregion::test_utils
