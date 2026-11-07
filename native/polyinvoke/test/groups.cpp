#include <algorithm>
#include <vector>

#include "fmt/format.h"

#include "polyinvoke/device_lock.h"
#include "polyregion/concurrency_utils.hpp"

#include "kernels/generated_cpu_reloc_group_abi.hpp"
#include "kernels/generated_cpu_reloc_groups.hpp"
#include "test_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::test_utils;
using namespace polyregion::concurrency_utils;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

struct Shape {
  size_t groups, lanes;
};

std::vector<int32_t> launch(Platform &platform, Device &device, const char *kernel, const Shape shape, const size_t perThread,
                            const bool withLocal = true) {
  auto q = device.createQueue(std::chrono::seconds(30));
  const size_t count = shape.groups * shape.lanes * perThread;
  const std::vector<int32_t> zeros(count, 0);
  auto out = device.mallocDevice(count * sizeof(int32_t), Access::RW);
  waitAll([&](auto &h) { q->enqueueHostToDeviceAsync(zeros.data(), out, 0, count * sizeof(int32_t), h); });
  ArgBuffer buffer;
  prependTidArg(platform, buffer);
  buffer.append({{Type::Ptr, &out}, {Type::Void, {}}});
  const Policy policy{{shape.groups, 1, 1}, withLocal ? std::optional{std::pair{Dim3{shape.lanes, 1, 1}, size_t{0}}} : std::nullopt};
  waitAll([&](auto &h) { q->enqueueInvokeAsync("module", kernel, buffer, policy, h); });
  std::vector<int32_t> result(count);
  waitAll([&](auto &h) { q->enqueueDeviceToHostAsync(out, 0, result.data(), count * sizeof(int32_t), h); });
  device.freeDevice(out);
  return result;
}

void runGroups(Context &ctx, Backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  POLYTEST_REQUIRE_S(ctx, imageGroup.size() == 1, "expected 1 image group, got {}", imageGroup.size());
  device.loadModule("module", imageGroup[0].second);

  const size_t maxLanes = device.maxThreadsPerBlock();
  for (auto shape : {Shape{1, 64}, Shape{3, 64}, Shape{4, 256}, Shape{2, 1024}}) {
    shape.lanes = std::min(shape.lanes, maxLanes);
    const auto reversed = launch(platform, device, "groups_reverse", shape, 1);
    for (size_t g = 0; g < shape.groups; ++g)
      for (size_t l = 0; l < shape.lanes; ++l) {
        const int32_t expected = static_cast<int32_t>(2 * g * shape.lanes + shape.lanes);
        POLYTEST_CHECK_S(ctx, reversed[g * shape.lanes + l] == expected, "reverse {}x{} group {} lane {}: got {}, expected {}",
                         shape.groups, shape.lanes, g, l, reversed[g * shape.lanes + l], expected);
      }

    const auto early = launch(platform, device, "groups_early_exit", shape, 1);
    for (size_t g = 0; g < shape.groups; ++g)
      for (size_t l = 0; l < shape.lanes; ++l) {
        const int32_t expected = l % 2 == 1 ? 0 : static_cast<int32_t>(g * shape.lanes + l + 2);
        POLYTEST_CHECK_S(ctx, early[g * shape.lanes + l] == expected, "early exit {}x{} group {} lane {}: got {}, expected {}",
                         shape.groups, shape.lanes, g, l, early[g * shape.lanes + l], expected);
      }

    const auto ids = launch(platform, device, "groups_ids", shape, 4);
    for (size_t g = 0; g < shape.groups; ++g)
      for (size_t l = 0; l < shape.lanes; ++l) {
        const auto *at = &ids[(g * shape.lanes + l) * 4];
        POLYTEST_CHECK_S(ctx, at[0] == int32_t(g) && at[1] == int32_t(shape.groups) && at[2] == int32_t(l) && at[3] == int32_t(shape.lanes),
                         "ids {}x{} group {} lane {}: got [{}, {}, {}, {}]", shape.groups, shape.lanes, g, l, at[0], at[1], at[2], at[3]);
      }
  }
}

void runGroupAbi(Context &ctx, Backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  POLYTEST_REQUIRE_S(ctx, imageGroup.size() == 1, "expected 1 image group, got {}", imageGroup.size());
  device.loadModule("module", imageGroup[0].second);

  const size_t manyGroups = 2 * device.computeUnits() + 1;
  for (const auto shape : {Shape{1, 64}, Shape{3, 64}, Shape{4, 256}, Shape{2, 1024}, Shape{manyGroups, 64}, Shape{5, 1}}) {
    const auto ids = launch(platform, device, "group_abi_ids", shape, 4);
    for (size_t g = 0; g < shape.groups; ++g)
      for (size_t l = 0; l < shape.lanes; ++l) {
        const auto *at = &ids[(g * shape.lanes + l) * 4];
        POLYTEST_CHECK_S(ctx, at[0] == int32_t(g) && at[1] == int32_t(shape.groups) && at[2] == int32_t(l) && at[3] == int32_t(shape.lanes),
                         "group abi ids {}x{} group {} lane {}: got [{}, {}, {}, {}]", shape.groups, shape.lanes, g, l, at[0], at[1], at[2],
                         at[3]);
      }

    const auto calls = launch(platform, device, "group_abi_calls", shape, 1);
    for (size_t g = 0; g < shape.groups; ++g)
      for (size_t l = 0; l < shape.lanes; ++l) {
        const int32_t expected = l == 0 ? 1 : 0;
        POLYTEST_CHECK_S(ctx, calls[g * shape.lanes + l] == expected, "group abi calls {}x{} group {} lane {}: got {}, expected {}",
                         shape.groups, shape.lanes, g, l, calls[g * shape.lanes + l], expected);
      }

    const auto state = launch(platform, device, "group_abi_state", shape, 1);
    for (size_t g = 0; g < shape.groups; ++g)
      for (size_t l = 0; l < shape.lanes; ++l) {
        const int32_t expected = static_cast<int32_t>(g * shape.lanes + (shape.lanes - 1 - l) + 1);
        POLYTEST_CHECK_S(ctx, state[g * shape.lanes + l] == expected, "group abi state {}x{} group {} lane {}: got {}, expected {}",
                         shape.groups, shape.lanes, g, l, state[g * shape.lanes + l], expected);
      }

    const auto local = launch(platform, device, "group_abi_local", shape, 1);
    for (size_t g = 0; g < shape.groups; ++g)
      for (size_t l = 0; l < shape.lanes; ++l)
        POLYTEST_CHECK_S(ctx, local[g * shape.lanes + l] == 0, "group abi local {}x{} group {} lane {}: got {}, expected 0", shape.groups,
                         shape.lanes, g, l, local[g * shape.lanes + l]);
  }

  const Shape ungrouped{7, 1};
  const auto ids = launch(platform, device, "group_abi_ids", ungrouped, 4, false);
  for (size_t g = 0; g < ungrouped.groups; ++g) {
    const auto *at = &ids[g * 4];
    POLYTEST_CHECK_S(ctx, at[0] == int32_t(g) && at[1] == int32_t(ungrouped.groups) && at[2] == 0 && at[3] == 1,
                     "group abi ungrouped group {}: got [{}, {}, {}, {}]", g, at[0], at[1], at[2], at[3]);
  }
}

void runFlatAbi(Context &ctx, Backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  POLYTEST_REQUIRE_S(ctx, imageGroup.size() == 1, "expected 1 image group, got {}", imageGroup.size());
  device.loadModule("module", imageGroup[0].second);

  for (const auto shape : {Shape{3, 64}, Shape{2, 1024}, Shape{9, 1}}) {
    const auto flat = launch(platform, device, "group_abi_flat", shape, 2);
    for (size_t id = 0; id < shape.groups * shape.lanes; ++id)
      POLYTEST_CHECK_S(ctx, flat[id * 2] == 1 && flat[id * 2 + 1] == int32_t(id), "flat abi {}x{} id {}: got [{}, {}], expected [1, {}]",
                       shape.groups, shape.lanes, id, flat[id * 2], flat[id * 2 + 1], id);
  }
}

std::vector<Task> discoverAll() {
  return discoverMatrix({
      {"groups-cpu-reloc", generated::cpu_reloc::groups, {Backend::RelocatableObject}, &runGroups},
      {"group-abi-cpu-reloc", generated::cpu_reloc::group_abi, {Backend::RelocatableObject}, &runGroupAbi},
      {"flat-abi-cpu-reloc", generated::cpu_reloc::group_abi, {Backend::RelocatableObject}, &runFlatAbi},
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
