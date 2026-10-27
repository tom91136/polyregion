#include <vector>

#include "fmt/format.h"

#include "polyinvoke/device_lock.h"
#include "polyregion/concurrency_utils.hpp"

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

std::vector<int32_t> launch(Platform &platform, Device &device, const char *kernel, const Shape shape, const size_t perThread) {
  auto q = device.createQueue(std::chrono::seconds(30));
  const size_t count = shape.groups * shape.lanes * perThread;
  const std::vector<int32_t> zeros(count, 0);
  auto out = device.mallocDevice(count * sizeof(int32_t), Access::RW);
  waitAll([&](auto &h) { q->enqueueHostToDeviceAsync(zeros.data(), out, 0, count * sizeof(int32_t), h); });
  ArgBuffer buffer;
  prependTidArg(platform, buffer);
  buffer.append({{Type::Ptr, &out}, {Type::Void, {}}});
  waitAll([&](auto &h) { q->enqueueInvokeAsync("module", kernel, buffer, Policy{{shape.groups, 1, 1}, {{{shape.lanes, 1, 1}, 0}}}, h); });
  std::vector<int32_t> result(count);
  waitAll([&](auto &h) { q->enqueueDeviceToHostAsync(out, 0, result.data(), count * sizeof(int32_t), h); });
  device.freeDevice(out);
  return result;
}

void runGroups(Context &ctx, Backend, Platform &platform, Device &device, const ImageGroup &imageGroup) {
  POLYTEST_REQUIRE_S(ctx, imageGroup.size() == 1, "expected 1 image group, got {}", imageGroup.size());
  device.loadModule("module", imageGroup[0].second);

  for (const auto shape : {Shape{1, 64}, Shape{3, 64}, Shape{4, 256}, Shape{2, 1024}}) {
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

std::vector<Task> discoverAll() {
  return discoverMatrix({
      {"groups-cpu-reloc", generated::cpu_reloc::groups, {Backend::RelocatableObject}, &runGroups},
  });
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
