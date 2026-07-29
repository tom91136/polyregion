#include <optional>

#include "polyinvoke/runtime.h"

#include "polytest/test_case.hpp"

using polyregion::invoke::detail::MemoryObjects;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

int runMemoryObjects() {
  Context ctx;
  MemoryObjects<int> objects;

  const auto first = objects.malloc(1);
  POLYTEST_CHECK_S(ctx, first != 0, "the first handle is {}, which no null check can tell apart from a null pointer", first);
  const auto second = objects.malloc(2);
  POLYTEST_CHECK_S(ctx, second != first, "handle {} was handed out twice", second);

  POLYTEST_CHECK(ctx, objects.query(first) == std::optional{1});
  POLYTEST_CHECK(ctx, objects.query(second) == std::optional{2});
  POLYTEST_CHECK(ctx, !objects.query(0));

  objects.erase(first);
  POLYTEST_CHECK(ctx, !objects.query(first));
  POLYTEST_CHECK(ctx, objects.query(second) == std::optional{2});
  POLYTEST_CHECK_S(ctx, objects.malloc(3) != 0, "a handle handed out after an erase is indistinguishable from a null pointer");

  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverAll() { return {Task{"memory-objects", "", &runMemoryObjects}}; }

} // namespace

POLYTEST_DISCOVER(discoverAll)
