#include "polyinvoke/runtime.h"

#include "polytest/test_case.hpp"

using polyregion::invoke::detail::MemoryObjects;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

int runMemoryObjects() {
  Context ctx;
  MemoryObjects<int> objects;

  const auto first = objects.malloc(16, 1);
  POLYTEST_CHECK_S(ctx, first != 0, "the first handle is {}, which no null check can tell apart from a null pointer", first);
  const auto second = objects.malloc(8, 2);
  POLYTEST_CHECK_S(ctx, second != first, "handle {} was handed out twice", second);

  POLYTEST_REQUIRE(ctx, objects.query(first).has_value());
  POLYTEST_REQUIRE(ctx, objects.query(second).has_value());
  POLYTEST_CHECK(ctx, objects.query(first)->value == 1);
  POLYTEST_CHECK(ctx, objects.query(second)->value == 2);
  POLYTEST_CHECK(ctx, !objects.query(0));

  const auto interior = objects.query(first + 7);
  POLYTEST_REQUIRE(ctx, interior.has_value());
  POLYTEST_CHECK(ctx, interior->value == 1);
  POLYTEST_CHECK(ctx, interior->offset == 7);
  POLYTEST_CHECK(ctx, interior->remaining == 9);

  const auto end = objects.query(first + 16);
  POLYTEST_REQUIRE(ctx, end.has_value());
  POLYTEST_CHECK(ctx, end->offset == 16);
  POLYTEST_CHECK(ctx, end->remaining == 0);
  POLYTEST_CHECK(ctx, !objects.query(first + 17));

  const auto range = objects.queryRange(first + 3, 4, 5);
  POLYTEST_REQUIRE(ctx, range.has_value());
  POLYTEST_CHECK(ctx, range->offset == 7);
  POLYTEST_CHECK(ctx, range->remaining == 9);
  POLYTEST_CHECK(ctx, !objects.queryRange(first + 3, 9, 5));
  POLYTEST_CHECK(ctx, !objects.queryRange(first + 3, 14, 0));

  constexpr uintptr_t external = 0x10000;
  POLYTEST_CHECK(ctx, objects.insert(external, 32, 4));
  POLYTEST_CHECK(ctx, !objects.insert(external + 16, 32, 5));
  POLYTEST_CHECK(ctx, !objects.insert(external - 16, 32, 5));
  POLYTEST_CHECK(ctx, objects.insert(external + 32, 16, 6));

  POLYTEST_CHECK(ctx, !objects.erase(second + 1));
  POLYTEST_CHECK(ctx, objects.query(second).has_value());

  POLYTEST_CHECK(ctx, objects.erase(first));
  POLYTEST_CHECK(ctx, !objects.query(first));
  POLYTEST_CHECK(ctx, objects.query(second)->value == 2);
  POLYTEST_CHECK_S(ctx, objects.malloc(4, 3) != 0, "a handle handed out after an erase is indistinguishable from a null pointer");

  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverAll() { return {Task{"memory-objects", "", &runMemoryObjects}}; }

} // namespace

POLYTEST_DISCOVER(discoverAll)
