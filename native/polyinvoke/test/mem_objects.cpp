#include <stdexcept>
#include <string>

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

int runMemoryObjectsReserve() {
  Context ctx;
  MemoryObjects<int> objects;

  constexpr size_t reserved = 1 << 20;
  const auto grown = objects.reserve(reserved, 4096, 1);
  POLYTEST_CHECK(ctx, grown != 0);
  POLYTEST_CHECK(ctx, objects.reservedSize(grown) == reserved);
  POLYTEST_CHECK(ctx, objects.query(grown + 4096)->remaining == 0);
  POLYTEST_CHECK_S(ctx, !objects.query(grown + 4097), "a pointer past the committed size resolved");
  POLYTEST_CHECK(ctx, objects.query(grown + 100)->remaining == 4096 - 100);

  const auto after = objects.malloc(16, 2);
  POLYTEST_CHECK_S(ctx, after >= grown + reserved, "the next handle {} lies inside the reservation [{}, {})", after, grown,
                   grown + reserved);
  POLYTEST_CHECK(ctx, !objects.insert(grown + reserved / 2, 16, 3));

  objects.commit(grown, 8192, 4);
  POLYTEST_CHECK(ctx, objects.query(grown + 4097)->remaining == 8192 - 4097);
  POLYTEST_CHECK(ctx, objects.query(grown)->value == 4);
  POLYTEST_CHECK(ctx, objects.query(after)->value == 2);

  objects.commit(grown, reserved, 5);
  POLYTEST_CHECK(ctx, objects.query(grown + reserved)->remaining == 0);

  try {
    objects.commit(grown, reserved + 1, 6);
    POLYTEST_FAIL(ctx, "commit past the reservation succeeded");
  } catch (const std::length_error &error) {
    POLYTEST_CHECK_S(ctx, std::string(error.what()).find(std::to_string(reserved)) != std::string::npos,
                     "the error `{}` does not name the {} byte limit", error.what(), reserved);
  }
  try {
    objects.commit(grown, 4096, 7);
    POLYTEST_FAIL(ctx, "commit shrank an allocation");
  } catch (const std::invalid_argument &) {
  }
  try {
    objects.commit(after + 1, 16, 8);
    POLYTEST_FAIL(ctx, "commit of an unknown handle succeeded");
  } catch (const std::out_of_range &) {
  }
  try {
    (void)objects.reserve(16, 32, 9);
    POLYTEST_FAIL(ctx, "reserve committed more than it reserved");
  } catch (const std::invalid_argument &) {
  }
  POLYTEST_CHECK(ctx, objects.query(grown)->value == 5);
  POLYTEST_CHECK(ctx, objects.erase(grown));
  POLYTEST_CHECK(ctx, !objects.reservedSize(grown));

  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverAll() {
  return {Task{"memory-objects", "", &runMemoryObjects}, Task{"memory-objects-reserve", "", &runMemoryObjectsReserve}};
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
