#include <chrono>

#include "polyinvoke/hip_platform.h"

#include "polytest/test_case.hpp"

using namespace std::chrono_literals;
using polyregion::invoke::hip::details::pollStreamUntil;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

int runHipStreamPolling() {
  Context ctx;
  const auto now = std::chrono::steady_clock::time_point{};
  size_t queries = 0;
  size_t pauses = 0;
  const auto timeout = pollStreamUntil(
      now,
      [&] {
        ++queries;
        return hipErrorNotReady;
      },
      [&] { return now; }, [&] { ++pauses; });

  POLYTEST_CHECK(ctx, !timeout);
  POLYTEST_CHECK(ctx, queries == 1);
  POLYTEST_CHECK(ctx, pauses == 0);

  auto clock = std::chrono::steady_clock::time_point{};
  queries = 0;
  pauses = 0;
  const auto completed = pollStreamUntil(
      clock + 2ms,
      [&] {
        ++queries;
        return queries == 2 ? hipSuccess : hipErrorNotReady;
      },
      [&] { return clock; },
      [&] {
        ++pauses;
        clock += 1ms;
      });

  POLYTEST_REQUIRE(ctx, completed.has_value());
  POLYTEST_CHECK(ctx, *completed == hipSuccess);
  POLYTEST_CHECK(ctx, queries == 2);
  POLYTEST_CHECK(ctx, pauses == 1);
  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverAll() { return {Task{"hip-stream-polling", "", &runHipStreamPolling}}; }

} // namespace

POLYTEST_DISCOVER(discoverAll)
