#include "polyinvoke/cl_platform.h"

#include "polytest/test_case.hpp"

using namespace polyregion::invoke;
using namespace polyregion::invoke::cl;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace {

void checkDim(Context &ctx, const Dim3 &actual, const Dim3 &expected, const char *what) {
  POLYTEST_CHECK_S(ctx, actual.x == expected.x && actual.y == expected.y && actual.z == expected.z, "{} was {}x{}x{}, expected {}x{}x{}",
                   what, actual.x, actual.y, actual.z, expected.x, expected.y, expected.z);
}

int runLaunchDimensions() {
  Context ctx;
  const Dim3 groups{7, 5, 3};
  const Dim3 local{256, 2, 1};

  const auto initial = details::launchDimensions(groups, local);
  checkDim(ctx, initial.local, local, "initial local size");
  checkDim(ctx, initial.global, Dim3{1792, 10, 3}, "initial global size");

  POLYTEST_CHECK(ctx, !details::retryLaunchDimensions(CL_SUCCESS, groups, local, 128));
  POLYTEST_CHECK(ctx, !details::retryLaunchDimensions(CL_OUT_OF_RESOURCES, groups, local, 128));
  POLYTEST_CHECK(ctx, !details::retryLaunchDimensions(CL_INVALID_WORK_GROUP_SIZE, groups, local, 512));

  const auto retry = details::retryLaunchDimensions(CL_INVALID_WORK_GROUP_SIZE, groups, local, 128);
  POLYTEST_REQUIRE(ctx, retry.has_value());
  checkDim(ctx, retry->local, Dim3{64, 2, 1}, "retry local size");
  checkDim(ctx, retry->global, Dim3{448, 10, 3}, "retry global size");
  POLYTEST_CHECK(ctx, retry->global.x / retry->local.x == groups.x);
  POLYTEST_CHECK(ctx, retry->global.y / retry->local.y == groups.y);
  POLYTEST_CHECK(ctx, retry->global.z / retry->local.z == groups.z);

  return ctx.failed ? 1 : 0;
}

int runErrorStrings() {
  Context ctx;
  POLYTEST_CHECK(ctx, details::errorString(CL_INVALID_VALUE) == "CL_INVALID_VALUE");
  POLYTEST_CHECK(ctx, details::errorString(-1001) == "CL_PLATFORM_NOT_FOUND_KHR");
  POLYTEST_CHECK(ctx, details::errorString(-9999) == "unknown OpenCL error (-9999)");
  POLYTEST_CHECK(ctx, details::errorString(9999) == "unknown OpenCL error (9999)");
  return ctx.failed ? 1 : 0;
}

int runSvmOwnership() {
  Context ctx;
  using Ownership = details::SVMTracker::Ownership;
  auto *base = reinterpret_cast<void *>(uintptr_t{0x10000});
  auto *interior = reinterpret_cast<void *>(uintptr_t{0x10020});

  details::SVMTracker tracker;
  tracker.track(base, 64);
  POLYTEST_CHECK(ctx, tracker.ownership(base) == Ownership::Device);
  POLYTEST_CHECK(ctx, !tracker.ownership(reinterpret_cast<void *>(uintptr_t{0x20000})));

  size_t maps = 0;
  const auto map = [&](void *ptr, size_t size) {
    ++maps;
    POLYTEST_CHECK(ctx, ptr == base);
    POLYTEST_CHECK(ctx, size == 64);
    return CL_SUCCESS;
  };
  POLYTEST_CHECK(ctx, tracker.mapForHost(interior, map) == CL_SUCCESS);
  POLYTEST_CHECK(ctx, tracker.mapForHost(base, map) == CL_SUCCESS);
  POLYTEST_CHECK(ctx, maps == 1);
  POLYTEST_CHECK(ctx, tracker.ownership(base) == Ownership::Host);

  size_t unmaps = 0;
  POLYTEST_CHECK(ctx, tracker.unmapAllForDevice([&](void *ptr) {
    ++unmaps;
    POLYTEST_CHECK(ctx, ptr == base);
    return CL_SUCCESS;
  }) == CL_SUCCESS);
  POLYTEST_CHECK(ctx, tracker.unmapAllForDevice([&](void *) {
    ++unmaps;
    return CL_SUCCESS;
  }) == CL_SUCCESS);
  POLYTEST_CHECK(ctx, unmaps == 1);
  POLYTEST_CHECK(ctx, tracker.ownership(base) == Ownership::Device);

  POLYTEST_CHECK(ctx, tracker.mapForHost(base, map) == CL_SUCCESS);
  tracker.untrack(base);
  tracker.track(base, 128);
  POLYTEST_CHECK(ctx, tracker.ownership(base) == Ownership::InheritedHost);
  POLYTEST_CHECK(ctx, tracker.unmapAllForDevice([](void *) { return CL_SUCCESS; }) == CL_SUCCESS);
  POLYTEST_CHECK(ctx, tracker.ownership(base) == Ownership::Device);
  POLYTEST_CHECK(ctx, !tracker.freeReleasesHostMap());

  POLYTEST_CHECK(ctx, tracker.mapForHost(base, [](void *, size_t) { return CL_SUCCESS; }) == CL_SUCCESS);
  POLYTEST_CHECK(ctx, tracker.unmapAllForDevice([](void *) { return CL_INVALID_VALUE; }) == CL_INVALID_VALUE);
  POLYTEST_CHECK(ctx, tracker.ownership(base) == Ownership::Host);
  POLYTEST_CHECK(ctx, tracker.unmapAllForDevice([](void *) { return CL_SUCCESS; }) == CL_SUCCESS);

  details::SVMTracker releasingTracker;
  releasingTracker.track(base, 64);
  POLYTEST_CHECK(ctx, releasingTracker.mapForHost(base, [](void *, size_t) { return CL_SUCCESS; }) == CL_SUCCESS);
  releasingTracker.untrack(base);
  releasingTracker.track(base, 64);
  POLYTEST_CHECK(ctx, releasingTracker.ownership(base) == Ownership::InheritedHost);
  POLYTEST_CHECK(ctx, releasingTracker.unmapAllForDevice([](void *) { return CL_INVALID_VALUE; }) == CL_SUCCESS);
  POLYTEST_CHECK(ctx, releasingTracker.ownership(base) == Ownership::Device);
  POLYTEST_CHECK(ctx, releasingTracker.freeReleasesHostMap());

  POLYTEST_CHECK(ctx, releasingTracker.mapForHost(base, [](void *, size_t) { return CL_SUCCESS; }) == CL_SUCCESS);
  releasingTracker.untrack(base);
  releasingTracker.track(base, 64);
  POLYTEST_CHECK(ctx, releasingTracker.ownership(base) == Ownership::Device);

  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverAll() {
  return {Task{"cl-launch-dimensions", "", &runLaunchDimensions}, Task{"cl-error-strings", "", &runErrorStrings},
          Task{"cl-svm-ownership", "", &runSvmOwnership}};
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
