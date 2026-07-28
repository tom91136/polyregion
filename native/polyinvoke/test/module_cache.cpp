#include "polyinvoke/module_cache.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/device_lock.h"
#include "polyregion/concurrency_utils.hpp"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"
#include "polyregion/io.hpp"

#include "kernels/generated_opencl_source_fma.hpp"
#include "kernels/generated_opencl_spirv_fma.hpp"
#include "test_utils.h"

using namespace polyregion;
using namespace polyregion::invoke;
using namespace polyregion::test_utils;
using namespace polyregion::concurrency_utils;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::Task;

namespace fs = llvm::sys::fs;

namespace {

const std::vector<uint8_t> payload = {0x00, 0x01, 0x02, 0x03, 0x7f, 0x80, 0xfe, 0xff};

std::string freshCacheDir(Context &ctx) {
  llvm::SmallString<128> dir;
  if (fs::createUniqueDirectory("polyinvoke-module-cache", dir)) POLYTEST_FAIL(ctx, "cannot create a temporary cache directory");
  env::put(env::PolyregionCacheDir, dir.c_str(), true);
  return dir.str().str();
}

std::vector<std::pair<std::string, std::vector<uint8_t>>> entriesUnder(Context &ctx, const std::string &dir) {
  std::vector<std::pair<std::string, std::vector<uint8_t>>> out;
  std::error_code ec;
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
    if (it->type() != fs::file_type::regular_file) continue;
    if (fs::file_status status; fs::status(it->path(), status)) POLYTEST_FAIL(ctx, "cannot stat cache entry {}", it->path());
    out.emplace_back(it->path(), read_struct<uint8_t>(it->path()));
  }
  std::sort(out.begin(), out.end(), [](auto &l, auto &r) { return l.first < r.first; });
  return out;
}

void writeFile(const std::string &path, const std::vector<uint8_t> &bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void runEntry(Context &ctx) {
  const auto dir = freshCacheDir(ctx);
  const auto path = detail::moduleCachePath("test", "device-a", "image-1");
  POLYTEST_REQUIRE_S(ctx, !path.empty(), "no cache path under {}", dir);
  POLYTEST_CHECK_S(ctx, cache::read(path).empty(), "cold read of {} hit", path);

  cache::write(path, payload.data(), payload.size());
  POLYTEST_CHECK_S(ctx, cache::read(path) == payload, "warm read of {} did not return the published payload", path);
  const auto published = entriesUnder(ctx, dir);
  POLYTEST_CHECK_S(ctx, published.size() == 1, "publishing left {} files under {}", published.size(), dir);

  for (const auto &[what, other] : {std::pair{"identity", detail::moduleCachePath("test", "device-b", "image-1")},
                                    std::pair{"image", detail::moduleCachePath("test", "device-a", "image-2")},
                                    std::pair{"domain", detail::moduleCachePath("other", "device-a", "image-1")}}) {
    POLYTEST_CHECK_S(ctx, other != path, "a changed {} kept the key {}", what, path);
    POLYTEST_CHECK_S(ctx, cache::read(other).empty(), "a changed {} hit {}", what, other);
  }

  const auto entry = read_struct<uint8_t>(path);
  for (const auto &[what, damaged] : {
           std::pair{"empty", std::vector<uint8_t>{}},
           std::pair{"header-truncated", std::vector<uint8_t>(entry.begin(), entry.begin() + 4)},
           std::pair{"payload-truncated", std::vector<uint8_t>(entry.begin(), entry.end() - 1)},
           std::pair{"garbage", std::vector<uint8_t>(entry.size(), 0x5a)},
       }) {
    writeFile(path, damaged);
    POLYTEST_CHECK_S(ctx, cache::read(path).empty(), "a {} entry read back as a hit", what);
  }
  auto flipped = entry;
  flipped.back() ^= 0xff;
  writeFile(path, flipped);
  POLYTEST_CHECK_S(ctx, cache::read(path).empty(), "an entry with a flipped payload byte read back as a hit");

  cache::write(path, payload.data(), payload.size());
  POLYTEST_CHECK_S(ctx, cache::read(path) == payload, "republishing over a damaged entry did not restore it");
  cache::evict(path);
  POLYTEST_CHECK_S(ctx, cache::read(path).empty(), "read after evicting {} hit", path);

  POLYTEST_CHECK(ctx, detail::moduleCachePath("test", "", "image-1").empty());
  POLYTEST_CHECK(ctx, cache::read({}).empty());
  cache::write({}, payload.data(), payload.size());

  fs::remove_directories(dir);
}

void runModuleCache(Context &ctx, Backend backend, Platform &, Device &device, const ImageGroup &imageGroup) {
  if (imageGroup.size() != 1) {
    POLYTEST_FAIL(ctx, "expected exactly 1 image for fma kernel, got {} ({})", imageGroup.size(), magic_enum::enum_name(backend));
  }
  const auto dir = freshCacheDir(ctx);
  const auto &image = imageGroup[0].second;
  const std::string function_ = device.singleEntryPerModule() ? "main" : "_fma";

  auto q = device.createQueue(std::chrono::seconds(30));
  auto out_d = device.template mallocDeviceTyped<float>(1, Access::RW);
  const auto invoke = [&](const std::string &module_) {
    float a = 2.f, b = 3.f, c = 4.f, out = {};
    ArgBuffer buffer;
    if (device.sharedAddressSpace()) buffer.append(Type::IntS64, nullptr);
    buffer.append({{Type::Float32, &a}, {Type::Float32, &b}, {Type::Float32, &c}, {Type::Ptr, &out_d}, {Type::Void, {}}});
    waitAll([&](auto &h) { q->enqueueInvokeAsync(module_, function_, buffer, {}, h); });
    waitAll([&](auto &h) { q->enqueueDeviceToHostAsyncTyped(out_d, &out, 1, h); });
    return out;
  };

  device.loadModule("cold", image);
  POLYTEST_CHECK_S(ctx, invoke("cold") == 10.f, "cold build computed the wrong result");
  const auto published = entriesUnder(ctx, dir);
  POLYTEST_REQUIRE_S(ctx, !published.empty(), "loading a module published no cache entry under {}", dir);

  device.loadModule("warm", image);
  POLYTEST_CHECK_S(ctx, invoke("warm") == 10.f, "warm load computed the wrong result");
  POLYTEST_CHECK_S(ctx, entriesUnder(ctx, dir) == published, "warm load rebuilt or republished an entry under {}", dir);

  for (const auto &entry : published)
    writeFile(entry.first, {0xde, 0xad, 0xbe, 0xef});
  device.loadModule("corrupt", image);
  POLYTEST_CHECK_S(ctx, invoke("corrupt") == 10.f, "load over a corrupt entry computed the wrong result");
  for (const auto &[entry, bytes] : published)
    POLYTEST_CHECK_S(ctx, read_struct<uint8_t>(entry) == bytes, "corrupt entry {} was not republished", entry);

  device.freeDevice(out_d);
  fs::remove_directories(dir);
}

std::vector<Task> discoverAll() {
  auto tasks = discoverMatrix({
#ifndef __APPLE__
      {"module-cache-opencl-source", generated::opencl_source::fma, {Backend::OpenCL}, &runModuleCache, skipHasSpirv},
      {"module-cache-opencl-spirv", generated::opencl_spirv::fma, {Backend::OpenCL}, &runModuleCache, skipNoSpirv},
      {"module-cache-levelzero", generated::opencl_spirv::fma, {Backend::LevelZero}, &runModuleCache},
#endif
  });
  tasks.emplace_back(Task{"module-cache-entry", "", [] {
                            Context ctx;
                            runEntry(ctx);
                            return ctx.failed ? 1 : 0;
                          }});
  return tasks;
}

} // namespace

POLYTEST_DISCOVER(discoverAll)
