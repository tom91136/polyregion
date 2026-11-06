#include "polyrt/device_heap.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "polyinvoke/object_platform.h"
#include "polyrt/rt.h"

using namespace polyregion::invoke;
using namespace polyregion::polyrt;

namespace {

struct HeapDevice final : object::ObjectDevice {
  std::string name() override { return "heap"; }
  void loadModule(const std::string &, const std::string &) override {}
  bool moduleLoaded(const std::string &) override { return false; }
  std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &) override { return {}; }
};

struct HeapQueue final : object::ObjectDeviceQueue {
  using ObjectDeviceQueue::ObjectDeviceQueue;
  void enqueueInvokeAsync(const std::string &, const std::string &, const std::vector<Type> &, std::vector<std::byte>, const Policy &,
                          const MaybeCallback &) override {}
};

struct Fixture {
  HeapDevice device;
  HeapQueue queue{std::chrono::seconds(1)};
  HeapCounters counters;
  size_t slack;
  HeapLimits limits;
  DeviceHeap heap;

  explicit Fixture(const size_t maxBytes = size_t{1} << 30, const size_t alignment = 16, const size_t slack = 0)
      : slack(slack), limits{maxBytes, alignment}, heap(device, queue, [] {}, slack, limits, counters) {}

  size_t blockOf(const size_t bytes) const { return (bytes + 1 + slack + limits.alignment - 1) / limits.alignment * limits.alignment; }
  uint8_t *bytes(const uintptr_t pointer) const { return reinterpret_cast<uint8_t *>(pointer); }
};

} // namespace

TEST_CASE("heap blocks are aligned, never at offset zero, and counted") {
  Fixture f(size_t{1} << 30, 256);
  std::vector<uintptr_t> blocks;
  for (const size_t bytes : {1, 7, 255, 256, 257, 1000, 4096, 1 << 20})
    blocks.push_back(f.heap.allocate(bytes));
  for (const auto block : blocks) {
    CHECK((block - f.heap.handle()) % 256 == 0);
    CHECK(block - f.heap.handle() >= 256);
  }
  CHECK(f.heap.blocks() == blocks.size());
  CHECK(f.counters.live == f.heap.liveBytes());
  for (const auto block : blocks)
    f.heap.release(block);
  CHECK(f.heap.blocks() == 0);
  CHECK(f.counters.live == 0);
}

TEST_CASE("heap zeroes the tail pad and slack of a reused block") {
  Fixture f(size_t{1} << 30, 16, 4096);
  const auto first = f.heap.allocate(100);
  std::memset(f.bytes(first), 0xFF, f.blockOf(100));
  f.heap.release(first);
  const auto second = f.heap.allocate(100);
  REQUIRE(second == first);
  for (size_t i = 100; i < f.blockOf(100); ++i)
    REQUIRE(f.bytes(second)[i] == 0);
}

TEST_CASE("heap blocks never touch so a one-past-end pointer belongs to its own block") {
  Fixture f;
  const auto a = f.heap.allocate(64);
  const auto b = f.heap.allocate(64);
  CHECK(a + 64 < b);
  CHECK(b > a);
}

TEST_CASE("heap picks the smallest hole that fits") {
  Fixture f;
  const auto a = f.heap.allocate(500);
  const auto separator1 = f.heap.allocate(16);
  const auto b = f.heap.allocate(200);
  const auto separator2 = f.heap.allocate(16);
  const auto c = f.heap.allocate(300);
  const auto separator3 = f.heap.allocate(16);
  f.heap.release(a);
  f.heap.release(b);
  f.heap.release(c);
  CHECK(f.heap.allocate(190) == b);
  CHECK(f.heap.allocate(290) == c);
  CHECK(f.heap.allocate(490) == a);
  for (const auto s : {separator1, separator2, separator3})
    f.heap.release(s);
}

TEST_CASE("freed neighbours coalesce into one span in any release order") {
  for (const std::array<int, 3> order : {std::array{0, 1, 2}, {2, 1, 0}, {1, 0, 2}, {0, 2, 1}, {1, 2, 0}, {2, 0, 1}}) {
    Fixture f(size_t{1} << 30, 16, 0);
    const std::array<uintptr_t, 3> blocks{f.heap.allocate(1000), f.heap.allocate(1000), f.heap.allocate(1000)};
    const auto separator = f.heap.allocate(16);
    const auto growths = f.counters.growths.load();
    for (const auto i : order)
      f.heap.release(blocks[i]);
    CHECK(f.heap.allocate(3 * f.blockOf(1000) - 1 - 15) == blocks[0]);
    CHECK(f.counters.growths == growths);
    f.heap.release(separator);
  }
}

TEST_CASE("heap grows in powers of two, keeps its handle and stays within the limit") {
  Fixture f(size_t{1} << 24);
  std::vector<uintptr_t> blocks;
  const auto handle = [&] { return f.heap.handle(); };
  blocks.push_back(f.heap.allocate(8));
  const auto base = handle();
  uint64_t seen = f.counters.growths;
  for (int i = 0; i < 100; ++i) {
    blocks.push_back(f.heap.allocate(100000));
    CHECK(handle() == base);
    CHECK(f.heap.committedBytes() <= (size_t{1} << 24));
    CHECK(std::has_single_bit(f.heap.committedBytes()));
    CHECK(f.counters.growths >= seen);
    seen = f.counters.growths;
  }
  CHECK(f.counters.committed == f.heap.committedBytes());
  CHECK(f.heap.liveBytes() <= f.heap.committedBytes());
  CHECK(seen > 0);
}

TEST_CASE("exhaustion names the request, the live bytes and the limit, and leaves the heap usable") {
  constexpr size_t limit = size_t{1} << 20;
  Fixture f(limit);
  std::vector<uintptr_t> blocks;
  std::string message;
  try {
    while (true)
      blocks.push_back(f.heap.allocate(50000));
  } catch (const DeviceHeapExhausted &e) {
    message = e.what();
  }
  CHECK(message.find("exhausted") != std::string::npos);
  CHECK(message.find(std::to_string(limit)) != std::string::npos);
  CHECK(message.find("50000") != std::string::npos);
  REQUIRE_FALSE(blocks.empty());
  const auto live = f.heap.liveBytes();
  CHECK(live <= limit);
  CHECK_THROWS_AS(f.heap.allocate(limit + 1), DeviceHeapExhausted);
  CHECK_THROWS_AS(f.heap.allocate(std::numeric_limits<size_t>::max()), DeviceHeapExhausted);
  CHECK(f.heap.liveBytes() == live);
  f.heap.release(blocks.back());
  CHECK_NOTHROW(f.heap.allocate(50000));
}

TEST_CASE("a block extends in place into the free span after it and zeroes its new pad") {
  Fixture f(size_t{1} << 30, 16, 4096);
  const auto block = f.heap.allocate(100);
  const auto liveBefore = f.heap.liveBytes();
  std::memset(f.bytes(block), 0xFF, f.heap.committedBytes() - (block - f.heap.handle()));
  REQUIRE(f.heap.extend(block, 10000));
  CHECK(f.heap.liveBytes() == liveBefore + f.blockOf(10000) - f.blockOf(100));
  CHECK(f.heap.blocks() == 1);
  for (size_t i = 10000; i < f.blockOf(10000); ++i)
    REQUIRE(f.bytes(block)[i] == 0);
  const auto next = f.heap.allocate(100);
  CHECK(next >= block + f.blockOf(10000));
  f.heap.release(block);
  f.heap.release(next);
  CHECK(f.heap.liveBytes() == 0);
}

TEST_CASE("a block does not extend over its neighbour and stays as it was") {
  Fixture f(size_t{1} << 30, 16);
  const auto block = f.heap.allocate(100);
  const auto neighbour = f.heap.allocate(100);
  const auto live = f.heap.liveBytes();
  CHECK_FALSE(f.heap.extend(block, 10000));
  CHECK(f.heap.liveBytes() == live);
  CHECK(f.heap.extend(block, 100));
  CHECK(f.heap.liveBytes() == live);
  f.heap.release(neighbour);
  CHECK(f.heap.extend(block, 10000));
}

TEST_CASE("extending the last block grows the heap, keeps its handle and respects the limit") {
  Fixture f(size_t{1} << 20, 16);
  const auto block = f.heap.allocate(100);
  const auto handle = f.heap.handle();
  const auto committed = f.heap.committedBytes();
  REQUIRE(f.heap.extend(block, 3 * committed));
  CHECK(f.heap.handle() == handle);
  CHECK(f.heap.committedBytes() > committed);
  CHECK(std::has_single_bit(f.heap.committedBytes()));
  CHECK(f.counters.growths >= 1);
  const auto live = f.heap.liveBytes();
  CHECK_FALSE(f.heap.extend(block, size_t{1} << 21));
  CHECK(f.heap.liveBytes() == live);
}

TEST_CASE("extending an unknown block is an error") {
  Fixture f;
  const auto block = f.heap.allocate(100);
  CHECK_THROWS_AS(f.heap.extend(block + 16, 200), std::logic_error);
  f.heap.release(block);
  CHECK_THROWS_AS(f.heap.extend(block, 200), std::logic_error);
}

TEST_CASE("releasing an unknown or already released block is an error") {
  Fixture f;
  const auto block = f.heap.allocate(32);
  CHECK_THROWS_AS(f.heap.release(block + 16), std::logic_error);
  CHECK_THROWS_AS(f.heap.release(f.heap.handle() - 16), std::logic_error);
  f.heap.release(block);
  CHECK_THROWS_AS(f.heap.release(block), std::logic_error);
}

TEST_CASE("random churn never overlaps live blocks and reclaims everything") {
  Fixture f(size_t{1} << 30, 64, 4096);
  std::mt19937_64 rng(0x5eed);
  std::map<uintptr_t, size_t> live;
  std::vector<uintptr_t> order;
  size_t peak = 0;
  for (int op = 0; op < 100000; ++op) {
    const bool doAlloc = live.empty() || order.size() < 2000 ? rng() % 100 < 60 : rng() % 100 < 40;
    if (doAlloc) {
      const size_t bytes = 1 + rng() % (rng() % 8 == 0 ? 20000 : 600);
      const auto block = f.heap.allocate(bytes);
      const auto next = live.lower_bound(block);
      if (next != live.end()) REQUIRE(block + f.blockOf(bytes) <= next->first);
      if (next != live.begin()) {
        const auto previous = std::prev(next);
        REQUIRE(previous->first + f.blockOf(previous->second) <= block);
      }
      live.emplace(block, bytes);
      order.push_back(block);
    } else {
      const auto index = rng() % order.size();
      const auto block = order[index];
      order[index] = order.back();
      order.pop_back();
      f.heap.release(block);
      live.erase(block);
    }
    peak = std::max(peak, f.heap.liveBytes());
    REQUIRE(f.heap.liveBytes() <= f.heap.committedBytes());
  }
  CHECK(f.heap.committedBytes() <= 4 * std::bit_ceil(peak + 64));
  for (const auto block : order)
    f.heap.release(block);
  CHECK(f.heap.liveBytes() == 0);
  const auto growths = f.counters.growths.load();
  CHECK_NOTHROW(f.heap.allocate(f.heap.committedBytes() - 64 - 1 - 4096));
  CHECK(f.counters.growths == growths);
}

TEST_CASE("concurrent allocation and release under one lock stays consistent") {
  Fixture f(size_t{1} << 30, 32, 0);
  std::mutex transaction;
  std::map<uintptr_t, size_t> live;
  std::vector<std::thread> threads;
  std::atomic<bool> failed{false};
  for (int t = 0; t < 8; ++t)
    threads.emplace_back([&, t] {
      std::mt19937_64 rng(t);
      std::vector<uintptr_t> mine;
      for (int op = 0; op < 5000; ++op) {
        const std::lock_guard lock(transaction);
        if (mine.empty() || rng() % 100 < 55) {
          const size_t bytes = 1 + rng() % 3000;
          const auto block = f.heap.allocate(bytes);
          const auto next = live.lower_bound(block);
          if ((next != live.end() && block + f.blockOf(bytes) > next->first)
              || (next != live.begin() && std::prev(next)->first + f.blockOf(std::prev(next)->second) > block))
            failed = true;
          live.emplace(block, bytes);
          mine.push_back(block);
        } else {
          const auto index = rng() % mine.size();
          const auto block = mine[index];
          mine[index] = mine.back();
          mine.pop_back();
          f.heap.release(block);
          live.erase(block);
        }
      }
      const std::lock_guard lock(transaction);
      for (const auto block : mine) {
        f.heap.release(block);
        live.erase(block);
      }
    });
  for (auto &thread : threads)
    thread.join();
  CHECK_FALSE(failed);
  CHECK(live.empty());
  CHECK(f.heap.liveBytes() == 0);
}
