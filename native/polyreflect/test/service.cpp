#define __RT_IMPL
#include <map>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "reflect-rt/rt_reflect.hpp"

using namespace polyregion::rt_reflect;
using namespace polyregion::rt_reflect::details;

namespace {

constexpr uintptr_t Origin = 0x10000000;
constexpr size_t Stride = 128, Extent = 64;

void fill(ReflectService &service, const size_t records) {
  for (size_t i = 0; i < records; ++i)
    service.blockingRecord(PtrInfo{Origin + i * Stride, Extent, Type::HeapMalloc});
}

} // namespace

TEST_CASE("exact, interior and past-end lookups resolve to the owning record") {
  ReflectService service;
  fill(service, 64);
  const auto base = Origin + 10 * Stride;

  auto meta = service.blockingQuery(base);
  CHECK(meta.size == Extent);
  CHECK(meta.offset == 0);

  meta = service.blockingQuery(base + 17);
  CHECK(meta.size == Extent);
  CHECK(meta.offset == 17);
  CHECK(meta.type == Type::HeapMalloc);

  meta = service.blockingQuery(base + Extent);
  CHECK(meta.size == Extent);
  CHECK(meta.offset == Extent);

  meta = service.blockingQuery(base + Extent + 1);
  CHECK(meta.size == 0);
  CHECK(meta.type == Type::Unknown);
  CHECK(service.blockingQuery(Origin - 1).size == 0);
  CHECK(service.blockingQuery(Origin + 64 * Stride).size == 0);
}

TEST_CASE("strict containment wins over the previous record's past-end") {
  ReflectService service;
  service.blockingRecord(PtrInfo{0x1000, 0x10, Type::HeapMalloc});
  service.blockingRecord(PtrInfo{0x1010, 0x20, Type::HeapMalloc});
  const auto meta = service.blockingQuery(uintptr_t{0x1018});
  CHECK(meta.size == 0x20);
  CHECK(meta.offset == 8);
  const auto edge = service.blockingQuery(uintptr_t{0x1010});
  CHECK(edge.size == 0x20);
  CHECK(edge.offset == 0);
}

TEST_CASE("an enclosing record is found beyond a nested record's end") {
  ReflectService service;
  service.blockingRecord(PtrInfo{0x1000, 0x1000, Type::StaticRodata});
  service.blockingRecord(PtrInfo{0x1100, 0x10, Type::HeapMalloc});
  const auto inner = service.blockingQuery(uintptr_t{0x1108});
  CHECK(inner.size == 0x10);
  CHECK(inner.type == Type::HeapMalloc);
  const auto outer = service.blockingQuery(uintptr_t{0x1800});
  CHECK(outer.size == 0x1000);
  CHECK(outer.offset == 0x800);
  CHECK(outer.type == Type::StaticRodata);
}

TEST_CASE("released and replaced records stop matching") {
  ReflectService service;
  fill(service, 32);
  const auto base = Origin + 5 * Stride;

  service.blockingRelease(base, Type::HeapFree);
  CHECK(service.blockingQuery(base + 8).size == 0);
  CHECK(service.blockingQuery(base - Stride + 8).size == Extent);

  service.blockingRecord(PtrInfo{base, 16, Type::HeapCalloc});
  service.blockingRecord(PtrInfo{base, 32, Type::HeapRealloc});
  const auto meta = service.blockingQuery(base + 24);
  CHECK(meta.size == 32);
  CHECK(meta.type == Type::HeapRealloc);
  CHECK(service.blockingQuery(base + 40).size == 0);
}

TEST_CASE("interior lookups examine a logarithmic number of records") {
  constexpr size_t records = 1 << 16;
  ReflectService service;
  fill(service, records);

  for (size_t i = 0; i < records; i += 7) {
    const auto meta = service.blockingQuery(Origin + i * Stride + 9);
    REQUIRE(meta.size == Extent);
    REQUIRE(meta.offset == 9);
  }
  for (size_t i = 0; i < records; i += 11) {
    const auto meta = service.blockingQuery(Origin + i * Stride + Extent);
    REQUIRE(meta.size == Extent);
    REQUIRE(meta.offset == Extent);
  }
  const size_t queries = (records + 6) / 7 + (records + 10) / 11;
  CHECK(service.subrangeProbes() <= queries * 128);
}

TEST_CASE("lookups stay correct while records churn") {
  constexpr size_t records = 2048;
  ReflectService service;
  fill(service, records);
  std::vector<bool> live(records, true);
  for (size_t round = 0; round < 4; ++round) {
    for (size_t i = round; i < records; i += 3) {
      if (live[i]) service.blockingRelease(Origin + i * Stride, Type::HeapFree);
      else service.blockingRecord(PtrInfo{Origin + i * Stride, Extent, Type::HeapMalloc});
      live[i] = !live[i];
    }
    for (size_t i = 0; i < records; ++i)
      REQUIRE(service.blockingQuery(Origin + i * Stride + 3).size == (live[i] ? Extent : 0));
  }
}

TEST_CASE("interval index agrees with a linear scan over overlapping intervals") {
  struct Entry {
    uintptr_t base;
    size_t size;
  };
  IntervalIndex<Entry> index;
  std::map<uintptr_t, Entry> reference;
  std::mt19937_64 rng(42);
  const auto below = [&rng](size_t bound) { return static_cast<size_t>(rng() % bound); };
  for (int step = 0; step < 20000; ++step) {
    const uintptr_t base = 0x1000 + below(2048) * 16;
    if (const auto it = reference.find(base); it != reference.end()) {
      REQUIRE(index.erase(base));
      reference.erase(it);
    } else {
      const auto &entry = reference.emplace(base, Entry{base, below(64) * 16}).first->second;
      index.insert(base, entry.size, &entry);
    }
    REQUIRE(index.size() == reference.size());

    const uintptr_t point = 0x1000 + below(2048 * 16 + 64 * 16);
    const Entry *expectedContaining = nullptr, *expectedEnding = nullptr;
    for (const auto &[b, entry] : reference) {
      if (b > point) break;
      if (point < b + entry.size) expectedContaining = &entry;
      if (point == b + entry.size) expectedEnding = &entry;
    }
    size_t visited = 0;
    REQUIRE(index.containing(point, visited) == expectedContaining);
    if (!expectedContaining) REQUIRE(index.endingAt(point, visited) == expectedEnding);
  }
}
