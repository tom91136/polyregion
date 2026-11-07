#pragma region case: group-regions-reduce
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-listing3
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-listing3-neighbour
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-continue
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-break
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-nested
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=6 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-early-exit
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=7 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-loop-return
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=8 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-rare-barrier
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=9 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-lane-state
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=10 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-recursion
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=11 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-collectives
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=12 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-local-arrays
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=13 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-reduce
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=1 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-listing3
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=2 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-listing3-neighbour
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=3 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-continue
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=4 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-break
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=5 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-nested
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=6 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-early-exit
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=7 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-loop-return
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=8 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-rare-barrier
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=9 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-lane-state
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=10 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-recursion
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=11 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-collectives
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=12 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-local-arrays
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=13 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-tile-transpose
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=14 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-uniform-rounds
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=15 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#pragma region case: group-regions-odd-lanes-uniform-rounds
#pragma region using: abi=group,lane
#pragma region offload-only
#pragma region targets: host x86_64 aarch64 arm
#pragma region do: POLYREGION_HOST_ABI={abi} polycpp {polycpp_defaults} {polycpp_stdpar} -DCHECK_KIND=15 -DODD_LANES=1 -o {output} {input}
#pragma region do: {output}
#pragma region requires: 0 0 0 0 0 0

#include <cstdint>
#include <cstdio>
#include <vector>

#include "test_utils.h"

#ifndef CHECK_KIND
  #define CHECK_KIND 1
#endif

namespace sycl {
template <int D> struct group {};
struct plus {};
struct minimum {};
struct maximum {};
template <class T, class Op> T reduce_over_group(group<1>, T value, Op) { return value; }
template <class T, class Op> T inclusive_scan_over_group(group<1>, T value, Op, T) { return value; }
template <class T, class Op> T exclusive_scan_over_group(group<1>, T value, T, Op) { return value; }
} // namespace sycl

struct Shape {
  uint32_t lanes, groups;
};

#if CHECK_KIND == 14
constexpr Shape shapes[] = {{4, 5}, {16, 1}, {16, 9}, {64, 3}, {256, 2}, {1024, 1}};
#elif defined(ODD_LANES)
constexpr Shape shapes[] = {{1, 2}, {5, 3}, {33, 3}, {100, 5}, {513, 2}, {1000, 1}};
#elif CHECK_KIND == 12
constexpr Shape shapes[] = {{64, 1}, {64, 3}, {256, 1}, {256, 2}, {1024, 1}, {1024, 2}};
#else
// lanes 1, 2, 3 and 1024; groups 1, prime, and more than the worker threads of a typical host
constexpr Shape shapes[] = {{1, 1}, {2, 13}, {3, 257}, {64, 13}, {256, 3}, {1024, 2}};
#endif

constexpr uint32_t MaxLanes = 1024;
constexpr int Rounds = 5;

#define barrier() __polyregion_gpu_barrier_local()

#if CHECK_KIND == 11
static int unwind(int n, [[clang::annotate("__polyregion_local")]] int *scratch, uint32_t lid, uint32_t lanes) {
  if (n == 0) return 0;
  scratch[lid] = n * 10 + static_cast<int>(lid % 7);
  barrier();
  const int seen = scratch[(lid + 1) % lanes];
  barrier();
  return unwind(n - 1, scratch, lid, lanes) + seen;
}
#endif

static uint32_t tileEdge(uint32_t lanes) {
  uint32_t edge = 1;
  while (edge * edge < lanes)
    ++edge;
  return edge;
}

static int expected(uint32_t lid, uint32_t group, uint32_t lanes) {
  const int global = static_cast<int>(group * lanes + lid);
  const auto tri = [](uint32_t l) { return static_cast<int>((1 + l) * (2 + l) / 2); };
  switch (CHECK_KIND) {
    case 1: {
      int sum = 0;
      for (uint32_t l = 0; l < lanes; ++l)
        sum += static_cast<int>(group * lanes + l) + 1;
      return sum;
    }
    case 2: return global + tri(lid);
    case 3: {
      const uint32_t nb = (lid + 1) % lanes;
      return static_cast<int>(group * lanes + nb) + tri(nb);
    }
    case 4: {
      const auto acc = [](uint32_t l) {
        int a = 0;
        for (int i = 0; i < 4; ++i)
          if ((l + i) % 2 == 1) a += i;
        return a;
      };
      return acc(lid) * 10 + acc((lid + 1) % lanes);
    }
    case 5: {
      const uint32_t nb = (lid + 1) % lanes;
      return nb % 4 == 0 ? 100 + static_cast<int>(nb) : static_cast<int>(nb % 4) - 1;
    }
    case 6: return lid < lanes / 2 && lid % 2 == 0 ? static_cast<int>(2 * lid + 1) : static_cast<int>(lid);
    case 7: {
      if (lid % 2 == 1) return -1;
      const uint32_t nb = (lid + 2) % lanes;
      return static_cast<int>(nb) + (nb % 2 == 0 ? 10 : 0);
    }
    case 8: return lid % 3 == 0 ? 1 : static_cast<int>((lid + 1) % lanes);
    case 9: return 1000;
    case 10: {
      int sum = 0;
      for (int k = 0; k < 20; ++k)
        sum += global * 3 + k;
      return sum;
    }
    case 11: {
      int sum = 0;
      for (int n = 1; n <= 5; ++n)
        sum += n * 10 + static_cast<int>(((lid + 1) % lanes) % 7);
      return sum;
    }
    case 12: {
      const int value = static_cast<int>(lid) + 1;
      const int n = static_cast<int>(lanes);
      return (n * (n + 1) / 2) % 1000 + (value * (value + 1) / 2) % 1000 * 1000 + (value - 1) * value / 2 % 1000 * 1000000;
    }
    case 13: return static_cast<int>(group * lanes + (lanes - 1 - lid));
    case 14: return static_cast<int>(group * lanes + lid % tileEdge(lanes) * tileEdge(lanes) + lid / tileEdge(lanes));
    case 15: {
      const uint32_t col = lid % 7;
      int acc = 0;
      for (int r = 0; r < Rounds; ++r)
        acc += (static_cast<int>(group * lanes + (lid + 1 + col) % lanes) + r) * (static_cast<int>(col) + 1);
      return acc;
    }
    default: return 0;
  }
}

int main() {
  const char *separator = "";
  for (const auto [lanes, groups] : shapes) {
    std::vector<int> out(lanes * groups);
    for (uint32_t i = 0; i < lanes * groups; ++i)
      out[i] = CHECK_KIND == 7 ? -1 : static_cast<int>(i);
    int *o = out.data();
    const uint32_t tile = tileEdge(lanes);
    const int rounds = Rounds;
    __polyregion_offload_workgroups__(groups, lanes, [o, tile, rounds](uint32_t global) {
      (void)tile;
      (void)rounds;
      [[clang::annotate("__polyregion_local")]] int scratch[MaxLanes];
      const uint32_t lid = __polyregion_gpu_local_idx(0);
      const uint32_t lanes = __polyregion_gpu_local_size(0);
#if CHECK_KIND == 1
      scratch[lid] = static_cast<int>(global) + 1;
      barrier();
      for (uint32_t s = 1; s < lanes; s *= 2) {
        if (lid % (2 * s) == 0 && lid + s < lanes) scratch[lid] += scratch[lid + s];
        barrier();
      }
      o[global] = scratch[0];
#elif CHECK_KIND == 2 || CHECK_KIND == 3
      scratch[lid] = o[global];
      barrier();
      for (size_t i = 0; i < 2 + lid; i++) {
        scratch[lid] += static_cast<int>(i);
        if (i < 2) barrier();
      }
  #if CHECK_KIND == 2
      o[global] = scratch[lid];
  #else
      barrier();
      o[global] = scratch[(lid + 1) % lanes];
  #endif
#elif CHECK_KIND == 4
      int acc = 0;
      scratch[lid] = 0;
      for (int i = 0; i < 4; i++) {
        if ((lid + i) % 2 == 0) continue;
        acc += i;
        scratch[lid] = acc;
        barrier();
      }
      barrier();
      o[global] = acc * 10 + scratch[(lid + 1) % lanes];
#elif CHECK_KIND == 5
      scratch[lid] = 100 + static_cast<int>(lid);
      for (int i = 0; i < 8; i++) {
        if (static_cast<uint32_t>(i) == lid % 4) break;
        scratch[lid] = i;
        barrier();
      }
      barrier();
      o[global] = scratch[(lid + 1) % lanes];
#elif CHECK_KIND == 6
      scratch[lid] = static_cast<int>(lid);
      if (lanes > 2) {
        if (lanes > 4) barrier();
        else barrier();
      }
      if (lid < lanes / 2) {
        if (lid % 2 == 0) {
          barrier();
          scratch[lid] += scratch[lid + 1];
        }
      }
      barrier();
      o[global] = scratch[lid];
#elif CHECK_KIND == 7
      scratch[lid] = static_cast<int>(lid);
      if (lid % 2 == 1) return;
      barrier();
      scratch[lid] += 10;
      barrier();
      o[global] = scratch[(lid + 2) % lanes];
#elif CHECK_KIND == 8
      scratch[lid] = static_cast<int>(lid);
      for (int j = 0; j < 4; j++) {
        if (j == 2 && lid % 3 == 0) return;
        o[global] = j;
      }
      barrier();
      o[global] = scratch[(lid + 1) % lanes];
#elif CHECK_KIND == 9
      scratch[lid] = 0;
      for (int i = 0; i < 1000; i++) {
        scratch[lid] += 1;
        if (i % 250 == 0) barrier();
      }
      barrier();
      o[global] = scratch[(lid + 1) % lanes];
#elif CHECK_KIND == 10
      int v[20];
      for (int k = 0; k < 20; k++)
        v[(k + lid) % 20] = static_cast<int>(global) * 3 + (k + static_cast<int>(lid)) % 20;
      scratch[lid] = v[lid % 20];
      barrier();
      int sum = 0;
      for (int k = 0; k < 20; k++)
        sum += v[k];
      o[global] = sum + scratch[lid] - v[lid % 20];
#elif CHECK_KIND == 11
      o[global] = unwind(5, scratch, lid, lanes);
#elif CHECK_KIND == 12
      const sycl::group<1> group;
      const int value = static_cast<int>(lid) + 1;
      const int sum = sycl::reduce_over_group(group, value, sycl::plus{});
      const int prefix = sycl::inclusive_scan_over_group(group, value, sycl::plus{}, 0);
      const int rest = sycl::exclusive_scan_over_group(group, value, 0, sycl::plus{});
      o[global] = sum % 1000 + prefix % 1000 * 1000 + rest % 1000 * 1000000;
#elif CHECK_KIND == 13
      [[clang::annotate("__polyregion_local")]] int flags[4];
      const int seen = flags[lid % 4];
      scratch[lid] = static_cast<int>(global);
      barrier();
      if (lid < 4) flags[lid] = static_cast<int>(__polyregion_gpu_group_idx(0)) + 1;
      barrier();
      o[global] = seen * 100000 + scratch[lanes - 1 - lid];
#elif CHECK_KIND == 14
      scratch[lid] = static_cast<int>(global);
      barrier();
      o[global] = scratch[(lid % tile) * tile + lid / tile];
#elif CHECK_KIND == 15
      int acc            = 0;
      const uint32_t col = lid % 7;
      for (int r = 0; r < rounds; r++) {
        scratch[lid] = static_cast<int>(global) + r;
        barrier();
        acc += scratch[(lid + 1 + col) % lanes] * (static_cast<int>(col) + 1);
        barrier();
      }
      o[global] = acc;
#endif
    });
    int bad = 0;
    for (uint32_t i = 0; i < lanes * groups; ++i)
      bad += out[i] != expected(i % lanes, i / lanes, lanes);
    std::printf("%s%d", separator, bad);
    separator = " ";
  }
  return 0;
}
