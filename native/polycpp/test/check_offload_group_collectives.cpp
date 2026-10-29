#pragma region case: group-collectives
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input}
#pragma region do: {output}
#pragma region requires: 64 64 64 64 64 64 64

#include <cstdint>
#include <cstdio>

#include "test_utils.h"

namespace sycl {
template <int D> struct group {};
struct plus {};
struct bit_or {};
struct bit_xor {};
struct minimum {};
struct maximum {};
template <class T, class Op> T reduce_over_group(group<1>, T value, Op) { return value; }
template <class T, class Op> T inclusive_scan_over_group(group<1>, T value, Op, T) { return value; }
template <class T, class Op> T exclusive_scan_over_group(group<1>, T value, T, Op) { return value; }
} // namespace sycl

int main() {
  constexpr uint32_t lanes = 64;
  int sum[lanes], bits[lanes], least[lanes], prefix[lanes], xors[lanes], rest[lanes];
  float peak[lanes];
  __polyregion_offload_workgroup__(lanes, [=, &sum, &bits, &least, &prefix, &xors, &rest, &peak](uint32_t lid) {
    const sycl::group<1> group;
    const int value = static_cast<int>(lid) + 1;
    sum[lid] = sycl::reduce_over_group(group, value, sycl::plus{});
    bits[lid] = sycl::reduce_over_group(group, 1 << (lid % 8), sycl::bit_or{});
    least[lid] = sycl::reduce_over_group(group, value, sycl::minimum{});
    peak[lid] = sycl::reduce_over_group(group, static_cast<float>(value) * 0.5f, sycl::maximum{});
    prefix[lid] = sycl::inclusive_scan_over_group(group, value, sycl::plus{}, 0);
    xors[lid] = sycl::inclusive_scan_over_group(group, value, sycl::bit_xor{}, 0);
    rest[lid] = sycl::exclusive_scan_over_group(group, value, 0, sycl::plus{});
  });
  int correct[7] = {};
  for (uint32_t i = 0; i < lanes; ++i) {
    int xorPrefix = 0;
    for (uint32_t j = 0; j <= i; ++j)
      xorPrefix ^= static_cast<int>(j) + 1;
    correct[0] += sum[i] == 2080;
    correct[1] += bits[i] == 255;
    correct[2] += least[i] == 1;
    correct[3] += peak[i] == 32.0f;
    correct[4] += prefix[i] == static_cast<int>((i + 1) * (i + 2) / 2);
    correct[5] += xors[i] == xorPrefix;
    correct[6] += rest[i] == static_cast<int>(i * (i + 1) / 2);
  }
  printf("%d %d %d %d %d %d %d", correct[0], correct[1], correct[2], correct[3], correct[4], correct[5], correct[6]);
  return 0;
}
