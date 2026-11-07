#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#if defined(__clang__)
  #define POLYREGION_SPECTRA_HOST_FALLBACK [[clang::annotate("polystl_host_fallback")]]
#else
  #define POLYREGION_SPECTRA_HOST_FALLBACK
#endif

// clang-format off
namespace spectra::host {

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void adjacent_difference(const T *in, T *out, std::int32_t n, Op op) {
  std::adjacent_difference(in, in + n, out, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline bool all_of(const T *in, std::int32_t n, Op op) {
  return std::all_of(in, in + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline bool any_of(const T *in, std::int32_t n, Op op) {
  return std::any_of(in, in + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline bool binary_search(const T *in, std::int32_t n, T value, Op op) {
  return std::binary_search(in, in + n, value, op);
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void copy(const T *in, std::int32_t n, T *out) {
  std::copy(in, in + n, out);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t copy_if(const T *in, std::int32_t n, T *out, Op op) {
  return static_cast<std::int32_t>(std::copy_if(in, in + n, out, op) - out);
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void copy_n(const T *in, std::int32_t n, T *out) {
  std::copy_n(in, n, out);
}

template <class T, class Eq>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t count(const T *in, std::int32_t n, T value, Eq eq) {
  return static_cast<std::int32_t>(std::count_if(in, in + n, [&](const T &x) { return eq(x, value); }));
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t count_if(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(std::count_if(in, in + n, op));
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline bool equal(const T *a, const T *b, std::int32_t n, Op op) {
  return std::equal(a, a + n, b, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void exclusive_scan(const T *in, T *out, std::int32_t n, T init, Op op) {
  std::exclusive_scan(in, in + n, out, init, op);
}

template <class K, class V, class Eq, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void exclusive_scan_by_key(
    const K *keys, const V *vals, V *out, std::int32_t n, V init, Eq eq, Op op) {
  V acc = init;
  for (std::int32_t i = 0; i < n; ++i) {
    if (i > 0 && !eq(keys[i - 1], keys[i])) acc = init;
    out[i] = acc;
    acc = op(acc, vals[i]);
  }
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void fill(T *out, std::int32_t n, T v) {
  std::fill_n(out, n, v);
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void fill_n(T *out, std::int32_t n, T v) {
  std::fill_n(out, n, v);
}

template <class T, class Eq>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t find(const T *in, std::int32_t n, T value, Eq eq) {
  return static_cast<std::int32_t>(std::find_if(in, in + n, [&](const T &x) { return eq(x, value); }) - in);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t find_if(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(std::find_if(in, in + n, op) - in);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t find_if_not(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(std::find_if_not(in, in + n, op) - in);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void for_each(T *data, std::int32_t n, Op op) {
  std::transform(data, data + n, data, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void for_each_n(T *data, std::int32_t n, Op op) {
  std::transform(data, data + n, data, op);
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void gather(
    const std::int32_t *map, std::int32_t n, const T *in, [[maybe_unused]] std::int32_t in_n, T *out) {
  for (std::int32_t i = 0; i < n; ++i)
    out[i] = in[map[i]];
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void generate(T *out, std::int32_t n, Op op) {
  std::generate(out, out + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void generate_n(T *out, std::int32_t n, Op op) {
  std::generate_n(out, n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline bool includes(const T *a, std::int32_t na, const T *b, std::int32_t nb, Op op) {
  return std::includes(a, a + na, b, b + nb, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void inclusive_scan(const T *in, T *out, std::int32_t n, Op op) {
  std::inclusive_scan(in, in + n, out, op);
}

template <class K, class V, class Eq, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void inclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, Eq eq, Op op) {
  for (std::int32_t i = 0; i < n; ++i)
    out[i] = i > 0 && eq(keys[i - 1], keys[i]) ? op(out[i - 1], vals[i]) : vals[i];
}

template <class T, class U, class V, class OpReduce, class OpProduct>
POLYREGION_SPECTRA_HOST_FALLBACK inline V inner_product(
    const T *a, const U *b, std::int32_t n, V init, OpReduce op_reduce, OpProduct op_product) {
  return std::inner_product(a, a + n, b, init, op_reduce, op_product);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline bool is_partitioned(const T *data, std::int32_t n, Op op) {
  return std::is_partitioned(data, data + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline bool is_sorted(const T *data, std::int32_t n, Op op) {
  return std::is_sorted(data, data + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t is_sorted_until(const T *data, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(std::is_sorted_until(data, data + n, op) - data);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t lower_bound(const T *in, std::int32_t n, T value, Op op) {
  return static_cast<std::int32_t>(std::lower_bound(in, in + n, value, op) - in);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline T max_element(const T *in, std::int32_t n, Op op) {
  return n == 0 ? T{} : *std::max_element(in, in + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void merge(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  std::merge(a, a + na, b, b + nb, out, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline T min_element(const T *in, std::int32_t n, Op op) {
  return n == 0 ? T{} : *std::min_element(in, in + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void minmax_element(const T *in, std::int32_t n, T *min_out, T *max_out, Op op) {
  if (n == 0) {
    *min_out = T{};
    *max_out = T{};
    return;
  }
  const auto [lo, hi] = std::minmax_element(in, in + n, op);
  *min_out = *lo;
  *max_out = *hi;
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t mismatch(const T *a, const T *b, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(std::mismatch(a, a + n, b, op).first - a);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline bool none_of(const T *in, std::int32_t n, Op op) {
  return std::none_of(in, in + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void partition(T *data, std::int32_t n, Op op) {
  std::partition(data, data + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t partition_point(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(std::partition_point(in, in + n, op) - in);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline T reduce(const T *in, std::int32_t n, T init, Op op) {
  return std::accumulate(in, in + n, init, op);
}

template <class K, class V, class Eq, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t reduce_by_key(
    const K *keys, const V *vals, K *kout, V *vout, std::int32_t n, Eq eq, Op op) {
  std::int32_t segments = 0;
  for (std::int32_t i = 0; i < n; ++i) {
    if (i > 0 && eq(keys[i - 1], keys[i])) {
      vout[segments - 1] = op(vout[segments - 1], vals[i]);
    } else {
      kout[segments] = keys[i];
      vout[segments] = vals[i];
      ++segments;
    }
  }
  return segments;
}

template <class T, class Eq>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t remove(T *data, std::int32_t n, T value, Eq eq) {
  return static_cast<std::int32_t>(std::remove_if(data, data + n, [&](const T &x) { return eq(x, value); }) - data);
}

template <class T, class Eq>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t remove_copy(const T *in, std::int32_t n, T *out, T value, Eq eq) {
  return static_cast<std::int32_t>(std::remove_copy_if(in, in + n, out, [&](const T &x) { return eq(x, value); }) - out);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t remove_copy_if(const T *in, std::int32_t n, T *out, Op op) {
  return static_cast<std::int32_t>(std::remove_copy_if(in, in + n, out, op) - out);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t remove_if(T *data, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(std::remove_if(data, data + n, op) - data);
}

template <class T, class Eq>
POLYREGION_SPECTRA_HOST_FALLBACK inline void replace(T *io, std::int32_t n, T oldv, T newv, Eq eq) {
  std::replace_if(io, io + n, [&](const T &x) { return eq(x, oldv); }, newv);
}

template <class T, class Eq>
POLYREGION_SPECTRA_HOST_FALLBACK inline void replace_copy(const T *in, T *out, std::int32_t n, T oldv, T newv, Eq eq) {
  std::replace_copy_if(in, in + n, out, [&](const T &x) { return eq(x, oldv); }, newv);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void replace_copy_if(const T *in, T *out, std::int32_t n, T new_value, Op op) {
  std::replace_copy_if(in, in + n, out, op, new_value);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void replace_if(T *data, std::int32_t n, T new_value, Op op) {
  std::replace_if(data, data + n, op, new_value);
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void reverse(T *data, std::int32_t n) {
  std::reverse(data, data + n);
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void reverse_copy(const T *in, std::int32_t n, T *out) {
  std::reverse_copy(in, in + n, out);
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void scatter(
    const T *in, std::int32_t n, const std::int32_t *map, T *out, [[maybe_unused]] std::int32_t out_n) {
  for (std::int32_t i = 0; i < n; ++i)
    out[map[i]] = in[i];
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t search(const T *in, std::int32_t n, const T *sub, std::int32_t m, Op op) {
  return static_cast<std::int32_t>(std::search(in, in + n, sub, sub + m, op) - in);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t search_n(const T *in, std::int32_t n, std::int32_t count, T value, Op op) {
  return static_cast<std::int32_t>(std::search_n(in, in + n, count, value, op) - in);
}

template <class T, class ValueAt>
POLYREGION_SPECTRA_HOST_FALLBACK inline void sequence(T *out, std::int32_t n, T init, T step, ValueAt value_at) {
  for (std::int32_t i = 0; i < n; ++i)
    out[i] = value_at(init, step, i);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t set_difference(
    const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(std::set_difference(a, a + na, b, b + nb, out, op) - out);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t set_intersection(
    const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(std::set_intersection(a, a + na, b, b + nb, out, op) - out);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t set_union(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(std::set_union(a, a + na, b, b + nb, out, op) - out);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void sort(T *data, std::int32_t n, Op op) {
  std::sort(data, data + n, op);
}

template <class K, class V, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void sort_by_key(K *keys, V *values, std::int32_t n, Op op) {
  std::vector<std::pair<K, V>> pairs;
  pairs.reserve(n);
  for (std::int32_t i = 0; i < n; ++i)
    pairs.emplace_back(keys[i], values[i]);
  std::stable_sort(pairs.begin(), pairs.end(), [&](const auto &a, const auto &b) { return op(a.first, b.first); });
  for (std::int32_t i = 0; i < n; ++i) {
    keys[i] = pairs[i].first;
    values[i] = pairs[i].second;
  }
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void stable_partition(T *data, std::int32_t n, Op op) {
  std::stable_partition(data, data + n, op);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void stable_sort(T *data, std::int32_t n, Op op) {
  std::stable_sort(data, data + n, op);
}

template <class T>
POLYREGION_SPECTRA_HOST_FALLBACK inline void swap_ranges(T *a, std::int32_t n, T *b) {
  std::swap_ranges(a, a + n, b);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void tabulate(T *out, std::int32_t n, Op op) {
  for (std::int32_t i = 0; i < n; ++i)
    out[i] = op(i);
}

template <class T, class U, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void transform(const T *in, U *out, std::int32_t n, Op op) {
  std::transform(in, in + n, out, op);
}

template <class T, class U, class V, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void transform_binary(const T *a, const U *b, V *out, std::int32_t n, Op op) {
  std::transform(a, a + n, b, out, op);
}

template <class T, class U, class Map, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void transform_exclusive_scan(const T *in, U *out, std::int32_t n, U init, Map map, Op op) {
  std::transform_exclusive_scan(in, in + n, out, init, op, map);
}

template <class T, class U, class Map, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline void transform_inclusive_scan(const T *in, U *out, std::int32_t n, Map map, Op op) {
  std::transform_inclusive_scan(in, in + n, out, op, map);
}

template <class T, class U, class Map, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline U transform_reduce(const T *in, std::int32_t n, U init, Map map, Op op) {
  return std::transform_reduce(in, in + n, init, op, map);
}

template <class T, class Eq>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t unique(T *data, std::int32_t n, Eq eq) {
  return static_cast<std::int32_t>(std::unique(data, data + n, eq) - data);
}

template <class T, class Eq>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t unique_copy(const T *in, std::int32_t n, T *out, Eq eq) {
  return static_cast<std::int32_t>(std::unique_copy(in, in + n, out, eq) - out);
}

template <class T, class Op>
POLYREGION_SPECTRA_HOST_FALLBACK inline std::int32_t upper_bound(const T *in, std::int32_t n, T value, Op op) {
  return static_cast<std::int32_t>(std::upper_bound(in, in + n, value, op) - in);
}

} // namespace spectra::host
// clang-format on

#undef POLYREGION_SPECTRA_HOST_FALLBACK

#define POLYREGION_SPECTRA_IMPLEMENT(function, ...) return ::spectra::host::function(__VA_ARGS__)
#include <polyregion/spectra_api.hpp>
#undef POLYREGION_SPECTRA_IMPLEMENT
