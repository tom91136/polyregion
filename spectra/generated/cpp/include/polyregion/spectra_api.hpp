#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__clang__)
#define POLYREGION_SPECTRA_ANNOTATE(value) [[clang::annotate(value)]]
#else
#define POLYREGION_SPECTRA_ANNOTATE(value)
#endif

#pragma push_macro("POLYREGION_SPECTRA_IMPLEMENT")
#ifndef POLYREGION_SPECTRA_IMPLEMENT
#define POLYREGION_SPECTRA_IMPLEMENT(function, ...) __builtin_trap()
#endif

namespace spectra {

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.adjacent_difference") //
inline void adjacent_difference(const T *in, T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(adjacent_difference, in, out, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.all_of") //
inline bool all_of(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(all_of, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.any_of") //
inline bool any_of(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(any_of, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.binary_search") //
inline bool binary_search(const T *in, std::int32_t n, T value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(binary_search, in, n, value, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.copy") //
inline void copy(const T *in, std::int32_t n, T *out) {
  POLYREGION_SPECTRA_IMPLEMENT(copy, in, n, out);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.copy_if") //
inline std::int32_t copy_if(const T *in, std::int32_t n, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(copy_if, in, n, out, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.copy_n") //
inline void copy_n(const T *in, std::int32_t n, T *out) {
  POLYREGION_SPECTRA_IMPLEMENT(copy_n, in, n, out);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.count") //
inline std::int32_t count(const T *in, std::int32_t n, T value) {
  POLYREGION_SPECTRA_IMPLEMENT(count, in, n, value);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.count_if") //
inline std::int32_t count_if(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(count_if, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.equal") //
inline bool equal(const T *a, const T *b, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(equal, a, b, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.exclusive_scan") //
inline void exclusive_scan(const T *in, T *out, std::int32_t n, T init, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(exclusive_scan, in, out, n, init, op);
}

template <class K, class V, class Eq, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.exclusive_scan_by_key") //
inline void exclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, V init, Eq eq, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const K &, const K &>, bool>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const V &, const V &>, V>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(exclusive_scan_by_key, keys, vals, out, n, init, eq, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.fill") //
inline void fill(T *out, std::int32_t n, T v) {
  POLYREGION_SPECTRA_IMPLEMENT(fill, out, n, v);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.fill_n") //
inline void fill_n(T *out, std::int32_t n, T v) {
  POLYREGION_SPECTRA_IMPLEMENT(fill_n, out, n, v);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.find") //
inline std::int32_t find(const T *in, std::int32_t n, T value) {
  POLYREGION_SPECTRA_IMPLEMENT(find, in, n, value);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.find_if") //
inline std::int32_t find_if(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(find_if, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.find_if_not") //
inline std::int32_t find_if_not(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(find_if_not, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.for_each") //
inline void for_each(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(for_each, data, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.for_each_n") //
inline void for_each_n(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(for_each_n, data, n, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.gather") //
inline void gather(const std::int32_t *map, std::int32_t n, const T *in, std::int32_t in_n, T *out) {
  POLYREGION_SPECTRA_IMPLEMENT(gather, map, n, in, in_n, out);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.generate") //
inline void generate(T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(generate, out, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.generate_n") //
inline void generate_n(T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(generate_n, out, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.includes") //
inline bool includes(const T *a, std::int32_t na, const T *b, std::int32_t nb, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(includes, a, na, b, nb, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.inclusive_scan") //
inline void inclusive_scan(const T *in, T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(inclusive_scan, in, out, n, op);
}

template <class K, class V, class Eq, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.inclusive_scan_by_key") //
inline void inclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, Eq eq, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const K &, const K &>, bool>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const V &, const V &>, V>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(inclusive_scan_by_key, keys, vals, out, n, eq, op);
}

template <class T, class U, class V, class OpReduce, class OpProduct> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.inner_product") //
inline V inner_product(const T *a, const U *b, std::int32_t n, V init, OpReduce op_reduce, OpProduct op_product) {
  static_assert(std::is_same_v<std::invoke_result_t<OpReduce &, const V &, const V &>, V>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<OpProduct &, const T &, const U &>, V>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(inner_product, a, b, n, init, op_reduce, op_product);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.is_partitioned") //
inline bool is_partitioned(const T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(is_partitioned, data, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.is_sorted") //
inline bool is_sorted(const T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(is_sorted, data, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.is_sorted_until") //
inline std::int32_t is_sorted_until(const T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(is_sorted_until, data, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.lower_bound") //
inline std::int32_t lower_bound(const T *in, std::int32_t n, T value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(lower_bound, in, n, value, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.max_element") //
inline T max_element(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(max_element, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.merge") //
inline void merge(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(merge, a, na, b, nb, out, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.min_element") //
inline T min_element(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(min_element, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.minmax_element") //
inline void minmax_element(const T *in, std::int32_t n, T *min_out, T *max_out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(minmax_element, in, n, min_out, max_out, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.mismatch") //
inline std::int32_t mismatch(const T *a, const T *b, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(mismatch, a, b, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.none_of") //
inline bool none_of(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(none_of, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.partition") //
inline void partition(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(partition, data, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.partition_point") //
inline std::int32_t partition_point(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(partition_point, in, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.reduce") //
inline T reduce(const T *in, std::int32_t n, T init, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(reduce, in, n, init, op);
}

template <class K, class V, class Eq, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.reduce_by_key") //
inline std::int32_t reduce_by_key(const K *keys, const V *vals, K *kout, V *vout, std::int32_t n, Eq eq, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const K &, const K &>, bool>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const V &, const V &>, V>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(reduce_by_key, keys, vals, kout, vout, n, eq, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.remove") //
inline std::int32_t remove(T *data, std::int32_t n, T value) {
  POLYREGION_SPECTRA_IMPLEMENT(remove, data, n, value);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.remove_copy") //
inline std::int32_t remove_copy(const T *in, std::int32_t n, T *out, T value) {
  POLYREGION_SPECTRA_IMPLEMENT(remove_copy, in, n, out, value);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.remove_copy_if") //
inline std::int32_t remove_copy_if(const T *in, std::int32_t n, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(remove_copy_if, in, n, out, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.remove_if") //
inline std::int32_t remove_if(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(remove_if, data, n, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.replace") //
inline void replace(T *io, std::int32_t n, T oldv, T newv) {
  POLYREGION_SPECTRA_IMPLEMENT(replace, io, n, oldv, newv);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.replace_copy") //
inline void replace_copy(const T *in, T *out, std::int32_t n, T oldv, T newv) {
  POLYREGION_SPECTRA_IMPLEMENT(replace_copy, in, out, n, oldv, newv);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.replace_copy_if") //
inline void replace_copy_if(const T *in, T *out, std::int32_t n, T new_value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(replace_copy_if, in, out, n, new_value, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.replace_if") //
inline void replace_if(T *data, std::int32_t n, T new_value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(replace_if, data, n, new_value, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.reverse") //
inline void reverse(T *data, std::int32_t n) {
  POLYREGION_SPECTRA_IMPLEMENT(reverse, data, n);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.reverse_copy") //
inline void reverse_copy(const T *in, std::int32_t n, T *out) {
  POLYREGION_SPECTRA_IMPLEMENT(reverse_copy, in, n, out);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.scatter") //
inline void scatter(const T *in, std::int32_t n, const std::int32_t *map, T *out, std::int32_t out_n) {
  POLYREGION_SPECTRA_IMPLEMENT(scatter, in, n, map, out, out_n);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.search") //
inline std::int32_t search(const T *in, std::int32_t n, const T *sub, std::int32_t m, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(search, in, n, sub, m, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.search_n") //
inline std::int32_t search_n(const T *in, std::int32_t n, std::int32_t count, T value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(search_n, in, n, count, value, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.sequence") //
inline void sequence(T *out, std::int32_t n, T init, T step) {
  POLYREGION_SPECTRA_IMPLEMENT(sequence, out, n, init, step);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.set_difference") //
inline std::int32_t set_difference(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(set_difference, a, na, b, nb, out, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.set_intersection") //
inline std::int32_t set_intersection(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, std::int32_t out_n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(set_intersection, a, na, b, nb, out, out_n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.set_union") //
inline std::int32_t set_union(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(set_union, a, na, b, nb, out, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.sort") //
inline void sort(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(sort, data, n, op);
}

template <class K, class V, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.sort_by_key") //
inline void sort_by_key(K *keys, V *values, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const K &, const K &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(sort_by_key, keys, values, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.stable_partition") //
inline void stable_partition(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(stable_partition, data, n, op);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.stable_sort") //
inline void stable_sort(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(stable_sort, data, n, op);
}

template <class T> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.swap_ranges") //
inline void swap_ranges(T *a, std::int32_t n, T *b) {
  POLYREGION_SPECTRA_IMPLEMENT(swap_ranges, a, n, b);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.tabulate") //
inline void tabulate(T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const std::int32_t &>, T>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(tabulate, out, n, op);
}

template <class T, class U, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.transform") //
inline void transform(const T *in, U *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, U>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(transform, in, out, n, op);
}

template <class T, class U, class V, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.transform_binary") //
inline void transform_binary(const T *a, const U *b, V *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const U &>, V>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(transform_binary, a, b, out, n, op);
}

template <class T, class U, class Map, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.transform_exclusive_scan") //
inline void transform_exclusive_scan(const T *in, U *out, std::int32_t n, U init, Map map, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Map &, const T &>, U>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const U &, const U &>, U>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(transform_exclusive_scan, in, out, n, init, map, op);
}

template <class T, class U, class Map, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.transform_inclusive_scan") //
inline void transform_inclusive_scan(const T *in, U *out, std::int32_t n, Map map, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Map &, const T &>, U>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const U &, const U &>, U>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(transform_inclusive_scan, in, out, n, map, op);
}

template <class T, class U, class Map, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.transform_reduce") //
inline U transform_reduce(const T *in, std::int32_t n, U init, Map map, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Map &, const T &>, U>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const U &, const U &>, U>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(transform_reduce, in, n, init, map, op);
}

template <class T, class Eq> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.unique") //
inline std::int32_t unique(T *data, std::int32_t n, Eq eq) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(unique, data, n, eq);
}

template <class T, class Eq> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.unique_copy") //
inline std::int32_t unique_copy(const T *in, std::int32_t n, T *out, Eq eq) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(unique_copy, in, n, out, eq);
}

template <class T, class Op> //
POLYREGION_SPECTRA_ANNOTATE("polyregion_interface:spectra:spectra.upper_bound") //
inline std::int32_t upper_bound(const T *in, std::int32_t n, T value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  POLYREGION_SPECTRA_IMPLEMENT(upper_bound, in, n, value, op);
}

} // namespace spectra

#undef POLYREGION_SPECTRA_ANNOTATE

#pragma pop_macro("POLYREGION_SPECTRA_IMPLEMENT")
