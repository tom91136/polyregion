#pragma once

#include <cstdint>
#include <type_traits>

namespace spectra {

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.adjacent_difference")]] inline void adjacent_difference(const T *in, T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.all_of")]] inline bool all_of(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.any_of")]] inline bool any_of(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.binary_search")]] inline bool binary_search(const T *in, std::int32_t n, T value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.copy")]] inline void copy(const T *in, std::int32_t n, T *out) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.copy_if")]] inline std::int32_t copy_if(const T *in, std::int32_t n, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.copy_n")]] inline void copy_n(const T *in, std::int32_t n, T *out) {
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.count")]] inline std::int32_t count(const T *in, std::int32_t n, T value) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.count_if")]] inline std::int32_t count_if(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.equal")]] inline bool equal(const T *a, const T *b, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.exclusive_scan")]] inline void exclusive_scan(const T *in, T *out, std::int32_t n, T init, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class K, class V, class Eq, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.exclusive_scan_by_key")]] inline void exclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, V init, Eq eq, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const K &, const K &>, bool>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const V &, const V &>, V>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.fill")]] inline void fill(T *out, std::int32_t n, T v) {
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.fill_n")]] inline void fill_n(T *out, std::int32_t n, T v) {
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.find")]] inline std::int32_t find(const T *in, std::int32_t n, T value) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.find_if")]] inline std::int32_t find_if(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.find_if_not")]] inline std::int32_t find_if_not(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.for_each")]] inline void for_each(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.for_each_n")]] inline void for_each_n(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.gather")]] inline void gather(const std::int32_t *map, std::int32_t n, const T *in, std::int32_t in_n, T *out) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.generate")]] inline void generate(T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.generate_n")]] inline void generate_n(T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.includes")]] inline bool includes(const T *a, std::int32_t na, const T *b, std::int32_t nb, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.inclusive_scan")]] inline void inclusive_scan(const T *in, T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class K, class V, class Eq, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.inclusive_scan_by_key")]] inline void inclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, Eq eq, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const K &, const K &>, bool>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const V &, const V &>, V>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class U, class V, class OpReduce, class OpProduct>
[[clang::annotate("polyregion_import:spectra:spectra.inner_product")]] inline V inner_product(const T *a, const U *b, std::int32_t n, V init, OpReduce op_reduce, OpProduct op_product) {
  static_assert(std::is_same_v<std::invoke_result_t<OpReduce &, const V &, const V &>, V>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<OpProduct &, const T &, const U &>, V>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.is_partitioned")]] inline bool is_partitioned(const T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.is_sorted")]] inline bool is_sorted(const T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.is_sorted_until")]] inline std::int32_t is_sorted_until(const T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.lower_bound")]] inline std::int32_t lower_bound(const T *in, std::int32_t n, T value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.max_element")]] inline T max_element(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.merge")]] inline void merge(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.min_element")]] inline T min_element(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.minmax_element")]] inline void minmax_element(const T *in, std::int32_t n, T *min_out, T *max_out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.mismatch")]] inline std::int32_t mismatch(const T *a, const T *b, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.none_of")]] inline bool none_of(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.partition")]] inline void partition(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.partition_point")]] inline std::int32_t partition_point(const T *in, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.reduce")]] inline T reduce(const T *in, std::int32_t n, T init, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class K, class V, class Eq, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.reduce_by_key")]] inline std::int32_t reduce_by_key(const K *keys, const V *vals, K *kout, V *vout, std::int32_t n, Eq eq, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const K &, const K &>, bool>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const V &, const V &>, V>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.remove")]] inline std::int32_t remove(T *data, std::int32_t n, T value) {
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.remove_copy")]] inline std::int32_t remove_copy(const T *in, std::int32_t n, T *out, T value) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.remove_copy_if")]] inline std::int32_t remove_copy_if(const T *in, std::int32_t n, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.remove_if")]] inline std::int32_t remove_if(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.replace")]] inline void replace(T *io, std::int32_t n, T oldv, T newv) {
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.replace_copy")]] inline void replace_copy(const T *in, T *out, std::int32_t n, T oldv, T newv) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.replace_copy_if")]] inline void replace_copy_if(const T *in, T *out, std::int32_t n, T new_value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.replace_if")]] inline void replace_if(T *data, std::int32_t n, T new_value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.reverse")]] inline void reverse(T *data, std::int32_t n) {
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.reverse_copy")]] inline void reverse_copy(const T *in, std::int32_t n, T *out) {
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.scatter")]] inline void scatter(const T *in, std::int32_t n, const std::int32_t *map, T *out, std::int32_t out_n) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.search")]] inline std::int32_t search(const T *in, std::int32_t n, const T *sub, std::int32_t m, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.search_n")]] inline std::int32_t search_n(const T *in, std::int32_t n, std::int32_t count, T value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.sequence")]] inline void sequence(T *out, std::int32_t n, T init, T step) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.set_difference")]] inline std::int32_t set_difference(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.set_intersection")]] inline std::int32_t set_intersection(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, std::int32_t out_n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.set_union")]] inline std::int32_t set_union(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.sort")]] inline void sort(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class K, class V, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.sort_by_key")]] inline void sort_by_key(K *keys, V *values, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const K &, const K &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.stable_partition")]] inline void stable_partition(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.stable_sort")]] inline void stable_sort(T *data, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T>
[[clang::annotate("polyregion_import:spectra:spectra.swap_ranges")]] inline void swap_ranges(T *a, std::int32_t n, T *b) {
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.tabulate")]] inline void tabulate(T *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const std::int32_t &>, T>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class U, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.transform")]] inline void transform(const T *in, U *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &>, U>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class U, class V, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.transform_binary")]] inline void transform_binary(const T *a, const U *b, V *out, std::int32_t n, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const U &>, V>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class U, class Map, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.transform_exclusive_scan")]] inline void transform_exclusive_scan(const T *in, U *out, std::int32_t n, U init, Map map, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Map &, const T &>, U>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const U &, const U &>, U>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class U, class Map, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.transform_inclusive_scan")]] inline void transform_inclusive_scan(const T *in, U *out, std::int32_t n, Map map, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Map &, const T &>, U>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const U &, const U &>, U>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class U, class Map, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.transform_reduce")]] inline U transform_reduce(const T *in, std::int32_t n, U init, Map map, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Map &, const T &>, U>, "callable signature mismatch");
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const U &, const U &>, U>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Eq>
[[clang::annotate("polyregion_import:spectra:spectra.unique")]] inline std::int32_t unique(T *data, std::int32_t n, Eq eq) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Eq>
[[clang::annotate("polyregion_import:spectra:spectra.unique_copy")]] inline std::int32_t unique_copy(const T *in, std::int32_t n, T *out, Eq eq) {
  static_assert(std::is_same_v<std::invoke_result_t<Eq &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

template <class T, class Op>
[[clang::annotate("polyregion_import:spectra:spectra.upper_bound")]] inline std::int32_t upper_bound(const T *in, std::int32_t n, T value, Op op) {
  static_assert(std::is_same_v<std::invoke_result_t<Op &, const T &, const T &>, bool>, "callable signature mismatch");
  __builtin_trap();
}

} // namespace spectra
