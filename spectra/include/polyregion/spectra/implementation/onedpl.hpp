#pragma once

#ifndef POLYREGION_SPECTRA_EXPORT
  #define POLYREGION_SPECTRA_EXPORT(name)
  #define POLYREGION_SPECTRA_UNDEF_EXPORT
#endif

#include <cstdint>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>

namespace spectra::implementation {

namespace detail {

template <class T> class device_allocation {
  sycl::queue queue;
  T *value;

public:
  device_allocation(T *value, sycl::queue queue) : queue(queue), value(value) {}
  device_allocation(const device_allocation &) = delete;
  device_allocation &operator=(const device_allocation &) = delete;
  ~device_allocation() { sycl::free(value, queue); }
};

} // namespace detail

template <class T, class Op>                   //
POLYREGION_SPECTRA_EXPORT(adjacent_difference) //
    void adjacent_difference(const T *in, T *out, std::int32_t n, Op op) {
  oneapi::dpl::adjacent_difference(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, op);
}

template <class T, class Op>      //
POLYREGION_SPECTRA_EXPORT(all_of) //
    bool all_of(const T *in, std::int32_t n, Op op) {
  return oneapi::dpl::all_of(oneapi::dpl::execution::device_policy<>{}, in, in + n, op);
}

template <class T, class Op>      //
POLYREGION_SPECTRA_EXPORT(any_of) //
    bool any_of(const T *in, std::int32_t n, Op op) {
  return oneapi::dpl::any_of(oneapi::dpl::execution::device_policy<>{}, in, in + n, op);
}

template <class T, class Op>             //
POLYREGION_SPECTRA_EXPORT(binary_search) //
    bool binary_search(const T *in, std::int32_t n, T value, Op op) {
  auto policy = oneapi::dpl::execution::device_policy<>{};
  auto queue = policy.queue();
  auto *device_value = sycl::malloc_device<T>(1, queue);
  auto *device_index = sycl::malloc_device<std::int32_t>(1, queue);
  auto *device_result = sycl::malloc_device<bool>(1, queue);
  detail::device_allocation<T> valueAllocation(device_value, queue);
  detail::device_allocation<std::int32_t> indexAllocation(device_index, queue);
  detail::device_allocation<bool> resultAllocation(device_result, queue);
  queue.memcpy(device_value, &value, sizeof(value)).wait();
  oneapi::dpl::lower_bound(policy, in, in + n, device_value, device_value + 1, device_index, op);
  oneapi::dpl::transform(policy, device_index, device_index + 1, device_result, [=](std::int32_t index) {
    return index < n && !op(in[index], device_value[0]) && !op(device_value[0], in[index]);
  });
  bool result = false;
  queue.memcpy(&result, device_result, sizeof(result)).wait();
  return result;
}

template <class T>              //
POLYREGION_SPECTRA_EXPORT(copy) //
    void copy(const T *in, std::int32_t n, T *out) {
  oneapi::dpl::copy(oneapi::dpl::execution::device_policy<>{}, in, in + n, out);
}

template <class T, class Op>       //
POLYREGION_SPECTRA_EXPORT(copy_if) //
    std::int32_t copy_if(const T *in, std::int32_t n, T *out, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::copy_if(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, op) - out);
}

template <class T>                //
POLYREGION_SPECTRA_EXPORT(copy_n) //
    void copy_n(const T *in, std::int32_t n, T *out) {
  oneapi::dpl::copy_n(oneapi::dpl::execution::device_policy<>{}, in, n, out);
}

template <class T, class Eq>     //
POLYREGION_SPECTRA_EXPORT(count) //
    std::int32_t count(const T *in, std::int32_t n, T value, Eq eq) {
  return static_cast<std::int32_t>(
      oneapi::dpl::count_if(oneapi::dpl::execution::device_policy<>{}, in, in + n, [=](T x) { return eq(x, value); }));
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(count_if) //
    std::int32_t count_if(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::count_if(oneapi::dpl::execution::device_policy<>{}, in, in + n, op));
}

template <class T, class Op>     //
POLYREGION_SPECTRA_EXPORT(equal) //
    bool equal(const T *a, const T *b, std::int32_t n, Op op) {
  return oneapi::dpl::equal(oneapi::dpl::execution::device_policy<>{}, a, a + n, b, op);
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(exclusive_scan) //
    void exclusive_scan(const T *in, T *out, std::int32_t n, T init, Op op) {
  oneapi::dpl::exclusive_scan(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, init, op);
}

template <class K, class V, class Eq, class Op>  //
POLYREGION_SPECTRA_EXPORT(exclusive_scan_by_key) //
    void exclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, V init, Eq eq, Op op) {
  // oneDPL 2022.9 default-constructs the supplied key predicate in its device-policy scan path.
  oneapi::dpl::exclusive_scan_by_key(oneapi::dpl::execution::device_policy<>{}, keys, keys + n, vals, out, init, eq, op);
}

template <class T>              //
POLYREGION_SPECTRA_EXPORT(fill) //
    void fill(T *out, std::int32_t n, T value) {
  oneapi::dpl::fill(oneapi::dpl::execution::device_policy<>{}, out, out + n, value);
}

template <class T>                //
POLYREGION_SPECTRA_EXPORT(fill_n) //
    void fill_n(T *out, std::int32_t n, T value) {
  oneapi::dpl::fill_n(oneapi::dpl::execution::device_policy<>{}, out, n, value);
}

template <class T, class Eq>    //
POLYREGION_SPECTRA_EXPORT(find) //
    std::int32_t find(const T *in, std::int32_t n, T value, Eq eq) {
  return static_cast<std::int32_t>(
      oneapi::dpl::find_if(oneapi::dpl::execution::device_policy<>{}, in, in + n, [=](T x) { return eq(x, value); }) - in);
}

template <class T, class Op>       //
POLYREGION_SPECTRA_EXPORT(find_if) //
    std::int32_t find_if(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::find_if(oneapi::dpl::execution::device_policy<>{}, in, in + n, op) - in);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(find_if_not) //
    std::int32_t find_if_not(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::find_if_not(oneapi::dpl::execution::device_policy<>{}, in, in + n, op) - in);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(for_each) //
    void for_each(T *data, std::int32_t n, Op op) {
  oneapi::dpl::for_each(oneapi::dpl::execution::device_policy<>{}, data, data + n, [=](T &x) { x = op(x); });
}

template <class T, class Op>          //
POLYREGION_SPECTRA_EXPORT(for_each_n) //
    void for_each_n(T *data, std::int32_t n, Op op) {
  oneapi::dpl::for_each_n(oneapi::dpl::execution::device_policy<>{}, data, n, [=](T &x) { x = op(x); });
}

template <class T>                //
POLYREGION_SPECTRA_EXPORT(gather) //
    void gather(const std::int32_t *map, std::int32_t n, const T *in, std::int32_t, T *out) {
  const auto source = oneapi::dpl::make_permutation_iterator(in, map);
  oneapi::dpl::copy(oneapi::dpl::execution::device_policy<>{}, source, source + n, out);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(generate) //
    void generate(T *out, std::int32_t n, Op op) {
  oneapi::dpl::generate(oneapi::dpl::execution::device_policy<>{}, out, out + n, op);
}

template <class T, class Op>          //
POLYREGION_SPECTRA_EXPORT(generate_n) //
    void generate_n(T *out, std::int32_t n, Op op) {
  oneapi::dpl::generate_n(oneapi::dpl::execution::device_policy<>{}, out, n, op);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(includes) //
    bool includes(const T *a, std::int32_t na, const T *b, std::int32_t nb, Op op) {
  return oneapi::dpl::includes(oneapi::dpl::execution::device_policy<>{}, a, a + na, b, b + nb, op);
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(inclusive_scan) //
    void inclusive_scan(const T *in, T *out, std::int32_t n, Op op) {
  oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, op);
}

template <class K, class V, class Eq, class Op>  //
POLYREGION_SPECTRA_EXPORT(inclusive_scan_by_key) //
    void inclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, Eq eq, Op op) {
  // oneDPL 2022.9 default-constructs the supplied key predicate in its device-policy scan path.
  oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy<>{}, keys, keys + n, vals, out, eq, op);
}

template <class T, class U, class V, class OpReduce, class OpProduct> //
POLYREGION_SPECTRA_EXPORT(inner_product)                              //
    V inner_product(const T *a, const U *b, std::int32_t n, V init, OpReduce reduce, OpProduct product) {
  return oneapi::dpl::transform_reduce(oneapi::dpl::execution::device_policy<>{}, a, a + n, b, init, reduce, product);
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(is_partitioned) //
    bool is_partitioned(const T *data, std::int32_t n, Op op) {
  return oneapi::dpl::is_partitioned(oneapi::dpl::execution::device_policy<>{}, data, data + n, op);
}

template <class T, class Op>         //
POLYREGION_SPECTRA_EXPORT(is_sorted) //
    bool is_sorted(const T *data, std::int32_t n, Op op) {
  return oneapi::dpl::is_sorted(oneapi::dpl::execution::device_policy<>{}, data, data + n, op);
}

template <class T, class Op>               //
POLYREGION_SPECTRA_EXPORT(is_sorted_until) //
    std::int32_t is_sorted_until(const T *data, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::is_sorted_until(oneapi::dpl::execution::device_policy<>{}, data, data + n, op) - data);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(lower_bound) //
    std::int32_t lower_bound(const T *in, std::int32_t n, T value, Op op) {
  auto policy = oneapi::dpl::execution::device_policy<>{};
  auto queue = policy.queue();
  auto *device_value = sycl::malloc_device<T>(1, queue);
  auto *device_result = sycl::malloc_device<std::int32_t>(1, queue);
  detail::device_allocation<T> valueAllocation(device_value, queue);
  detail::device_allocation<std::int32_t> resultAllocation(device_result, queue);
  queue.memcpy(device_value, &value, sizeof(value)).wait();
  oneapi::dpl::lower_bound(policy, in, in + n, device_value, device_value + 1, device_result, op);
  std::int32_t result = 0;
  queue.memcpy(&result, device_result, sizeof(result)).wait();
  return result;
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(max_element) //
    T max_element(const T *in, std::int32_t n, Op op) {
  if (n <= 0) return T{};
  auto policy = oneapi::dpl::execution::device_policy<>{};
  auto queue = policy.queue();
  const auto *device_result = oneapi::dpl::max_element(policy, in, in + n, op);
  T result{};
  queue.memcpy(&result, device_result, sizeof(result)).wait();
  return result;
}

template <class T, class Op>     //
POLYREGION_SPECTRA_EXPORT(merge) //
    void merge(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  oneapi::dpl::merge(oneapi::dpl::execution::device_policy<>{}, a, a + na, b, b + nb, out, op);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(min_element) //
    T min_element(const T *in, std::int32_t n, Op op) {
  if (n <= 0) return T{};
  auto policy = oneapi::dpl::execution::device_policy<>{};
  auto queue = policy.queue();
  const auto *device_result = oneapi::dpl::min_element(policy, in, in + n, op);
  T result{};
  queue.memcpy(&result, device_result, sizeof(result)).wait();
  return result;
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(minmax_element) //
    void minmax_element(const T *in, std::int32_t n, T *min_out, T *max_out, Op op) {
  auto policy = oneapi::dpl::execution::device_policy<>{};
  if (n <= 0) {
    oneapi::dpl::fill_n(policy, min_out, 1, T{});
    oneapi::dpl::fill_n(policy, max_out, 1, T{});
    return;
  }
  auto queue = policy.queue();
  const auto result = oneapi::dpl::minmax_element(policy, in, in + n, op);
  oneapi::dpl::copy_n(policy, result.first, 1, min_out);
  oneapi::dpl::copy_n(policy, result.second, 1, max_out);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(mismatch) //
    std::int32_t mismatch(const T *a, const T *b, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::mismatch(oneapi::dpl::execution::device_policy<>{}, a, a + n, b, op).first - a);
}

template <class T, class Op>       //
POLYREGION_SPECTRA_EXPORT(none_of) //
    bool none_of(const T *in, std::int32_t n, Op op) {
  return oneapi::dpl::none_of(oneapi::dpl::execution::device_policy<>{}, in, in + n, op);
}

template <class T, class Op>         //
POLYREGION_SPECTRA_EXPORT(partition) //
    void partition(T *data, std::int32_t n, Op op) {
  oneapi::dpl::partition(oneapi::dpl::execution::device_policy<>{}, data, data + n, op);
}

template <class T, class Op>               //
POLYREGION_SPECTRA_EXPORT(partition_point) //
    std::int32_t partition_point(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::find_if_not(oneapi::dpl::execution::device_policy<>{}, in, in + n, op) - in);
}

template <class T, class Op>      //
POLYREGION_SPECTRA_EXPORT(reduce) //
    T reduce(const T *in, std::int32_t n, T init, Op op) {
  return oneapi::dpl::reduce(oneapi::dpl::execution::device_policy<>{}, in, in + n, init, op);
}

template <class K, class V, class Eq, class Op> //
POLYREGION_SPECTRA_EXPORT(reduce_by_key)        //
    std::int32_t reduce_by_key(const K *keys, const V *vals, K *kout, V *vout, std::int32_t n, Eq eq, Op op) {
  const auto result = oneapi::dpl::reduce_by_key(oneapi::dpl::execution::device_policy<>{}, keys, keys + n, vals, kout, vout, eq, op);
  return static_cast<std::int32_t>(result.first - kout);
}

template <class T, class Eq>      //
POLYREGION_SPECTRA_EXPORT(remove) //
    std::int32_t remove(T *data, std::int32_t n, T value, Eq eq) {
  struct Predicate {
    T value;
    Eq eq;
    bool operator()(T x) const { return eq(x, value); }
  };
  return static_cast<std::int32_t>(oneapi::dpl::remove_if(oneapi::dpl::execution::device_policy<>{}, data, data + n, Predicate{value, eq})
                                   - data);
}

template <class T, class Eq>           //
POLYREGION_SPECTRA_EXPORT(remove_copy) //
    std::int32_t remove_copy(const T *in, std::int32_t n, T *out, T value, Eq eq) {
  struct Predicate {
    T value;
    Eq eq;
    bool operator()(T x) const { return eq(x, value); }
  };
  return static_cast<std::int32_t>(
      oneapi::dpl::remove_copy_if(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, Predicate{value, eq}) - out);
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(remove_copy_if) //
    std::int32_t remove_copy_if(const T *in, std::int32_t n, T *out, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::remove_copy_if(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, op) - out);
}

template <class T, class Op>         //
POLYREGION_SPECTRA_EXPORT(remove_if) //
    std::int32_t remove_if(T *data, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::remove_if(oneapi::dpl::execution::device_policy<>{}, data, data + n, op) - data);
}

template <class T, class Eq>       //
POLYREGION_SPECTRA_EXPORT(replace) //
    void replace(T *data, std::int32_t n, T old_value, T new_value, Eq eq) {
  oneapi::dpl::replace_if(oneapi::dpl::execution::device_policy<>{}, data, data + n, [=](T x) { return eq(x, old_value); }, new_value);
}

template <class T, class Eq>            //
POLYREGION_SPECTRA_EXPORT(replace_copy) //
    void replace_copy(const T *in, T *out, std::int32_t n, T old_value, T new_value, Eq eq) {
  oneapi::dpl::replace_copy_if(
      oneapi::dpl::execution::device_policy<>{}, in, in + n, out, [=](T x) { return eq(x, old_value); }, new_value);
}

template <class T, class Op>               //
POLYREGION_SPECTRA_EXPORT(replace_copy_if) //
    void replace_copy_if(const T *in, T *out, std::int32_t n, T new_value, Op op) {
  oneapi::dpl::replace_copy_if(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, op, new_value);
}

template <class T, class Op>          //
POLYREGION_SPECTRA_EXPORT(replace_if) //
    void replace_if(T *data, std::int32_t n, T new_value, Op op) {
  oneapi::dpl::replace_if(oneapi::dpl::execution::device_policy<>{}, data, data + n, op, new_value);
}

template <class T>                 //
POLYREGION_SPECTRA_EXPORT(reverse) //
    void reverse(T *data, std::int32_t n) {
  oneapi::dpl::reverse(oneapi::dpl::execution::device_policy<>{}, data, data + n);
}

template <class T>                      //
POLYREGION_SPECTRA_EXPORT(reverse_copy) //
    void reverse_copy(const T *in, std::int32_t n, T *out) {
  oneapi::dpl::reverse_copy(oneapi::dpl::execution::device_policy<>{}, in, in + n, out);
}

template <class T>                 //
POLYREGION_SPECTRA_EXPORT(scatter) //
    void scatter(const T *in, std::int32_t n, const std::int32_t *map, T *out, std::int32_t) {
  const auto destination = oneapi::dpl::make_permutation_iterator(out, map);
  oneapi::dpl::copy(oneapi::dpl::execution::device_policy<>{}, in, in + n, destination);
}

template <class T, class Op>      //
POLYREGION_SPECTRA_EXPORT(search) //
    std::int32_t search(const T *in, std::int32_t n, const T *sub, std::int32_t m, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::search(oneapi::dpl::execution::device_policy<>{}, in, in + n, sub, sub + m, op) - in);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(search_n) //
    std::int32_t search_n(const T *in, std::int32_t n, std::int32_t count, T value, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::search_n(oneapi::dpl::execution::device_policy<>{}, in, in + n, count, value, op) - in);
}

template <class T, class ValueAt>   //
POLYREGION_SPECTRA_EXPORT(sequence) //
    void sequence(T *out, std::int32_t n, T init, T step, ValueAt value_at) {
  oneapi::dpl::transform(oneapi::dpl::execution::device_policy<>{}, oneapi::dpl::counting_iterator<std::int32_t>(0),
                         oneapi::dpl::counting_iterator<std::int32_t>(n), out,
                         [=](std::int32_t index) { return value_at(init, step, index); });
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(set_difference) //
    std::int32_t set_difference(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::set_difference(oneapi::dpl::execution::device_policy<>{}, a, a + na, b, b + nb, out, op)
                                   - out);
}

template <class T, class Op>                //
POLYREGION_SPECTRA_EXPORT(set_intersection) //
    std::int32_t set_intersection(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::set_intersection(oneapi::dpl::execution::device_policy<>{}, a, a + na, b, b + nb, out, op)
                                   - out);
}

template <class T, class Op>         //
POLYREGION_SPECTRA_EXPORT(set_union) //
    std::int32_t set_union(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(oneapi::dpl::set_union(oneapi::dpl::execution::device_policy<>{}, a, a + na, b, b + nb, out, op) - out);
}

template <class T, class Op>    //
POLYREGION_SPECTRA_EXPORT(sort) //
    void sort(T *data, std::int32_t n, Op op) {
  oneapi::dpl::sort(oneapi::dpl::execution::device_policy<>{}, data, data + n, op);
}

template <class K, class V, class Op>  //
POLYREGION_SPECTRA_EXPORT(sort_by_key) //
    void sort_by_key(K *keys, V *values, std::int32_t n, Op op) {
  oneapi::dpl::sort_by_key(oneapi::dpl::execution::device_policy<>{}, keys, keys + n, values, op);
}

template <class T, class Op>                //
POLYREGION_SPECTRA_EXPORT(stable_partition) //
    void stable_partition(T *data, std::int32_t n, Op op) {
  oneapi::dpl::stable_partition(oneapi::dpl::execution::device_policy<>{}, data, data + n, op);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(stable_sort) //
    void stable_sort(T *data, std::int32_t n, Op op) {
  oneapi::dpl::stable_sort(oneapi::dpl::execution::device_policy<>{}, data, data + n, op);
}

template <class T>                     //
POLYREGION_SPECTRA_EXPORT(swap_ranges) //
    void swap_ranges(T *a, std::int32_t n, T *b) {
  oneapi::dpl::swap_ranges(oneapi::dpl::execution::device_policy<>{}, a, a + n, b);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(tabulate) //
    void tabulate(T *out, std::int32_t n, Op op) {
  oneapi::dpl::transform(oneapi::dpl::execution::device_policy<>{}, oneapi::dpl::counting_iterator<std::int32_t>(0),
                         oneapi::dpl::counting_iterator<std::int32_t>(n), out, op);
}

template <class T, class U, class Op> //
POLYREGION_SPECTRA_EXPORT(transform)  //
    void transform(const T *in, U *out, std::int32_t n, Op op) {
  oneapi::dpl::transform(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, op);
}

template <class T, class U, class V, class Op> //
POLYREGION_SPECTRA_EXPORT(transform_binary)    //
    void transform_binary(const T *a, const U *b, V *out, std::int32_t n, Op op) {
  oneapi::dpl::transform(oneapi::dpl::execution::device_policy<>{}, a, a + n, b, out, op);
}

template <class T, class U, class Map, class Op>    //
POLYREGION_SPECTRA_EXPORT(transform_exclusive_scan) //
    void transform_exclusive_scan(const T *in, U *out, std::int32_t n, U init, Map map, Op op) {
  const auto first = oneapi::dpl::make_transform_iterator(in, map);
  oneapi::dpl::exclusive_scan(oneapi::dpl::execution::device_policy<>{}, first, first + n, out, init, op);
}

template <class T, class U, class Map, class Op>    //
POLYREGION_SPECTRA_EXPORT(transform_inclusive_scan) //
    void transform_inclusive_scan(const T *in, U *out, std::int32_t n, Map map, Op op) {
  const auto first = oneapi::dpl::make_transform_iterator(in, map);
  oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy<>{}, first, first + n, out, op);
}

template <class T, class U, class Map, class Op> //
POLYREGION_SPECTRA_EXPORT(transform_reduce)      //
    U transform_reduce(const T *in, std::int32_t n, U init, Map map, Op op) {
  return oneapi::dpl::transform_reduce(oneapi::dpl::execution::device_policy<>{}, in, in + n, init, op, map);
}

template <class T, class Eq>      //
POLYREGION_SPECTRA_EXPORT(unique) //
    std::int32_t unique(T *data, std::int32_t n, Eq eq) {
  return static_cast<std::int32_t>(oneapi::dpl::unique(oneapi::dpl::execution::device_policy<>{}, data, data + n, eq) - data);
}

template <class T, class Eq>           //
POLYREGION_SPECTRA_EXPORT(unique_copy) //
    std::int32_t unique_copy(const T *in, std::int32_t n, T *out, Eq eq) {
  return static_cast<std::int32_t>(oneapi::dpl::unique_copy(oneapi::dpl::execution::device_policy<>{}, in, in + n, out, eq) - out);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(upper_bound) //
    std::int32_t upper_bound(const T *in, std::int32_t n, T value, Op op) {
  auto policy = oneapi::dpl::execution::device_policy<>{};
  auto queue = policy.queue();
  auto *device_value = sycl::malloc_device<T>(1, queue);
  auto *device_result = sycl::malloc_device<std::int32_t>(1, queue);
  detail::device_allocation<T> valueAllocation(device_value, queue);
  detail::device_allocation<std::int32_t> resultAllocation(device_result, queue);
  queue.memcpy(device_value, &value, sizeof(value)).wait();
  oneapi::dpl::upper_bound(policy, in, in + n, device_value, device_value + 1, device_result, op);
  std::int32_t result = 0;
  queue.memcpy(&result, device_result, sizeof(result)).wait();
  return result;
}

} // namespace spectra::implementation

#ifdef POLYREGION_SPECTRA_UNDEF_EXPORT
  #undef POLYREGION_SPECTRA_UNDEF_EXPORT
  #undef POLYREGION_SPECTRA_EXPORT
#endif
