#pragma once

#ifndef POLYREGION_SPECTRA_EXPORT
  #define POLYREGION_SPECTRA_EXPORT(name)
  #define POLYREGION_SPECTRA_UNDEF_EXPORT
#endif
#ifndef POLYREGION_SPECTRA_CORE_EXPORT
  #define POLYREGION_SPECTRA_CORE_EXPORT(name)
  #define POLYREGION_SPECTRA_UNDEF_CORE_EXPORT
#endif

#include <cstdint>
#include <new>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/merge.h>
#include <thrust/mismatch.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/swap.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#if defined(__HIP__)
  #include <rocprim/rocprim.hpp>
#else
  #include <cub/util_ptx.cuh>
#endif

namespace spectra::implementation {

template <class F, class... Args>
[[clang::annotate("__polyregion_element")]] __host__ __device__ decltype(auto) element(F &f, Args... args) {
  return f(args...);
}

template <class T, class Map> struct element_map {
  Map map;
  __host__ __device__ T operator()(std::int64_t i) const { return element(map, i); }
};

template <class T, class Source, class Map> struct source_map {
  Source in;
  Map map;
  __host__ __device__ T operator()(std::int64_t i) const { return map(in[i]); }
};

template <class T, class Source> struct source_element {
  Source in;
  __host__ __device__ T operator()(std::int64_t i) const { return in[i]; }
};

constexpr std::int32_t ordered_block = 256;
constexpr std::int32_t ordered_warp = 32;
constexpr std::int32_t ordered_warps = ordered_block / ordered_warp;
constexpr std::int64_t ordered_tile_bytes = 65536;
constexpr std::int64_t ordered_target_blocks = 1024;

inline void synchronise_device() {
#if defined(__HIP__)
  hipDeviceSynchronize();
#else
  cudaDeviceSynchronize();
#endif
}

template <class T> std::int32_t ordered_rows(std::int64_t n) {
  constexpr std::int64_t max_rows = sizeof(T) * ordered_block < ordered_tile_bytes ? ordered_tile_bytes / (sizeof(T) * ordered_block) : 1;
  const std::int64_t single = (n + ordered_block - 1) / ordered_block;
  if (single <= max_rows) return static_cast<std::int32_t>(single);
  const std::int64_t spread = (n + ordered_block * ordered_target_blocks - 1) / (ordered_block * ordered_target_blocks);
  return static_cast<std::int32_t>(spread < max_rows ? spread : max_rows);
}

template <class T> __device__ T shuffle_down(T value, std::int32_t offset) {
#if defined(__HIP__)
  return rocprim::warp_shuffle_down(value, offset, ordered_warp);
#else
  return cub::ShuffleDown<ordered_warp>(value, offset, ordered_warp - 1, 0xffffffffu);
#endif
}

template <class T> struct shared_totals {
  alignas(T) unsigned char bytes[ordered_warps * sizeof(T)];

  __device__ void store(std::int32_t index, const T &value) { new (bytes + index * sizeof(T)) T(value); }
  __device__ const T &load(std::int32_t index) const { return *reinterpret_cast<const T *>(bytes + index * sizeof(T)); }
};

template <class T, class Element, class Op, bool Final>
__global__ void ordered_reduce_kernel(Element element_at, std::int64_t n, std::int32_t rows, Op op, T init, T *out) {
  __shared__ shared_totals<T> totals;
  const std::int32_t lane = threadIdx.x % ordered_warp;
  const std::int32_t warp = threadIdx.x / ordered_warp;
  const std::int64_t span = std::int64_t(ordered_warp) * rows;
  const std::int64_t tile = std::int64_t(ordered_warps) * span;
  const std::int64_t begin = std::int64_t(blockIdx.x) * tile + std::int64_t(warp) * span;
  const std::int64_t remaining = n - begin;
  const std::int32_t row_count = remaining <= 0      ? 0
                                 : remaining >= span ? rows
                                                     : std::int32_t((remaining + ordered_warp - 1) / ordered_warp);
  T acc{};
  if (row_count > 0) {
    const auto load = [&](std::int32_t row) {
      const std::int64_t left = remaining - std::int64_t(row) * ordered_warp;
      const std::int32_t count = left < ordered_warp ? std::int32_t(left) : ordered_warp;
      return element_at(begin + std::int64_t(row) * ordered_warp + (lane < count ? lane : count - 1));
    };
    T current = load(0);
    for (std::int32_t row = 0; row < row_count; ++row) {
      const std::int64_t left = remaining - std::int64_t(row) * ordered_warp;
      const std::int32_t count = left < ordered_warp ? std::int32_t(left) : ordered_warp;
      T next = current;
      if (row + 1 < row_count) next = load(row + 1);
      T folded = current;
      for (std::int32_t offset = 1; offset < ordered_warp; offset <<= 1) {
        const T other = shuffle_down(folded, offset);
        if (lane + offset < count) folded = op(folded, other);
      }
      if (lane == 0) {
        if (row == 0) acc = folded;
        else acc = op(acc, folded);
      }
      current = next;
    }
  }
  if (lane == 0) totals.store(warp, acc);
  __syncthreads();
  if (threadIdx.x == 0) {
    const std::int64_t left = n - std::int64_t(blockIdx.x) * tile;
    const std::int32_t warps = left >= tile ? ordered_warps : std::int32_t((left + span - 1) / span);
    T total = totals.load(0);
    for (std::int32_t i = 1; i < warps; ++i)
      total = op(total, totals.load(i));
    if (Final) total = op(init, total);
    out[blockIdx.x] = total;
  }
}

template <class T> T *device_buffer(std::int64_t count) {
  return static_cast<T *>(thrust::raw_pointer_cast(thrust::malloc(thrust::device, static_cast<std::size_t>(count) * sizeof(T))));
}

template <class T, class Element, class Op> T ordered_reduce(std::int64_t n, T init, Element element_at, Op op) {
  if (n <= 0) return init;
  const std::int32_t rows = ordered_rows<T>(n);
  const std::int64_t tile = std::int64_t(ordered_block) * rows;
  const std::int64_t blocks = (n + tile - 1) / tile;
  T *partial = device_buffer<T>(blocks);
  if (blocks == 1) {
    ordered_reduce_kernel<T, Element, Op, true><<<1, ordered_block>>>(element_at, n, rows, op, init, partial);
    synchronise_device();
    const T result = *thrust::device_ptr<T>(partial);
    thrust::free(thrust::device, partial);
    return result;
  }
  ordered_reduce_kernel<T, Element, Op, false><<<static_cast<unsigned>(blocks), ordered_block>>>(element_at, n, rows, op, init, partial);
  const T result = ordered_reduce(blocks, init, source_element<T, const T *>{partial}, op);
  thrust::free(thrust::device, partial);
  return result;
}

template <class Body> void core_for_each(std::int64_t n, Body body) {
  thrust::for_each_n(thrust::device, thrust::counting_iterator<std::int64_t>(0), n,
                     [=] __host__ __device__(std::int64_t i) { element(body, i); });
}

template <class T, class Map, class Combine>     //
POLYREGION_SPECTRA_CORE_EXPORT(transform_reduce) //
    T core_transform_reduce(std::int64_t n, T init, Map map, Combine combine) {
  return ordered_reduce(n, init, element_map<T, Map>{map}, combine);
}

template <class T, class Map, class Combine>               //
POLYREGION_SPECTRA_CORE_EXPORT(transform_reduce_unordered) //
    T core_transform_reduce_unordered(std::int64_t n, T init, Map map, Combine combine) {
  return thrust::transform_reduce(thrust::device, thrust::counting_iterator<std::int64_t>(0), thrust::counting_iterator<std::int64_t>(n),
                                  element_map<T, Map>{map}, init, combine);
}

template <class T, class Op>                   //
POLYREGION_SPECTRA_EXPORT(adjacent_difference) //
    void adjacent_difference(const T *in, T *out, std::int32_t n, Op op) {
  if (n <= 0) return;
  thrust::transform(thrust::device, in + 1, in + n, in, out + 1, op);
  thrust::copy_n(thrust::device, in, 1, out);
}

template <class T, class Op>      //
POLYREGION_SPECTRA_EXPORT(all_of) //
    bool all_of(const T *in, std::int32_t n, Op op) {
  const auto count = thrust::transform_reduce(
      thrust::device, in, in + n, [=] __host__ __device__(T x) -> std::int32_t { return op(x) ? 1 : 0; }, std::int32_t{0},
      [] __host__ __device__(std::int32_t a, std::int32_t b) { return a + b; });
  return count == n;
}

template <class T, class Op>      //
POLYREGION_SPECTRA_EXPORT(any_of) //
    bool any_of(const T *in, std::int32_t n, Op op) {
  const auto count = thrust::transform_reduce(
      thrust::device, in, in + n, [=] __host__ __device__(T x) -> std::int32_t { return op(x) ? 1 : 0; }, std::int32_t{0},
      [] __host__ __device__(std::int32_t a, std::int32_t b) { return a + b; });
  return count != 0;
}

template <class T, class Op>             //
POLYREGION_SPECTRA_EXPORT(binary_search) //
    bool binary_search(const T *in, std::int32_t n, T value, Op op) {
  return thrust::binary_search(thrust::device, in, in + n, value, op);
}

template <class T>              //
POLYREGION_SPECTRA_EXPORT(copy) //
    void copy(const T *in, std::int32_t n, T *out) {
  thrust::copy(thrust::device, in, in + n, out);
}

template <class T, class Op>       //
POLYREGION_SPECTRA_EXPORT(copy_if) //
    std::int32_t copy_if(const T *in, std::int32_t n, T *out, Op op) {
  return static_cast<std::int32_t>(thrust::copy_if(thrust::device, in, in + n, out, op) - out);
}

template <class T>                //
POLYREGION_SPECTRA_EXPORT(copy_n) //
    void copy_n(const T *in, std::int32_t n, T *out) {
  thrust::copy_n(thrust::device, in, n, out);
}

template <class T, class Eq>     //
POLYREGION_SPECTRA_EXPORT(count) //
    std::int32_t count(const T *in, std::int32_t n, T value, Eq eq) {
  return thrust::transform_reduce(
      thrust::device, in, in + n, [=] __host__ __device__(T x) -> std::int32_t { return eq(x, value) ? 1 : 0; }, std::int32_t{0},
      [] __host__ __device__(std::int32_t a, std::int32_t b) { return a + b; });
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(count_if) //
    std::int32_t count_if(const T *in, std::int32_t n, Op op) {
  return thrust::transform_reduce(
      thrust::device, in, in + n, [=] __host__ __device__(T x) -> std::int32_t { return op(x) ? 1 : 0; }, std::int32_t{0},
      [] __host__ __device__(std::int32_t a, std::int32_t b) { return a + b; });
}

template <class T, class Op>     //
POLYREGION_SPECTRA_EXPORT(equal) //
    bool equal(const T *a, const T *b, std::int32_t n, Op op) {
  return thrust::equal(thrust::device, a, a + n, b, op);
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(exclusive_scan) //
    void exclusive_scan(const T *in, T *out, std::int32_t n, T init, Op op) {
  thrust::exclusive_scan(thrust::device, in, in + n, out, init, op);
}

template <class K, class V, class Eq, class Op>  //
POLYREGION_SPECTRA_EXPORT(exclusive_scan_by_key) //
    void exclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, V init, Eq eq, Op op) {
  thrust::exclusive_scan_by_key(thrust::device, keys, keys + n, vals, out, init, eq, op);
}

template <class T>              //
POLYREGION_SPECTRA_EXPORT(fill) //
    void fill(T *out, std::int32_t n, T value) {
  thrust::fill(thrust::device, out, out + n, value);
}

template <class T>                //
POLYREGION_SPECTRA_EXPORT(fill_n) //
    void fill_n(T *out, std::int32_t n, T value) {
  thrust::fill_n(thrust::device, out, n, value);
}

template <class T, class Eq>    //
POLYREGION_SPECTRA_EXPORT(find) //
    std::int32_t find(const T *in, std::int32_t n, T value, Eq eq) {
  return static_cast<std::int32_t>(thrust::find_if(thrust::device, in, in + n, [=] __host__ __device__(T x) { return eq(x, value); }) - in);
}

template <class T, class Op>       //
POLYREGION_SPECTRA_EXPORT(find_if) //
    std::int32_t find_if(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(thrust::find_if(thrust::device, in, in + n, op) - in);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(find_if_not) //
    std::int32_t find_if_not(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(thrust::find_if_not(thrust::device, in, in + n, op) - in);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(for_each) //
    void for_each(T *data, std::int32_t n, Op op) {
  thrust::for_each(thrust::device, data, data + n, [=] __host__ __device__(T & x) { x = op(x); });
}

template <class T, class Op>          //
POLYREGION_SPECTRA_EXPORT(for_each_n) //
    void for_each_n(T *data, std::int32_t n, Op op) {
  thrust::for_each_n(thrust::device, data, n, [=] __host__ __device__(T & x) { x = op(x); });
}

template <class T>                //
POLYREGION_SPECTRA_EXPORT(gather) //
    void gather(const std::int32_t *map, std::int32_t n, const T *in, std::int32_t, T *out) {
  thrust::gather(thrust::device, map, map + n, in, out);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(generate) //
    void generate(T *out, std::int32_t n, Op op) {
  thrust::generate(thrust::device, out, out + n, op);
}

template <class T, class Op>          //
POLYREGION_SPECTRA_EXPORT(generate_n) //
    void generate_n(T *out, std::int32_t n, Op op) {
  thrust::generate_n(thrust::device, out, n, op);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(includes) //
    bool includes(const T *a, std::int32_t na, const T *b, std::int32_t nb, Op op) {
  const auto output = thrust::make_discard_iterator();
  return thrust::set_difference(thrust::device, b, b + nb, a, a + na, output, op) == output;
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(inclusive_scan) //
    void inclusive_scan(const T *in, T *out, std::int32_t n, Op op) {
  thrust::inclusive_scan(thrust::device, in, in + n, out, op);
}

template <class K, class V, class Eq, class Op>  //
POLYREGION_SPECTRA_EXPORT(inclusive_scan_by_key) //
    void inclusive_scan_by_key(const K *keys, const V *vals, V *out, std::int32_t n, Eq eq, Op op) {
  thrust::inclusive_scan_by_key(thrust::device, keys, keys + n, vals, out, eq, op);
}

template <class T, class U, class V, class OpReduce, class OpProduct> //
POLYREGION_SPECTRA_EXPORT(inner_product)                              //
    V inner_product(const T *a, const U *b, std::int32_t n, V init, OpReduce reduce, OpProduct product) {
  return thrust::inner_product(thrust::device, a, a + n, b, init, reduce, product);
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(is_partitioned) //
    bool is_partitioned(const T *data, std::int32_t n, Op op) {
  return thrust::is_partitioned(thrust::device, data, data + n, op);
}

template <class T, class Op>         //
POLYREGION_SPECTRA_EXPORT(is_sorted) //
    bool is_sorted(const T *data, std::int32_t n, Op op) {
  return thrust::is_sorted(thrust::device, data, data + n, op);
}

template <class T, class Op>               //
POLYREGION_SPECTRA_EXPORT(is_sorted_until) //
    std::int32_t is_sorted_until(const T *data, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(thrust::is_sorted_until(thrust::device, data, data + n, op) - data);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(lower_bound) //
    std::int32_t lower_bound(const T *in, std::int32_t n, T value, Op op) {
  return static_cast<std::int32_t>(thrust::lower_bound(thrust::device, in, in + n, value, op) - in);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(max_element) //
    std::int32_t max_element(const T *in, std::int32_t n, Op op) {
  if (n <= 0) return 0;
  return static_cast<std::int32_t>(thrust::max_element(thrust::device, in, in + n, op) - in);
}

template <class T, class Op>     //
POLYREGION_SPECTRA_EXPORT(merge) //
    void merge(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  thrust::merge(thrust::device, a, a + na, b, b + nb, out, op);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(min_element) //
    std::int32_t min_element(const T *in, std::int32_t n, Op op) {
  if (n <= 0) return 0;
  return static_cast<std::int32_t>(thrust::min_element(thrust::device, in, in + n, op) - in);
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(minmax_element) //
    void minmax_element(const T *in, std::int32_t n, std::int32_t *min_out, std::int32_t *max_out, Op op) {
  if (n <= 0) {
    thrust::fill_n(thrust::device, min_out, 1, std::int32_t{0});
    thrust::fill_n(thrust::device, max_out, 1, std::int32_t{0});
    return;
  }
  const auto last = thrust::make_reverse_iterator(in + n);
  const auto lo = thrust::min_element(thrust::device, in, in + n, op) - in;
  const auto hi = n - 1 - (thrust::max_element(thrust::device, last, last + n, op) - last);
  thrust::fill_n(thrust::device, min_out, 1, static_cast<std::int32_t>(lo));
  thrust::fill_n(thrust::device, max_out, 1, static_cast<std::int32_t>(hi));
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(mismatch) //
    std::int32_t mismatch(const T *a, const T *b, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(thrust::mismatch(thrust::device, a, a + n, b, op).first - a);
}

template <class T, class Op>       //
POLYREGION_SPECTRA_EXPORT(none_of) //
    bool none_of(const T *in, std::int32_t n, Op op) {
  const auto count = thrust::transform_reduce(
      thrust::device, in, in + n, [=] __host__ __device__(T x) -> std::int32_t { return op(x) ? 1 : 0; }, std::int32_t{0},
      [] __host__ __device__(std::int32_t a, std::int32_t b) { return a + b; });
  return count == 0;
}

template <class T, class Op>         //
POLYREGION_SPECTRA_EXPORT(partition) //
    std::int32_t partition(T *data, std::int32_t n, Op op) {
  thrust::stable_sort(thrust::device, data, data + n, [=] __host__ __device__(const T &a, const T &b) { return op(a) && !op(b); });
  return static_cast<std::int32_t>(thrust::partition_point(thrust::device, data, data + n, op) - data);
}

template <class T, class Op>               //
POLYREGION_SPECTRA_EXPORT(partition_point) //
    std::int32_t partition_point(const T *in, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(thrust::partition_point(thrust::device, in, in + n, op) - in);
}

template <class T, class Op>      //
POLYREGION_SPECTRA_EXPORT(reduce) //
    T reduce(const T *in, std::int32_t n, T init, Op op) {
  return ordered_reduce(n, init, source_element<T, const T *>{in}, op);
}

template <class T, class Op>                //
POLYREGION_SPECTRA_EXPORT(reduce_unordered) //
    T reduce_unordered(const T *in, std::int32_t n, T init, Op op) {
  return thrust::reduce(thrust::device, in, in + n, init, op);
}

template <class K, class V, class Eq, class Op> //
POLYREGION_SPECTRA_EXPORT(reduce_by_key)        //
    std::int32_t reduce_by_key(const K *keys, const V *vals, K *kout, V *vout, std::int32_t n, Eq eq, Op op) {
  const auto result = thrust::reduce_by_key(thrust::device, keys, keys + n, vals, kout, vout, eq, op);
  return static_cast<std::int32_t>(result.first - kout);
}

template <class T, class Eq>      //
POLYREGION_SPECTRA_EXPORT(remove) //
    std::int32_t remove(T *data, std::int32_t n, T value, Eq eq) {
  return static_cast<std::int32_t>(thrust::remove_if(thrust::device, data, data + n, [=] __host__ __device__(T x) { return eq(x, value); })
                                   - data);
}

template <class T, class Eq>           //
POLYREGION_SPECTRA_EXPORT(remove_copy) //
    std::int32_t remove_copy(const T *in, std::int32_t n, T *out, T value, Eq eq) {
  return static_cast<std::int32_t>(
      thrust::remove_copy_if(thrust::device, in, in + n, out, [=] __host__ __device__(T x) { return eq(x, value); }) - out);
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(remove_copy_if) //
    std::int32_t remove_copy_if(const T *in, std::int32_t n, T *out, Op op) {
  return static_cast<std::int32_t>(thrust::remove_copy_if(thrust::device, in, in + n, out, op) - out);
}

template <class T, class Op>         //
POLYREGION_SPECTRA_EXPORT(remove_if) //
    std::int32_t remove_if(T *data, std::int32_t n, Op op) {
  return static_cast<std::int32_t>(thrust::remove_if(thrust::device, data, data + n, op) - data);
}

template <class T, class Eq>       //
POLYREGION_SPECTRA_EXPORT(replace) //
    void replace(T *data, std::int32_t n, T old_value, T new_value, Eq eq) {
  thrust::replace_if(thrust::device, data, data + n, [=] __host__ __device__(T x) { return eq(x, old_value); }, new_value);
}

template <class T, class Eq>            //
POLYREGION_SPECTRA_EXPORT(replace_copy) //
    void replace_copy(const T *in, T *out, std::int32_t n, T old_value, T new_value, Eq eq) {
  thrust::replace_copy_if(thrust::device, in, in + n, out, [=] __host__ __device__(T x) { return eq(x, old_value); }, new_value);
}

template <class T, class Op>               //
POLYREGION_SPECTRA_EXPORT(replace_copy_if) //
    void replace_copy_if(const T *in, T *out, std::int32_t n, T new_value, Op op) {
  thrust::replace_copy_if(thrust::device, in, in + n, out, op, new_value);
}

template <class T, class Op>          //
POLYREGION_SPECTRA_EXPORT(replace_if) //
    void replace_if(T *data, std::int32_t n, T new_value, Op op) {
  thrust::replace_if(thrust::device, data, data + n, op, new_value);
}

template <class T>                 //
POLYREGION_SPECTRA_EXPORT(reverse) //
    void reverse(T *data, std::int32_t n) {
  thrust::for_each(thrust::device, thrust::make_counting_iterator<std::int32_t>(0), thrust::make_counting_iterator<std::int32_t>(n / 2),
                   [=] __host__ __device__(std::int32_t i) {
                     T value = data[i];
                     data[i] = data[n - i - 1];
                     data[n - i - 1] = value;
                   });
}

template <class T>                      //
POLYREGION_SPECTRA_EXPORT(reverse_copy) //
    void reverse_copy(const T *in, std::int32_t n, T *out) {
  thrust::for_each(thrust::device, thrust::make_counting_iterator<std::int32_t>(0), thrust::make_counting_iterator<std::int32_t>(n),
                   [=] __host__ __device__(std::int32_t i) { out[i] = in[n - i - 1]; });
}

template <class T>                 //
POLYREGION_SPECTRA_EXPORT(scatter) //
    void scatter(const T *in, std::int32_t n, const std::int32_t *map, T *out, std::int32_t) {
  thrust::scatter(thrust::device, in, in + n, map, out);
}

template <class T, class Op>      //
POLYREGION_SPECTRA_EXPORT(search) //
    std::int32_t search(const T *in, std::int32_t n, const T *sub, std::int32_t m, Op op) {
  if (m <= 0) return 0;
  if (m > n) return n;
  const auto first = thrust::counting_iterator<std::int32_t>(0);
  return static_cast<std::int32_t>(thrust::find_if(thrust::device, first, first + (n - m + 1),
                                                   [=] __host__ __device__(std::int32_t index) {
                                                     for (std::int32_t offset = 0; offset < m; ++offset)
                                                       if (!op(in[index + offset], sub[offset])) return false;
                                                     return true;
                                                   })
                                   - first);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(search_n) //
    std::int32_t search_n(const T *in, std::int32_t n, std::int32_t count, T value, Op op) {
  if (count <= 0) return 0;
  if (count > n) return n;
  const auto first = thrust::counting_iterator<std::int32_t>(0);
  return static_cast<std::int32_t>(thrust::find_if(thrust::device, first, first + (n - count + 1),
                                                   [=] __host__ __device__(std::int32_t index) {
                                                     for (std::int32_t offset = 0; offset < count; ++offset)
                                                       if (!op(in[index + offset], value)) return false;
                                                     return true;
                                                   })
                                   - first);
}

template <class T, class ValueAt>   //
POLYREGION_SPECTRA_EXPORT(sequence) //
    void sequence(T *out, std::int32_t n, T init, T step, ValueAt value_at) {
  thrust::tabulate(thrust::device, out, out + n, [=] __host__ __device__(std::int32_t index) { return value_at(init, step, index); });
}

template <class T, class Op>              //
POLYREGION_SPECTRA_EXPORT(set_difference) //
    std::int32_t set_difference(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(thrust::set_difference(thrust::device, a, a + na, b, b + nb, out, op) - out);
}

template <class T, class Op>                //
POLYREGION_SPECTRA_EXPORT(set_intersection) //
    std::int32_t set_intersection(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(thrust::set_intersection(thrust::device, a, a + na, b, b + nb, out, op) - out);
}

template <class T, class Op>         //
POLYREGION_SPECTRA_EXPORT(set_union) //
    std::int32_t set_union(const T *a, std::int32_t na, const T *b, std::int32_t nb, T *out, Op op) {
  return static_cast<std::int32_t>(thrust::set_union(thrust::device, a, a + na, b, b + nb, out, op) - out);
}

template <class T, class Op>    //
POLYREGION_SPECTRA_EXPORT(sort) //
    void sort(T *data, std::int32_t n, Op op) {
  thrust::sort(thrust::device, data, data + n, op);
}

template <class K, class V, class Op>  //
POLYREGION_SPECTRA_EXPORT(sort_by_key) //
    void sort_by_key(K *keys, V *values, std::int32_t n, Op op) {
  thrust::sort_by_key(thrust::device, keys, keys + n, values, op);
}

template <class T, class Op>                //
POLYREGION_SPECTRA_EXPORT(stable_partition) //
    std::int32_t stable_partition(T *data, std::int32_t n, Op op) {
  thrust::stable_sort(thrust::device, data, data + n, [=] __host__ __device__(const T &a, const T &b) { return op(a) && !op(b); });
  return static_cast<std::int32_t>(thrust::partition_point(thrust::device, data, data + n, op) - data);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(stable_sort) //
    void stable_sort(T *data, std::int32_t n, Op op) {
  thrust::stable_sort(thrust::device, data, data + n, op);
}

template <class T>                     //
POLYREGION_SPECTRA_EXPORT(swap_ranges) //
    void swap_ranges(T *a, std::int32_t n, T *b) {
  thrust::swap_ranges(thrust::device, a, a + n, b);
}

template <class T, class Op>        //
POLYREGION_SPECTRA_EXPORT(tabulate) //
    void tabulate(T *out, std::int32_t n, Op op) {
  thrust::tabulate(thrust::device, out, out + n, op);
}

template <class T, class U, class Op> //
POLYREGION_SPECTRA_EXPORT(transform)  //
    void transform(const T *in, U *out, std::int32_t n, Op op) {
  thrust::transform(thrust::device, in, in + n, out, op);
}

template <class T, class U, class V, class Op> //
POLYREGION_SPECTRA_EXPORT(transform_binary)    //
    void transform_binary(const T *a, const U *b, V *out, std::int32_t n, Op op) {
  thrust::transform(thrust::device, a, a + n, b, out, op);
}

template <class T, class U, class Map, class Op>    //
POLYREGION_SPECTRA_EXPORT(transform_exclusive_scan) //
    void transform_exclusive_scan(const T *in, U *out, std::int32_t n, U init, Map map, Op op) {
  thrust::transform_exclusive_scan(thrust::device, in, in + n, out, map, init, op);
}

template <class T, class U, class Map, class Op>    //
POLYREGION_SPECTRA_EXPORT(transform_inclusive_scan) //
    void transform_inclusive_scan(const T *in, U *out, std::int32_t n, Map map, Op op) {
  thrust::transform_inclusive_scan(thrust::device, in, in + n, out, map, op);
}

template <class T, class U, class Map, class Op> //
POLYREGION_SPECTRA_EXPORT(transform_reduce)      //
    U transform_reduce(const T *in, std::int32_t n, U init, Map map, Op op) {
  return ordered_reduce(n, init, source_map<U, const T *, Map>{in, map}, op);
}

template <class T, class U, class Map, class Op>      //
POLYREGION_SPECTRA_EXPORT(transform_reduce_unordered) //
    U transform_reduce_unordered(const T *in, std::int32_t n, U init, Map map, Op op) {
  return thrust::transform_reduce(thrust::device, in, in + n, map, init, op);
}

template <class T, class Eq>      //
POLYREGION_SPECTRA_EXPORT(unique) //
    std::int32_t unique(T *data, std::int32_t n, Eq eq) {
  return static_cast<std::int32_t>(thrust::unique(thrust::device, data, data + n, eq) - data);
}

template <class T, class Eq>           //
POLYREGION_SPECTRA_EXPORT(unique_copy) //
    std::int32_t unique_copy(const T *in, std::int32_t n, T *out, Eq eq) {
  return static_cast<std::int32_t>(thrust::unique_copy(thrust::device, in, in + n, out, eq) - out);
}

template <class T, class Op>           //
POLYREGION_SPECTRA_EXPORT(upper_bound) //
    std::int32_t upper_bound(const T *in, std::int32_t n, T value, Op op) {
  return static_cast<std::int32_t>(thrust::upper_bound(thrust::device, in, in + n, value, op) - in);
}

} // namespace spectra::implementation

#ifdef POLYREGION_SPECTRA_UNDEF_EXPORT
  #undef POLYREGION_SPECTRA_UNDEF_EXPORT
  #undef POLYREGION_SPECTRA_EXPORT
#endif
#ifdef POLYREGION_SPECTRA_UNDEF_CORE_EXPORT
  #undef POLYREGION_SPECTRA_UNDEF_CORE_EXPORT
  #undef POLYREGION_SPECTRA_CORE_EXPORT
#endif
