// AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
// Compile-only instantiations of every host reference body; runtime coverage is in test/host.
#include <cstdint>

#include <polyregion/spectra/host.hpp>

namespace {
struct SpectraSurfaceValue12 {
  std::int32_t x, y, z;
};
}

void check_surface() {
  constexpr std::int32_t n = 4;
  std::int32_t value4{};
  std::int64_t value8{};
  SpectraSurfaceValue12 value12{};

  // reduce: T=w8
  std::int64_t reduce_in[2 * n]{};
  spectra::host::reduce<std::int64_t>(
      reduce_in,
      n,
      value8,
      [](std::int64_t, std::int64_t) -> std::int64_t { return std::int64_t{}; }
  );

  // transform: T=w12, U=w4
  SpectraSurfaceValue12 transform_in[2 * n]{};
  std::int32_t transform_out[2 * n]{};
  spectra::host::transform<SpectraSurfaceValue12, std::int32_t>(
      transform_in,
      transform_out,
      n,
      [](SpectraSurfaceValue12) -> std::int32_t { return std::int32_t{}; }
  );

  // inclusive_scan: T=w8
  std::int64_t inclusive_scan_in[2 * n]{};
  std::int64_t inclusive_scan_out[2 * n]{};
  spectra::host::inclusive_scan<std::int64_t>(
      inclusive_scan_in,
      inclusive_scan_out,
      n,
      [](std::int64_t, std::int64_t) -> std::int64_t { return std::int64_t{}; }
  );

  // exclusive_scan: T=w12
  SpectraSurfaceValue12 exclusive_scan_in[2 * n]{};
  SpectraSurfaceValue12 exclusive_scan_out[2 * n]{};
  spectra::host::exclusive_scan<SpectraSurfaceValue12>(
      exclusive_scan_in,
      exclusive_scan_out,
      n,
      value12,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> SpectraSurfaceValue12 { return SpectraSurfaceValue12{}; }
  );

  // sort: T=w4
  std::int32_t sort_data[2 * n]{};
  spectra::host::sort<std::int32_t>(sort_data, n, [](std::int32_t, std::int32_t) -> bool { return bool{}; });

  // count_if: T=w12
  SpectraSurfaceValue12 count_if_in[2 * n]{};
  spectra::host::count_if<SpectraSurfaceValue12>(count_if_in, n, [](SpectraSurfaceValue12) -> bool { return bool{}; });

  // min_element: T=w8
  std::int64_t min_element_in[2 * n]{};
  spectra::host::min_element<std::int64_t>(
      min_element_in,
      n,
      [](std::int64_t, std::int64_t) -> bool { return bool{}; }
  );

  // max_element: T=w12
  SpectraSurfaceValue12 max_element_in[2 * n]{};
  spectra::host::max_element<SpectraSurfaceValue12>(
      max_element_in,
      n,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // find_if: T=w8
  std::int64_t find_if_in[2 * n]{};
  spectra::host::find_if<std::int64_t>(find_if_in, n, [](std::int64_t) -> bool { return bool{}; });

  // is_sorted: T=w8
  std::int64_t is_sorted_data[2 * n]{};
  spectra::host::is_sorted<std::int64_t>(is_sorted_data, n, [](std::int64_t, std::int64_t) -> bool { return bool{}; });

  // merge: T=w4
  std::int32_t merge_a[2 * n]{};
  std::int32_t merge_b[2 * n]{};
  std::int32_t merge_out[2 * n]{};
  spectra::host::merge<std::int32_t>(
      merge_a,
      n,
      merge_b,
      n,
      merge_out,
      [](std::int32_t, std::int32_t) -> bool { return bool{}; }
  );

  // set_difference: T=w4
  std::int32_t set_difference_a[2 * n]{};
  std::int32_t set_difference_b[2 * n]{};
  std::int32_t set_difference_out[2 * n]{};
  spectra::host::set_difference<std::int32_t>(
      set_difference_a,
      n,
      set_difference_b,
      n,
      set_difference_out,
      [](std::int32_t, std::int32_t) -> bool { return bool{}; }
  );

  // set_intersection: T=w12
  SpectraSurfaceValue12 set_intersection_a[2 * n]{};
  SpectraSurfaceValue12 set_intersection_b[2 * n]{};
  SpectraSurfaceValue12 set_intersection_out[2 * n]{};
  spectra::host::set_intersection<SpectraSurfaceValue12>(
      set_intersection_a,
      n,
      set_intersection_b,
      n,
      set_intersection_out,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // set_union: T=w12
  SpectraSurfaceValue12 set_union_a[2 * n]{};
  SpectraSurfaceValue12 set_union_b[2 * n]{};
  SpectraSurfaceValue12 set_union_out[2 * n]{};
  spectra::host::set_union<SpectraSurfaceValue12>(
      set_union_a,
      n,
      set_union_b,
      n,
      set_union_out,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // includes: T=w4
  std::int32_t includes_a[2 * n]{};
  std::int32_t includes_b[2 * n]{};
  spectra::host::includes<std::int32_t>(
      includes_a,
      n,
      includes_b,
      n,
      [](std::int32_t, std::int32_t) -> bool { return bool{}; }
  );

  // copy_if: T=w8
  std::int64_t copy_if_in[2 * n]{};
  std::int64_t copy_if_out[2 * n]{};
  spectra::host::copy_if<std::int64_t>(copy_if_in, n, copy_if_out, [](std::int64_t) -> bool { return bool{}; });

  // remove_copy_if: T=w4
  std::int32_t remove_copy_if_in[2 * n]{};
  std::int32_t remove_copy_if_out[2 * n]{};
  spectra::host::remove_copy_if<std::int32_t>(
      remove_copy_if_in,
      n,
      remove_copy_if_out,
      [](std::int32_t) -> bool { return bool{}; }
  );

  // unique_copy: T=w8
  std::int64_t unique_copy_in[2 * n]{};
  std::int64_t unique_copy_out[2 * n]{};
  spectra::host::unique_copy<std::int64_t>(
      unique_copy_in,
      n,
      unique_copy_out,
      [](std::int64_t, std::int64_t) -> bool { return bool{}; }
  );

  // remove_if: T=w8
  std::int64_t remove_if_data[2 * n]{};
  spectra::host::remove_if<std::int64_t>(remove_if_data, n, [](std::int64_t) -> bool { return bool{}; });

  // unique: T=w12
  SpectraSurfaceValue12 unique_data[2 * n]{};
  spectra::host::unique<SpectraSurfaceValue12>(
      unique_data,
      n,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // sort_by_key: K=w12, V=w4
  SpectraSurfaceValue12 sort_by_key_keys[2 * n]{};
  std::int32_t sort_by_key_values[2 * n]{};
  spectra::host::sort_by_key<SpectraSurfaceValue12, std::int32_t>(
      sort_by_key_keys,
      sort_by_key_values,
      n,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // for_each: T=w12
  SpectraSurfaceValue12 for_each_data[2 * n]{};
  spectra::host::for_each<SpectraSurfaceValue12>(
      for_each_data,
      n,
      [](SpectraSurfaceValue12) -> SpectraSurfaceValue12 { return SpectraSurfaceValue12{}; }
  );

  // gather: T=w8
  std::int32_t gather_map[2 * n]{};
  std::int64_t gather_in[2 * n]{};
  std::int64_t gather_out[2 * n]{};
  spectra::host::gather<std::int64_t>(gather_map, n, gather_in, n, gather_out);

  // scatter: T=w12
  SpectraSurfaceValue12 scatter_in[2 * n]{};
  std::int32_t scatter_map[2 * n]{};
  SpectraSurfaceValue12 scatter_out[2 * n]{};
  spectra::host::scatter<SpectraSurfaceValue12>(scatter_in, n, scatter_map, scatter_out, n);

  // copy: T=w12
  SpectraSurfaceValue12 copy_in[2 * n]{};
  SpectraSurfaceValue12 copy_out[2 * n]{};
  spectra::host::copy<SpectraSurfaceValue12>(copy_in, n, copy_out);

  // reverse: T=w12
  SpectraSurfaceValue12 reverse_data[2 * n]{};
  spectra::host::reverse<SpectraSurfaceValue12>(reverse_data, n);

  // fill: T=w4
  std::int32_t fill_out[2 * n]{};
  spectra::host::fill<std::int32_t>(fill_out, n, value4);

  // transform_reduce: T=w4, U=w8
  std::int32_t transform_reduce_in[2 * n]{};
  spectra::host::transform_reduce<std::int32_t, std::int64_t>(
      transform_reduce_in,
      n,
      value8,
      [](std::int32_t) -> std::int64_t { return std::int64_t{}; },
      [](std::int64_t, std::int64_t) -> std::int64_t { return std::int64_t{}; }
  );

  // transform_inclusive_scan: T=w4, U=w8
  std::int32_t transform_inclusive_scan_in[2 * n]{};
  std::int64_t transform_inclusive_scan_out[2 * n]{};
  spectra::host::transform_inclusive_scan<std::int32_t, std::int64_t>(
      transform_inclusive_scan_in,
      transform_inclusive_scan_out,
      n,
      [](std::int32_t) -> std::int64_t { return std::int64_t{}; },
      [](std::int64_t, std::int64_t) -> std::int64_t { return std::int64_t{}; }
  );

  // transform_exclusive_scan: T=w8, U=w12
  std::int64_t transform_exclusive_scan_in[2 * n]{};
  SpectraSurfaceValue12 transform_exclusive_scan_out[2 * n]{};
  spectra::host::transform_exclusive_scan<std::int64_t, SpectraSurfaceValue12>(
      transform_exclusive_scan_in,
      transform_exclusive_scan_out,
      n,
      value12,
      [](std::int64_t) -> SpectraSurfaceValue12 { return SpectraSurfaceValue12{}; },
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> SpectraSurfaceValue12 { return SpectraSurfaceValue12{}; }
  );

  // reduce_by_key: K=w4, V=w8
  std::int32_t reduce_by_key_keys[2 * n]{};
  std::int64_t reduce_by_key_vals[2 * n]{};
  std::int32_t reduce_by_key_kout[2 * n]{};
  std::int64_t reduce_by_key_vout[2 * n]{};
  spectra::host::reduce_by_key<std::int32_t, std::int64_t>(
      reduce_by_key_keys,
      reduce_by_key_vals,
      reduce_by_key_kout,
      reduce_by_key_vout,
      n,
      [](std::int32_t, std::int32_t) -> bool { return bool{}; },
      [](std::int64_t, std::int64_t) -> std::int64_t { return std::int64_t{}; }
  );

  // inclusive_scan_by_key: K=w12, V=w4
  SpectraSurfaceValue12 inclusive_scan_by_key_keys[2 * n]{};
  std::int32_t inclusive_scan_by_key_vals[2 * n]{};
  std::int32_t inclusive_scan_by_key_out[2 * n]{};
  spectra::host::inclusive_scan_by_key<SpectraSurfaceValue12, std::int32_t>(
      inclusive_scan_by_key_keys,
      inclusive_scan_by_key_vals,
      inclusive_scan_by_key_out,
      n,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; },
      [](std::int32_t, std::int32_t) -> std::int32_t { return std::int32_t{}; }
  );

  // exclusive_scan_by_key: K=w4, V=w8
  std::int32_t exclusive_scan_by_key_keys[2 * n]{};
  std::int64_t exclusive_scan_by_key_vals[2 * n]{};
  std::int64_t exclusive_scan_by_key_out[2 * n]{};
  spectra::host::exclusive_scan_by_key<std::int32_t, std::int64_t>(
      exclusive_scan_by_key_keys,
      exclusive_scan_by_key_vals,
      exclusive_scan_by_key_out,
      n,
      value8,
      [](std::int32_t, std::int32_t) -> bool { return bool{}; },
      [](std::int64_t, std::int64_t) -> std::int64_t { return std::int64_t{}; }
  );

  // transform_binary: T=w8, U=w12, V=w4
  std::int64_t transform_binary_a[2 * n]{};
  SpectraSurfaceValue12 transform_binary_b[2 * n]{};
  std::int32_t transform_binary_out[2 * n]{};
  spectra::host::transform_binary<std::int64_t, SpectraSurfaceValue12, std::int32_t>(
      transform_binary_a,
      transform_binary_b,
      transform_binary_out,
      n,
      [](std::int64_t, SpectraSurfaceValue12) -> std::int32_t { return std::int32_t{}; }
  );

  // inner_product: T=w8, U=w12, V=w4
  std::int64_t inner_product_a[2 * n]{};
  SpectraSurfaceValue12 inner_product_b[2 * n]{};
  spectra::host::inner_product<std::int64_t, SpectraSurfaceValue12, std::int32_t>(
      inner_product_a,
      inner_product_b,
      n,
      value4,
      [](std::int32_t, std::int32_t) -> std::int32_t { return std::int32_t{}; },
      [](std::int64_t, SpectraSurfaceValue12) -> std::int32_t { return std::int32_t{}; }
  );

  // equal: T=w12
  SpectraSurfaceValue12 equal_a[2 * n]{};
  SpectraSurfaceValue12 equal_b[2 * n]{};
  spectra::host::equal<SpectraSurfaceValue12>(
      equal_a,
      equal_b,
      n,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // mismatch: T=w8
  std::int64_t mismatch_a[2 * n]{};
  std::int64_t mismatch_b[2 * n]{};
  spectra::host::mismatch<std::int64_t>(
      mismatch_a,
      mismatch_b,
      n,
      [](std::int64_t, std::int64_t) -> bool { return bool{}; }
  );

  // search: T=w12
  SpectraSurfaceValue12 search_in[2 * n]{};
  SpectraSurfaceValue12 search_sub[2 * n]{};
  spectra::host::search<SpectraSurfaceValue12>(
      search_in,
      n,
      search_sub,
      n,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // search_n: T=w12
  SpectraSurfaceValue12 search_n_in[2 * n]{};
  spectra::host::search_n<SpectraSurfaceValue12>(
      search_n_in,
      n,
      n,
      value12,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // all_of: T=w12
  SpectraSurfaceValue12 all_of_in[2 * n]{};
  spectra::host::all_of<SpectraSurfaceValue12>(all_of_in, n, [](SpectraSurfaceValue12) -> bool { return bool{}; });

  // any_of: T=w12
  SpectraSurfaceValue12 any_of_in[2 * n]{};
  spectra::host::any_of<SpectraSurfaceValue12>(any_of_in, n, [](SpectraSurfaceValue12) -> bool { return bool{}; });

  // none_of: T=w4
  std::int32_t none_of_in[2 * n]{};
  spectra::host::none_of<std::int32_t>(none_of_in, n, [](std::int32_t) -> bool { return bool{}; });

  // find_if_not: T=w4
  std::int32_t find_if_not_in[2 * n]{};
  spectra::host::find_if_not<std::int32_t>(find_if_not_in, n, [](std::int32_t) -> bool { return bool{}; });

  // generate: T=w4
  std::int32_t generate_out[2 * n]{};
  spectra::host::generate<std::int32_t>(generate_out, n, []() -> std::int32_t { return std::int32_t{}; });

  // generate_n: T=w8
  std::int64_t generate_n_out[2 * n]{};
  spectra::host::generate_n<std::int64_t>(generate_n_out, n, []() -> std::int64_t { return std::int64_t{}; });

  // tabulate: T=w12
  SpectraSurfaceValue12 tabulate_out[2 * n]{};
  spectra::host::tabulate<SpectraSurfaceValue12>(
      tabulate_out,
      n,
      [](std::int32_t) -> SpectraSurfaceValue12 { return SpectraSurfaceValue12{}; }
  );

  // lower_bound: T=w12
  SpectraSurfaceValue12 lower_bound_in[2 * n]{};
  spectra::host::lower_bound<SpectraSurfaceValue12>(
      lower_bound_in,
      n,
      value12,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // upper_bound: T=w8
  std::int64_t upper_bound_in[2 * n]{};
  spectra::host::upper_bound<std::int64_t>(
      upper_bound_in,
      n,
      value8,
      [](std::int64_t, std::int64_t) -> bool { return bool{}; }
  );

  // binary_search: T=w12
  SpectraSurfaceValue12 binary_search_in[2 * n]{};
  spectra::host::binary_search<SpectraSurfaceValue12>(
      binary_search_in,
      n,
      value12,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // partition: T=w12
  SpectraSurfaceValue12 partition_data[2 * n]{};
  spectra::host::partition<SpectraSurfaceValue12>(
      partition_data,
      n,
      [](SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // stable_partition: T=w8
  std::int64_t stable_partition_data[2 * n]{};
  spectra::host::stable_partition<std::int64_t>(stable_partition_data, n, [](std::int64_t) -> bool { return bool{}; });

  // partition_point: T=w4
  std::int32_t partition_point_in[2 * n]{};
  spectra::host::partition_point<std::int32_t>(partition_point_in, n, [](std::int32_t) -> bool { return bool{}; });

  // is_partitioned: T=w8
  std::int64_t is_partitioned_data[2 * n]{};
  spectra::host::is_partitioned<std::int64_t>(is_partitioned_data, n, [](std::int64_t) -> bool { return bool{}; });

  // is_sorted_until: T=w12
  SpectraSurfaceValue12 is_sorted_until_data[2 * n]{};
  spectra::host::is_sorted_until<SpectraSurfaceValue12>(
      is_sorted_until_data,
      n,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // stable_sort: T=w12
  SpectraSurfaceValue12 stable_sort_data[2 * n]{};
  spectra::host::stable_sort<SpectraSurfaceValue12>(
      stable_sort_data,
      n,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // for_each_n: T=w12
  SpectraSurfaceValue12 for_each_n_data[2 * n]{};
  spectra::host::for_each_n<SpectraSurfaceValue12>(
      for_each_n_data,
      n,
      [](SpectraSurfaceValue12) -> SpectraSurfaceValue12 { return SpectraSurfaceValue12{}; }
  );

  // minmax_element: T=w12
  SpectraSurfaceValue12 minmax_element_in[2 * n]{};
  SpectraSurfaceValue12 minmax_element_min_out[2 * n]{};
  SpectraSurfaceValue12 minmax_element_max_out[2 * n]{};
  spectra::host::minmax_element<SpectraSurfaceValue12>(
      minmax_element_in,
      n,
      minmax_element_min_out,
      minmax_element_max_out,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // copy_n: T=w12
  SpectraSurfaceValue12 copy_n_in[2 * n]{};
  SpectraSurfaceValue12 copy_n_out[2 * n]{};
  spectra::host::copy_n<SpectraSurfaceValue12>(copy_n_in, n, copy_n_out);

  // reverse_copy: T=w8
  std::int64_t reverse_copy_in[2 * n]{};
  std::int64_t reverse_copy_out[2 * n]{};
  spectra::host::reverse_copy<std::int64_t>(reverse_copy_in, n, reverse_copy_out);

  // fill_n: T=w4
  std::int32_t fill_n_out[2 * n]{};
  spectra::host::fill_n<std::int32_t>(fill_n_out, n, value4);

  // swap_ranges: T=w8
  std::int64_t swap_ranges_a[2 * n]{};
  std::int64_t swap_ranges_b[2 * n]{};
  spectra::host::swap_ranges<std::int64_t>(swap_ranges_a, n, swap_ranges_b);

  // count: T=w8
  std::int64_t count_in[2 * n]{};
  spectra::host::count<std::int64_t>(count_in, n, value8, [](std::int64_t, std::int64_t) -> bool { return bool{}; });

  // find: T=w4
  std::int32_t find_in[2 * n]{};
  spectra::host::find<std::int32_t>(find_in, n, value4, [](std::int32_t, std::int32_t) -> bool { return bool{}; });

  // remove: T=w12
  SpectraSurfaceValue12 remove_data[2 * n]{};
  spectra::host::remove<SpectraSurfaceValue12>(
      remove_data,
      n,
      value12,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // remove_copy: T=w8
  std::int64_t remove_copy_in[2 * n]{};
  std::int64_t remove_copy_out[2 * n]{};
  spectra::host::remove_copy<std::int64_t>(
      remove_copy_in,
      n,
      remove_copy_out,
      value8,
      [](std::int64_t, std::int64_t) -> bool { return bool{}; }
  );

  // replace: T=w4
  std::int32_t replace_io[2 * n]{};
  spectra::host::replace<std::int32_t>(
      replace_io,
      n,
      value4,
      value4,
      [](std::int32_t, std::int32_t) -> bool { return bool{}; }
  );

  // replace_copy: T=w12
  SpectraSurfaceValue12 replace_copy_in[2 * n]{};
  SpectraSurfaceValue12 replace_copy_out[2 * n]{};
  spectra::host::replace_copy<SpectraSurfaceValue12>(
      replace_copy_in,
      replace_copy_out,
      n,
      value12,
      value12,
      [](SpectraSurfaceValue12, SpectraSurfaceValue12) -> bool { return bool{}; }
  );

  // replace_if: T=w4
  std::int32_t replace_if_data[2 * n]{};
  spectra::host::replace_if<std::int32_t>(replace_if_data, n, value4, [](std::int32_t) -> bool { return bool{}; });

  // replace_copy_if: T=w8
  std::int64_t replace_copy_if_in[2 * n]{};
  std::int64_t replace_copy_if_out[2 * n]{};
  spectra::host::replace_copy_if<std::int64_t>(
      replace_copy_if_in,
      replace_copy_if_out,
      n,
      value8,
      [](std::int64_t) -> bool { return bool{}; }
  );

  // sequence: T=w4
  std::int32_t sequence_out[2 * n]{};
  spectra::host::sequence<std::int32_t>(
      sequence_out,
      n,
      value4,
      value4,
      [](std::int32_t, std::int32_t, std::int32_t) -> std::int32_t { return std::int32_t{}; }
  );

  // adjacent_difference: T=w4
  std::int32_t adjacent_difference_in[2 * n]{};
  std::int32_t adjacent_difference_out[2 * n]{};
  spectra::host::adjacent_difference<std::int32_t>(
      adjacent_difference_in,
      adjacent_difference_out,
      n,
      [](std::int32_t, std::int32_t) -> std::int32_t { return std::int32_t{}; }
  );
}
