// Explicitly select the canonical implementation template specialisations exported by Spectra. There are no
// algorithm bodies here: EXPORT_W only takes the address of a specialisation, which forces Clang to instantiate it;
// polycpp derives the _wN export suffixes from its sized phantom type arguments.

#define EXPORT_W(name, ...) (void)&::spectra::implementation::name<__VA_ARGS__>;

#define SPECTRA_W(W)                                                                                                                       \
  static void instantiate_w##W() {                                                                                                         \
    EXPORT_W(adjacent_difference, T##W, fn<T##W, T##W, T##W>)                                                                              \
    EXPORT_W(all_of, T##W, fn<bool, T##W>)                                                                                                 \
    EXPORT_W(any_of, T##W, fn<bool, T##W>)                                                                                                 \
    EXPORT_W(binary_search, T##W, fn<bool, T##W, T##W>)                                                                                    \
    EXPORT_W(copy, T##W)                                                                                                                   \
    EXPORT_W(copy_if, T##W, fn<bool, T##W>)                                                                                                \
    EXPORT_W(copy_n, T##W)                                                                                                                 \
    EXPORT_W(count, T##W, fn<bool, T##W, T##W>)                                                                                            \
    EXPORT_W(count_if, T##W, fn<bool, T##W>)                                                                                               \
    EXPORT_W(equal, T##W, fn<bool, T##W, T##W>)                                                                                            \
    EXPORT_W(exclusive_scan, T##W, fn<T##W, T##W, T##W>)                                                                                   \
    EXPORT_W(fill, T##W)                                                                                                                   \
    EXPORT_W(fill_n, T##W)                                                                                                                 \
    EXPORT_W(find, T##W, fn<bool, T##W, T##W>)                                                                                             \
    EXPORT_W(find_if, T##W, fn<bool, T##W>)                                                                                                \
    EXPORT_W(find_if_not, T##W, fn<bool, T##W>)                                                                                            \
    EXPORT_W(for_each, T##W, fn<T##W, T##W>)                                                                                               \
    EXPORT_W(for_each_n, T##W, fn<T##W, T##W>)                                                                                             \
    EXPORT_W(gather, T##W)                                                                                                                 \
    EXPORT_W(generate, T##W, fn<T##W>)                                                                                                     \
    EXPORT_W(generate_n, T##W, fn<T##W>)                                                                                                   \
    EXPORT_W(includes, T##W, fn<bool, T##W, T##W>)                                                                                         \
    EXPORT_W(inclusive_scan, T##W, fn<T##W, T##W, T##W>)                                                                                   \
    EXPORT_W(is_partitioned, T##W, fn<bool, T##W>)                                                                                         \
    EXPORT_W(is_sorted, T##W, fn<bool, T##W, T##W>)                                                                                        \
    EXPORT_W(is_sorted_until, T##W, fn<bool, T##W, T##W>)                                                                                  \
    EXPORT_W(lower_bound, T##W, fn<bool, T##W, T##W>)                                                                                      \
    EXPORT_W(max_element, T##W, fn<bool, T##W, T##W>)                                                                                      \
    EXPORT_W(merge, T##W, fn<bool, T##W, T##W>)                                                                                            \
    EXPORT_W(min_element, T##W, fn<bool, T##W, T##W>)                                                                                      \
    EXPORT_W(minmax_element, T##W, fn<bool, T##W, T##W>)                                                                                   \
    EXPORT_W(mismatch, T##W, fn<bool, T##W, T##W>)                                                                                         \
    EXPORT_W(none_of, T##W, fn<bool, T##W>)                                                                                                \
    EXPORT_W(partition, T##W, fn<bool, T##W>)                                                                                              \
    EXPORT_W(partition_point, T##W, fn<bool, T##W>)                                                                                        \
    EXPORT_W(reduce, T##W, fn<T##W, T##W, T##W>)                                                                                           \
    EXPORT_W(remove, T##W, fn<bool, T##W, T##W>)                                                                                           \
    EXPORT_W(remove_copy, T##W, fn<bool, T##W, T##W>)                                                                                      \
    EXPORT_W(remove_copy_if, T##W, fn<bool, T##W>)                                                                                         \
    EXPORT_W(remove_if, T##W, fn<bool, T##W>)                                                                                              \
    EXPORT_W(replace, T##W, fn<bool, T##W, T##W>)                                                                                          \
    EXPORT_W(replace_copy, T##W, fn<bool, T##W, T##W>)                                                                                     \
    EXPORT_W(replace_copy_if, T##W, fn<bool, T##W>)                                                                                        \
    EXPORT_W(replace_if, T##W, fn<bool, T##W>)                                                                                             \
    EXPORT_W(reverse, T##W)                                                                                                                \
    EXPORT_W(reverse_copy, T##W)                                                                                                           \
    EXPORT_W(scatter, T##W)                                                                                                                \
    EXPORT_W(search, T##W, fn<bool, T##W, T##W>)                                                                                           \
    EXPORT_W(search_n, T##W, fn<bool, T##W, T##W>)                                                                                         \
    EXPORT_W(sequence, T##W, fn<T##W, T##W, T##W, int>)                                                                                    \
    EXPORT_W(set_difference, T##W, fn<bool, T##W, T##W>)                                                                                   \
    EXPORT_W(set_intersection, T##W, fn<bool, T##W, T##W>)                                                                                 \
    EXPORT_W(set_union, T##W, fn<bool, T##W, T##W>)                                                                                        \
    EXPORT_W(sort, T##W, fn<bool, T##W, T##W>)                                                                                             \
    EXPORT_W(stable_partition, T##W, fn<bool, T##W>)                                                                                       \
    EXPORT_W(stable_sort, T##W, fn<bool, T##W, T##W>)                                                                                      \
    EXPORT_W(swap_ranges, T##W)                                                                                                            \
    EXPORT_W(tabulate, T##W, fn<T##W, int>)                                                                                                \
    EXPORT_W(unique, T##W, fn<bool, T##W, T##W>)                                                                                           \
    EXPORT_W(unique_copy, T##W, fn<bool, T##W, T##W>)                                                                                      \
    EXPORT_W(upper_bound, T##W, fn<bool, T##W, T##W>)                                                                                      \
  }

#define SPECTRA_VALUE(W)

#define SPECTRA_TU(TW, UW)                                                                                                                 \
  static void instantiate_w##TW##_w##UW() {                                                                                                \
    EXPORT_W(transform, T##TW, U##UW, fn<U##UW, T##TW>)                                                                                    \
    EXPORT_W(transform_exclusive_scan, T##TW, U##UW, fn<U##UW, T##TW>, fn2<U##UW, U##UW, U##UW>)                                           \
    EXPORT_W(transform_inclusive_scan, T##TW, U##UW, fn<U##UW, T##TW>, fn2<U##UW, U##UW, U##UW>)                                           \
    EXPORT_W(transform_reduce, T##TW, U##UW, fn<U##UW, T##TW>, fn2<U##UW, U##UW, U##UW>)                                                   \
  }

#define SPECTRA_TUV(TW, UW, VW)                                                                                                            \
  static void instantiate_w##TW##_w##UW##_w##VW() {                                                                                        \
    EXPORT_W(transform_binary, T##TW, U##UW, V##VW, fn<V##VW, T##TW, U##UW>)                                                               \
    EXPORT_W(inner_product, T##TW, U##UW, V##VW, fn<V##VW, V##VW, V##VW>, fn2<V##VW, T##TW, U##UW>)                                        \
  }

#define SPECTRA_BYKEY(KW, VW)                                                                                                              \
  static void instantiate_key_w##KW##_value_w##VW() {                                                                                      \
    EXPORT_W(exclusive_scan_by_key, K##KW, V##VW, fn<bool, K##KW, K##KW>, fn2<V##VW, V##VW, V##VW>)                                        \
    EXPORT_W(inclusive_scan_by_key, K##KW, V##VW, fn<bool, K##KW, K##KW>, fn2<V##VW, V##VW, V##VW>)                                        \
    EXPORT_W(reduce_by_key, K##KW, V##VW, fn<bool, K##KW, K##KW>, fn2<V##VW, V##VW, V##VW>)                                                \
    EXPORT_W(sort_by_key, K##KW, V##VW, fn<bool, K##KW, K##KW>)                                                                            \
  }

#include "width_matrix.inc"

#undef SPECTRA_BYKEY
#undef SPECTRA_TUV
#undef SPECTRA_TU
#undef SPECTRA_VALUE
#undef SPECTRA_W
#undef EXPORT_W
