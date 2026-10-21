! AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT.
! Compile-only instantiations of every generated Fortran declaration; runtime package coverage is separate.
module spectra_surface_values
  use iso_c_binding
  implicit none
  type, bind(c) :: spectra_surface_value12
    integer(c_int32_t) :: x, y, z
  end type spectra_surface_value12
contains
  subroutine spectra_surface_reduce_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_reduce_op

  subroutine spectra_surface_transform_op(arg0, r)
    type(spectra_surface_value12), intent(in) :: arg0
    integer(c_int32_t), intent(out) :: r
    r = 0_c_int32_t
  end subroutine spectra_surface_transform_op

  subroutine spectra_surface_inclusive_scan_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_inclusive_scan_op

  subroutine spectra_surface_exclusive_scan_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    type(spectra_surface_value12), intent(out) :: r
    r = spectra_surface_value12(0, 0, 0)
  end subroutine spectra_surface_exclusive_scan_op

  subroutine spectra_surface_sort_op(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_sort_op

  subroutine spectra_surface_count_if_op(arg0, r)
    type(spectra_surface_value12), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_count_if_op

  subroutine spectra_surface_min_element_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_min_element_op

  subroutine spectra_surface_max_element_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_max_element_op

  subroutine spectra_surface_find_if_op(arg0, r)
    integer(c_int64_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_find_if_op

  subroutine spectra_surface_is_sorted_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_is_sorted_op

  subroutine spectra_surface_merge_op(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_merge_op

  subroutine spectra_surface_set_difference_op(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_set_difference_op

  subroutine spectra_surface_set_intersection_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_set_intersection_op

  subroutine spectra_surface_set_union_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_set_union_op

  subroutine spectra_surface_includes_op(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_includes_op

  subroutine spectra_surface_copy_if_op(arg0, r)
    integer(c_int64_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_copy_if_op

  subroutine spectra_surface_remove_copy_if_op(arg0, r)
    integer(c_int32_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_remove_copy_if_op

  subroutine spectra_surface_unique_copy_eq(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_unique_copy_eq

  subroutine spectra_surface_remove_if_op(arg0, r)
    integer(c_int64_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_remove_if_op

  subroutine spectra_surface_unique_eq(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_unique_eq

  subroutine spectra_surface_sort_by_key_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_sort_by_key_op

  subroutine spectra_surface_for_each_op(arg0, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(out) :: r
    r = spectra_surface_value12(0, 0, 0)
  end subroutine spectra_surface_for_each_op

  subroutine spectra_surface_transform_reduce_map(arg0, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_transform_reduce_map

  subroutine spectra_surface_transform_reduce_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_transform_reduce_op

  subroutine spectra_surface_transform_inclusive_scan_map(arg0, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_transform_inclusive_scan_map

  subroutine spectra_surface_transform_inclusive_scan_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_transform_inclusive_scan_op

  subroutine spectra_surface_transform_exclusive_scan_map(arg0, r)
    integer(c_int64_t), intent(in) :: arg0
    type(spectra_surface_value12), intent(out) :: r
    r = spectra_surface_value12(0, 0, 0)
  end subroutine spectra_surface_transform_exclusive_scan_map

  subroutine spectra_surface_transform_exclusive_scan_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    type(spectra_surface_value12), intent(out) :: r
    r = spectra_surface_value12(0, 0, 0)
  end subroutine spectra_surface_transform_exclusive_scan_op

  subroutine spectra_surface_reduce_by_key_eq(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_reduce_by_key_eq

  subroutine spectra_surface_reduce_by_key_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_reduce_by_key_op

  subroutine spectra_surface_inclusive_scan_by_key_eq(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_inclusive_scan_by_key_eq

  subroutine spectra_surface_inclusive_scan_by_key_op(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    integer(c_int32_t), intent(out) :: r
    r = 0_c_int32_t
  end subroutine spectra_surface_inclusive_scan_by_key_op

  subroutine spectra_surface_exclusive_scan_by_key_eq(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_exclusive_scan_by_key_eq

  subroutine spectra_surface_exclusive_scan_by_key_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_exclusive_scan_by_key_op

  subroutine spectra_surface_transform_binary_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    integer(c_int32_t), intent(out) :: r
    r = 0_c_int32_t
  end subroutine spectra_surface_transform_binary_op

  subroutine spectra_surface_inner_product_op_reduce(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    integer(c_int32_t), intent(out) :: r
    r = 0_c_int32_t
  end subroutine spectra_surface_inner_product_op_reduce

  subroutine spectra_surface_inner_product_op_product(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    integer(c_int32_t), intent(out) :: r
    r = 0_c_int32_t
  end subroutine spectra_surface_inner_product_op_product

  subroutine spectra_surface_equal_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_equal_op

  subroutine spectra_surface_mismatch_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_mismatch_op

  subroutine spectra_surface_search_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_search_op

  subroutine spectra_surface_search_n_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_search_n_op

  subroutine spectra_surface_all_of_op(arg0, r)
    type(spectra_surface_value12), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_all_of_op

  subroutine spectra_surface_any_of_op(arg0, r)
    type(spectra_surface_value12), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_any_of_op

  subroutine spectra_surface_none_of_op(arg0, r)
    integer(c_int32_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_none_of_op

  subroutine spectra_surface_find_if_not_op(arg0, r)
    integer(c_int32_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_find_if_not_op

  subroutine spectra_surface_generate_op(r)

    integer(c_int32_t), intent(out) :: r
    r = 0_c_int32_t
  end subroutine spectra_surface_generate_op

  subroutine spectra_surface_generate_n_op(r)

    integer(c_int64_t), intent(out) :: r
    r = 0_c_int64_t
  end subroutine spectra_surface_generate_n_op

  subroutine spectra_surface_tabulate_op(arg0, r)
    integer(c_int32_t), intent(in) :: arg0
    type(spectra_surface_value12), intent(out) :: r
    r = spectra_surface_value12(0, 0, 0)
  end subroutine spectra_surface_tabulate_op

  subroutine spectra_surface_lower_bound_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_lower_bound_op

  subroutine spectra_surface_upper_bound_op(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_upper_bound_op

  subroutine spectra_surface_binary_search_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_binary_search_op

  subroutine spectra_surface_partition_op(arg0, r)
    type(spectra_surface_value12), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_partition_op

  subroutine spectra_surface_stable_partition_op(arg0, r)
    integer(c_int64_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_stable_partition_op

  subroutine spectra_surface_partition_point_op(arg0, r)
    integer(c_int32_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_partition_point_op

  subroutine spectra_surface_is_partitioned_op(arg0, r)
    integer(c_int64_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_is_partitioned_op

  subroutine spectra_surface_is_sorted_until_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_is_sorted_until_op

  subroutine spectra_surface_stable_sort_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_stable_sort_op

  subroutine spectra_surface_for_each_n_op(arg0, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(out) :: r
    r = spectra_surface_value12(0, 0, 0)
  end subroutine spectra_surface_for_each_n_op

  subroutine spectra_surface_minmax_element_op(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_minmax_element_op

  subroutine spectra_surface_count_eq(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_count_eq

  subroutine spectra_surface_find_eq(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_find_eq

  subroutine spectra_surface_remove_eq(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_remove_eq

  subroutine spectra_surface_remove_copy_eq(arg0, arg1, r)
    integer(c_int64_t), intent(in) :: arg0
    integer(c_int64_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_remove_copy_eq

  subroutine spectra_surface_replace_eq(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_replace_eq

  subroutine spectra_surface_replace_copy_eq(arg0, arg1, r)
    type(spectra_surface_value12), intent(in) :: arg0
    type(spectra_surface_value12), intent(in) :: arg1
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_replace_copy_eq

  subroutine spectra_surface_replace_if_op(arg0, r)
    integer(c_int32_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_replace_if_op

  subroutine spectra_surface_replace_copy_if_op(arg0, r)
    integer(c_int64_t), intent(in) :: arg0
    logical(c_bool), intent(out) :: r
    r = .false.
  end subroutine spectra_surface_replace_copy_if_op

  subroutine spectra_surface_sequence_value_at(arg0, arg1, arg2, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    integer(c_int32_t), intent(in) :: arg2
    integer(c_int32_t), intent(out) :: r
    r = 0_c_int32_t
  end subroutine spectra_surface_sequence_value_at

  subroutine spectra_surface_adjacent_difference_op(arg0, arg1, r)
    integer(c_int32_t), intent(in) :: arg0
    integer(c_int32_t), intent(in) :: arg1
    integer(c_int32_t), intent(out) :: r
    r = 0_c_int32_t
  end subroutine spectra_surface_adjacent_difference_op
end module spectra_surface_values

program check
  use iso_c_binding
  use spectra_api
  use spectra_surface_values
  implicit none
  integer(c_int32_t), parameter :: n = 4
  integer(c_int32_t) :: values4(n), value4, integer_result
  integer(c_int64_t) :: values8(n), value8
  type(spectra_surface_value12) :: values12(n), value12
  logical(c_bool) :: logical_result

  ! reduce: T=w8
  call polyregion_reduce( &
    values8, &
    n, &
    value8, &
    spectra_surface_reduce_op, &
    value8)

  ! transform: T=w12, U=w4
  call polyregion_transform( &
    values12, &
    values4, &
    n, &
    spectra_surface_transform_op)

  ! inclusive_scan: T=w8
  call polyregion_inclusive_scan( &
    values8, &
    values8, &
    n, &
    spectra_surface_inclusive_scan_op)

  ! exclusive_scan: T=w12
  call polyregion_exclusive_scan( &
    values12, &
    values12, &
    n, &
    value12, &
    spectra_surface_exclusive_scan_op)

  ! sort: T=w4
  call polyregion_sort( &
    values4, &
    n, &
    spectra_surface_sort_op)

  ! count_if: T=w12
  integer_result = polyregion_count_if( &
    values12, &
    n, &
    spectra_surface_count_if_op)

  ! min_element: T=w8
  call polyregion_min_element( &
    values8, &
    n, &
    spectra_surface_min_element_op, &
    value8)

  ! max_element: T=w12
  call polyregion_max_element( &
    values12, &
    n, &
    spectra_surface_max_element_op, &
    value12)

  ! find_if: T=w8
  integer_result = polyregion_find_if( &
    values8, &
    n, &
    spectra_surface_find_if_op)

  ! is_sorted: T=w8
  logical_result = polyregion_is_sorted( &
    values8, &
    n, &
    spectra_surface_is_sorted_op)

  ! merge: T=w4
  call polyregion_merge( &
    values4, &
    n, &
    values4, &
    n, &
    values4, &
    spectra_surface_merge_op)

  ! set_difference: T=w4
  integer_result = polyregion_set_difference( &
    values4, &
    n, &
    values4, &
    n, &
    values4, &
    spectra_surface_set_difference_op)

  ! set_intersection: T=w12
  integer_result = polyregion_set_intersection( &
    values12, &
    n, &
    values12, &
    n, &
    values12, &
    spectra_surface_set_intersection_op)

  ! set_union: T=w12
  integer_result = polyregion_set_union( &
    values12, &
    n, &
    values12, &
    n, &
    values12, &
    spectra_surface_set_union_op)

  ! includes: T=w4
  logical_result = polyregion_includes( &
    values4, &
    n, &
    values4, &
    n, &
    spectra_surface_includes_op)

  ! copy_if: T=w8
  integer_result = polyregion_copy_if( &
    values8, &
    n, &
    values8, &
    spectra_surface_copy_if_op)

  ! remove_copy_if: T=w4
  integer_result = polyregion_remove_copy_if( &
    values4, &
    n, &
    values4, &
    spectra_surface_remove_copy_if_op)

  ! unique_copy: T=w8
  integer_result = polyregion_unique_copy( &
    values8, &
    n, &
    values8, &
    spectra_surface_unique_copy_eq)

  ! remove_if: T=w8
  integer_result = polyregion_remove_if( &
    values8, &
    n, &
    spectra_surface_remove_if_op)

  ! unique: T=w12
  integer_result = polyregion_unique( &
    values12, &
    n, &
    spectra_surface_unique_eq)

  ! sort_by_key: K=w12, V=w4
  call polyregion_sort_by_key( &
    values12, &
    values4, &
    n, &
    spectra_surface_sort_by_key_op)

  ! for_each: T=w12
  call polyregion_for_each( &
    values12, &
    n, &
    spectra_surface_for_each_op)

  ! gather: T=w8
  call polyregion_gather( &
    values4, &
    n, &
    values8, &
    n, &
    values8)

  ! scatter: T=w12
  call polyregion_scatter( &
    values12, &
    n, &
    values4, &
    values12, &
    n)

  ! copy: T=w12
  call polyregion_copy( &
    values12, &
    n, &
    values12)

  ! reverse: T=w12
  call polyregion_reverse( &
    values12, &
    n)

  ! fill: T=w4
  call polyregion_fill( &
    values4, &
    n, &
    value4)

  ! transform_reduce: T=w4, U=w8
  call polyregion_transform_reduce( &
    values4, &
    n, &
    value8, &
    spectra_surface_transform_reduce_map, &
    spectra_surface_transform_reduce_op, &
    value8)

  ! transform_inclusive_scan: T=w4, U=w8
  call polyregion_transform_inclusive_scan( &
    values4, &
    values8, &
    n, &
    spectra_surface_transform_inclusive_scan_map, &
    spectra_surface_transform_inclusive_scan_op)

  ! transform_exclusive_scan: T=w8, U=w12
  call polyregion_transform_exclusive_scan( &
    values8, &
    values12, &
    n, &
    value12, &
    spectra_surface_transform_exclusive_scan_map, &
    spectra_surface_transform_exclusive_scan_op)

  ! reduce_by_key: K=w4, V=w8
  integer_result = polyregion_reduce_by_key( &
    values4, &
    values8, &
    values4, &
    values8, &
    n, &
    spectra_surface_reduce_by_key_eq, &
    spectra_surface_reduce_by_key_op)

  ! inclusive_scan_by_key: K=w12, V=w4
  call polyregion_inclusive_scan_by_key( &
    values12, &
    values4, &
    values4, &
    n, &
    spectra_surface_inclusive_scan_by_key_eq, &
    spectra_surface_inclusive_scan_by_key_op)

  ! exclusive_scan_by_key: K=w4, V=w8
  call polyregion_exclusive_scan_by_key( &
    values4, &
    values8, &
    values8, &
    n, &
    value8, &
    spectra_surface_exclusive_scan_by_key_eq, &
    spectra_surface_exclusive_scan_by_key_op)

  ! transform_binary: T=w8, U=w12, V=w4
  call polyregion_transform_binary( &
    values8, &
    values12, &
    values4, &
    n, &
    spectra_surface_transform_binary_op)

  ! inner_product: T=w8, U=w12, V=w4
  call polyregion_inner_product( &
    values8, &
    values12, &
    n, &
    value4, &
    spectra_surface_inner_product_op_reduce, &
    spectra_surface_inner_product_op_product, &
    value4)

  ! equal: T=w12
  logical_result = polyregion_equal( &
    values12, &
    values12, &
    n, &
    spectra_surface_equal_op)

  ! mismatch: T=w8
  integer_result = polyregion_mismatch( &
    values8, &
    values8, &
    n, &
    spectra_surface_mismatch_op)

  ! search: T=w12
  integer_result = polyregion_search( &
    values12, &
    n, &
    values12, &
    n, &
    spectra_surface_search_op)

  ! search_n: T=w12
  integer_result = polyregion_search_n( &
    values12, &
    n, &
    n, &
    value12, &
    spectra_surface_search_n_op)

  ! all_of: T=w12
  logical_result = polyregion_all_of( &
    values12, &
    n, &
    spectra_surface_all_of_op)

  ! any_of: T=w12
  logical_result = polyregion_any_of( &
    values12, &
    n, &
    spectra_surface_any_of_op)

  ! none_of: T=w4
  logical_result = polyregion_none_of( &
    values4, &
    n, &
    spectra_surface_none_of_op)

  ! find_if_not: T=w4
  integer_result = polyregion_find_if_not( &
    values4, &
    n, &
    spectra_surface_find_if_not_op)

  ! generate: T=w4
  call polyregion_generate( &
    values4, &
    n, &
    spectra_surface_generate_op)

  ! generate_n: T=w8
  call polyregion_generate_n( &
    values8, &
    n, &
    spectra_surface_generate_n_op)

  ! tabulate: T=w12
  call polyregion_tabulate( &
    values12, &
    n, &
    spectra_surface_tabulate_op)

  ! lower_bound: T=w12
  integer_result = polyregion_lower_bound( &
    values12, &
    n, &
    value12, &
    spectra_surface_lower_bound_op)

  ! upper_bound: T=w8
  integer_result = polyregion_upper_bound( &
    values8, &
    n, &
    value8, &
    spectra_surface_upper_bound_op)

  ! binary_search: T=w12
  logical_result = polyregion_binary_search( &
    values12, &
    n, &
    value12, &
    spectra_surface_binary_search_op)

  ! partition: T=w12
  call polyregion_partition( &
    values12, &
    n, &
    spectra_surface_partition_op)

  ! stable_partition: T=w8
  call polyregion_stable_partition( &
    values8, &
    n, &
    spectra_surface_stable_partition_op)

  ! partition_point: T=w4
  integer_result = polyregion_partition_point( &
    values4, &
    n, &
    spectra_surface_partition_point_op)

  ! is_partitioned: T=w8
  logical_result = polyregion_is_partitioned( &
    values8, &
    n, &
    spectra_surface_is_partitioned_op)

  ! is_sorted_until: T=w12
  integer_result = polyregion_is_sorted_until( &
    values12, &
    n, &
    spectra_surface_is_sorted_until_op)

  ! stable_sort: T=w12
  call polyregion_stable_sort( &
    values12, &
    n, &
    spectra_surface_stable_sort_op)

  ! for_each_n: T=w12
  call polyregion_for_each_n( &
    values12, &
    n, &
    spectra_surface_for_each_n_op)

  ! minmax_element: T=w12
  call polyregion_minmax_element( &
    values12, &
    n, &
    values12, &
    values12, &
    spectra_surface_minmax_element_op)

  ! copy_n: T=w12
  call polyregion_copy_n( &
    values12, &
    n, &
    values12)

  ! reverse_copy: T=w8
  call polyregion_reverse_copy( &
    values8, &
    n, &
    values8)

  ! fill_n: T=w4
  call polyregion_fill_n( &
    values4, &
    n, &
    value4)

  ! swap_ranges: T=w8
  call polyregion_swap_ranges( &
    values8, &
    n, &
    values8)

  ! count: T=w8
  integer_result = polyregion_count( &
    values8, &
    n, &
    value8, &
    spectra_surface_count_eq)

  ! find: T=w4
  integer_result = polyregion_find( &
    values4, &
    n, &
    value4, &
    spectra_surface_find_eq)

  ! remove: T=w12
  integer_result = polyregion_remove( &
    values12, &
    n, &
    value12, &
    spectra_surface_remove_eq)

  ! remove_copy: T=w8
  integer_result = polyregion_remove_copy( &
    values8, &
    n, &
    values8, &
    value8, &
    spectra_surface_remove_copy_eq)

  ! replace: T=w4
  call polyregion_replace( &
    values4, &
    n, &
    value4, &
    value4, &
    spectra_surface_replace_eq)

  ! replace_copy: T=w12
  call polyregion_replace_copy( &
    values12, &
    values12, &
    n, &
    value12, &
    value12, &
    spectra_surface_replace_copy_eq)

  ! replace_if: T=w4
  call polyregion_replace_if( &
    values4, &
    n, &
    value4, &
    spectra_surface_replace_if_op)

  ! replace_copy_if: T=w8
  call polyregion_replace_copy_if( &
    values8, &
    values8, &
    n, &
    value8, &
    spectra_surface_replace_copy_if_op)

  ! sequence: T=w4
  call polyregion_sequence( &
    values4, &
    n, &
    value4, &
    value4, &
    spectra_surface_sequence_value_at)

  ! adjacent_difference: T=w4
  call polyregion_adjacent_difference( &
    values4, &
    values4, &
    n, &
    spectra_surface_adjacent_difference_op)
end program check
