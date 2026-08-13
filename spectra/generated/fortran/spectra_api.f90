module spectra_api
  use iso_c_binding, only: c_bool, c_double, c_float, c_int8_t, c_int16_t, c_int32_t, c_int64_t
  implicit none
  private

  public :: polyregion_adjacent_difference
  public :: polyregion_all_of
  public :: polyregion_any_of
  public :: polyregion_binary_search
  public :: polyregion_copy
  public :: polyregion_copy_if
  public :: polyregion_copy_n
  public :: polyregion_count
  public :: polyregion_count_if
  public :: polyregion_equal
  public :: polyregion_exclusive_scan
  public :: polyregion_exclusive_scan_by_key
  public :: polyregion_fill
  public :: polyregion_fill_n
  public :: polyregion_find
  public :: polyregion_find_if
  public :: polyregion_find_if_not
  public :: polyregion_for_each
  public :: polyregion_for_each_n
  public :: polyregion_gather
  public :: polyregion_generate
  public :: polyregion_generate_n
  public :: polyregion_includes
  public :: polyregion_inclusive_scan
  public :: polyregion_inclusive_scan_by_key
  public :: polyregion_inner_product
  public :: polyregion_is_partitioned
  public :: polyregion_is_sorted
  public :: polyregion_is_sorted_until
  public :: polyregion_lower_bound
  public :: polyregion_max_element
  public :: polyregion_merge
  public :: polyregion_min_element
  public :: polyregion_minmax_element
  public :: polyregion_mismatch
  public :: polyregion_none_of
  public :: polyregion_partition
  public :: polyregion_partition_point
  public :: polyregion_reduce
  public :: polyregion_reduce_by_key
  public :: polyregion_remove
  public :: polyregion_remove_copy
  public :: polyregion_remove_copy_if
  public :: polyregion_remove_if
  public :: polyregion_replace
  public :: polyregion_replace_copy
  public :: polyregion_replace_copy_if
  public :: polyregion_replace_if
  public :: polyregion_reverse
  public :: polyregion_reverse_copy
  public :: polyregion_scatter
  public :: polyregion_search
  public :: polyregion_search_n
  public :: polyregion_sequence
  public :: polyregion_set_difference
  public :: polyregion_set_intersection
  public :: polyregion_set_union
  public :: polyregion_sort
  public :: polyregion_sort_by_key
  public :: polyregion_stable_partition
  public :: polyregion_stable_sort
  public :: polyregion_swap_ranges
  public :: polyregion_tabulate
  public :: polyregion_transform
  public :: polyregion_transform_binary
  public :: polyregion_transform_exclusive_scan
  public :: polyregion_transform_inclusive_scan
  public :: polyregion_transform_reduce
  public :: polyregion_unique
  public :: polyregion_unique_copy
  public :: polyregion_upper_bound

  interface
    subroutine polyregion_import(identity)
      character(len=*), intent(in) :: identity
    end subroutine polyregion_import
  end interface

contains

  subroutine polyregion_adjacent_difference(in, out, n, op)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.adjacent_difference")
    error stop 'compiler did not replace'
  end subroutine polyregion_adjacent_difference

  function polyregion_all_of(in, n, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    logical(c_bool) :: r
    r = .false.
    call polyregion_import("polyregion_import:spectra:spectra.all_of")
    error stop 'compiler did not replace'
  end function polyregion_all_of

  function polyregion_any_of(in, n, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    logical(c_bool) :: r
    r = .false.
    call polyregion_import("polyregion_import:spectra:spectra.any_of")
    error stop 'compiler did not replace'
  end function polyregion_any_of

  function polyregion_binary_search(in, n, value, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: value
    procedure() :: op
    logical(c_bool) :: r
    r = .false.
    call polyregion_import("polyregion_import:spectra:spectra.binary_search")
    error stop 'compiler did not replace'
  end function polyregion_binary_search

  subroutine polyregion_copy(in, n, out)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: out
    call polyregion_import("polyregion_import:spectra:spectra.copy")
    error stop 'compiler did not replace'
  end subroutine polyregion_copy

  function polyregion_copy_if(in, n, out, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: out
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.copy_if")
    error stop 'compiler did not replace'
  end function polyregion_copy_if

  subroutine polyregion_copy_n(in, n, out)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: out
    call polyregion_import("polyregion_import:spectra:spectra.copy_n")
    error stop 'compiler did not replace'
  end subroutine polyregion_copy_n

  function polyregion_count(in, n, value) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: value
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.count")
    error stop 'compiler did not replace'
  end function polyregion_count

  function polyregion_count_if(in, n, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.count_if")
    error stop 'compiler did not replace'
  end function polyregion_count_if

  function polyregion_equal(a, b, n, op) result(r)
    type(*), dimension(*), intent(in) :: a
    type(*), dimension(*), intent(in) :: b
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    logical(c_bool) :: r
    r = .false.
    call polyregion_import("polyregion_import:spectra:spectra.equal")
    error stop 'compiler did not replace'
  end function polyregion_equal

  subroutine polyregion_exclusive_scan(in, out, n, init, op)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: init
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.exclusive_scan")
    error stop 'compiler did not replace'
  end subroutine polyregion_exclusive_scan

  subroutine polyregion_exclusive_scan_by_key(keys, vals, out, n, init, eq, op)
    type(*), dimension(*), intent(in) :: keys
    type(*), dimension(*), intent(in) :: vals
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: init
    procedure() :: eq
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.exclusive_scan_by_key")
    error stop 'compiler did not replace'
  end subroutine polyregion_exclusive_scan_by_key

  subroutine polyregion_fill(out, n, v)
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: v
    call polyregion_import("polyregion_import:spectra:spectra.fill")
    error stop 'compiler did not replace'
  end subroutine polyregion_fill

  subroutine polyregion_fill_n(out, n, v)
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: v
    call polyregion_import("polyregion_import:spectra:spectra.fill_n")
    error stop 'compiler did not replace'
  end subroutine polyregion_fill_n

  function polyregion_find(in, n, value) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: value
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.find")
    error stop 'compiler did not replace'
  end function polyregion_find

  function polyregion_find_if(in, n, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.find_if")
    error stop 'compiler did not replace'
  end function polyregion_find_if

  function polyregion_find_if_not(in, n, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.find_if_not")
    error stop 'compiler did not replace'
  end function polyregion_find_if_not

  subroutine polyregion_for_each(data, n, op)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.for_each")
    error stop 'compiler did not replace'
  end subroutine polyregion_for_each

  subroutine polyregion_for_each_n(data, n, op)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.for_each_n")
    error stop 'compiler did not replace'
  end subroutine polyregion_for_each_n

  subroutine polyregion_gather(map, n, in, in_n, out)
    integer(c_int32_t), intent(in) :: map(*)
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: in_n
    type(*), dimension(*), intent(inout) :: out
    call polyregion_import("polyregion_import:spectra:spectra.gather")
    error stop 'compiler did not replace'
  end subroutine polyregion_gather

  subroutine polyregion_generate(out, n, op)
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.generate")
    error stop 'compiler did not replace'
  end subroutine polyregion_generate

  subroutine polyregion_generate_n(out, n, op)
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.generate_n")
    error stop 'compiler did not replace'
  end subroutine polyregion_generate_n

  function polyregion_includes(a, na, b, nb, op) result(r)
    type(*), dimension(*), intent(in) :: a
    integer(c_int32_t), intent(in), value :: na
    type(*), dimension(*), intent(in) :: b
    integer(c_int32_t), intent(in), value :: nb
    procedure() :: op
    logical(c_bool) :: r
    r = .false.
    call polyregion_import("polyregion_import:spectra:spectra.includes")
    error stop 'compiler did not replace'
  end function polyregion_includes

  subroutine polyregion_inclusive_scan(in, out, n, op)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.inclusive_scan")
    error stop 'compiler did not replace'
  end subroutine polyregion_inclusive_scan

  subroutine polyregion_inclusive_scan_by_key(keys, vals, out, n, eq, op)
    type(*), dimension(*), intent(in) :: keys
    type(*), dimension(*), intent(in) :: vals
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: eq
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.inclusive_scan_by_key")
    error stop 'compiler did not replace'
  end subroutine polyregion_inclusive_scan_by_key

  subroutine polyregion_inner_product(a, b, n, init, op_reduce, op_product, polyregion_result)
    type(*), dimension(*), intent(in) :: a
    type(*), dimension(*), intent(in) :: b
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: init
    procedure() :: op_reduce
    procedure() :: op_product
    type(*), intent(inout) :: polyregion_result
    call polyregion_import("polyregion_import:spectra:spectra.inner_product")
    error stop 'compiler did not replace'
  end subroutine polyregion_inner_product

  function polyregion_is_partitioned(data, n, op) result(r)
    type(*), dimension(*), intent(in) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    logical(c_bool) :: r
    r = .false.
    call polyregion_import("polyregion_import:spectra:spectra.is_partitioned")
    error stop 'compiler did not replace'
  end function polyregion_is_partitioned

  function polyregion_is_sorted(data, n, op) result(r)
    type(*), dimension(*), intent(in) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    logical(c_bool) :: r
    r = .false.
    call polyregion_import("polyregion_import:spectra:spectra.is_sorted")
    error stop 'compiler did not replace'
  end function polyregion_is_sorted

  function polyregion_is_sorted_until(data, n, op) result(r)
    type(*), dimension(*), intent(in) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.is_sorted_until")
    error stop 'compiler did not replace'
  end function polyregion_is_sorted_until

  function polyregion_lower_bound(in, n, value, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: value
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.lower_bound")
    error stop 'compiler did not replace'
  end function polyregion_lower_bound

  subroutine polyregion_max_element(in, n, op, polyregion_result)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    type(*), intent(inout) :: polyregion_result
    call polyregion_import("polyregion_import:spectra:spectra.max_element")
    error stop 'compiler did not replace'
  end subroutine polyregion_max_element

  subroutine polyregion_merge(a, na, b, nb, out, op)
    type(*), dimension(*), intent(in) :: a
    integer(c_int32_t), intent(in), value :: na
    type(*), dimension(*), intent(in) :: b
    integer(c_int32_t), intent(in), value :: nb
    type(*), dimension(*), intent(inout) :: out
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.merge")
    error stop 'compiler did not replace'
  end subroutine polyregion_merge

  subroutine polyregion_min_element(in, n, op, polyregion_result)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    type(*), intent(inout) :: polyregion_result
    call polyregion_import("polyregion_import:spectra:spectra.min_element")
    error stop 'compiler did not replace'
  end subroutine polyregion_min_element

  subroutine polyregion_minmax_element(in, n, min_out, max_out, op)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: min_out
    type(*), dimension(*), intent(inout) :: max_out
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.minmax_element")
    error stop 'compiler did not replace'
  end subroutine polyregion_minmax_element

  function polyregion_mismatch(a, b, n, op) result(r)
    type(*), dimension(*), intent(in) :: a
    type(*), dimension(*), intent(in) :: b
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.mismatch")
    error stop 'compiler did not replace'
  end function polyregion_mismatch

  function polyregion_none_of(in, n, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    logical(c_bool) :: r
    r = .false.
    call polyregion_import("polyregion_import:spectra:spectra.none_of")
    error stop 'compiler did not replace'
  end function polyregion_none_of

  subroutine polyregion_partition(data, n, op)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.partition")
    error stop 'compiler did not replace'
  end subroutine polyregion_partition

  function polyregion_partition_point(in, n, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.partition_point")
    error stop 'compiler did not replace'
  end function polyregion_partition_point

  subroutine polyregion_reduce(in, n, init, op, polyregion_result)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: init
    procedure() :: op
    type(*), intent(inout) :: polyregion_result
    call polyregion_import("polyregion_import:spectra:spectra.reduce")
    error stop 'compiler did not replace'
  end subroutine polyregion_reduce

  function polyregion_reduce_by_key(keys, vals, kout, vout, n, eq, op) result(r)
    type(*), dimension(*), intent(in) :: keys
    type(*), dimension(*), intent(in) :: vals
    type(*), dimension(*), intent(inout) :: kout
    type(*), dimension(*), intent(inout) :: vout
    integer(c_int32_t), intent(in), value :: n
    procedure() :: eq
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.reduce_by_key")
    error stop 'compiler did not replace'
  end function polyregion_reduce_by_key

  function polyregion_remove(data, n, value) result(r)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: value
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.remove")
    error stop 'compiler did not replace'
  end function polyregion_remove

  function polyregion_remove_copy(in, n, out, value) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: out
    type(*), intent(in) :: value
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.remove_copy")
    error stop 'compiler did not replace'
  end function polyregion_remove_copy

  function polyregion_remove_copy_if(in, n, out, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: out
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.remove_copy_if")
    error stop 'compiler did not replace'
  end function polyregion_remove_copy_if

  function polyregion_remove_if(data, n, op) result(r)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.remove_if")
    error stop 'compiler did not replace'
  end function polyregion_remove_if

  subroutine polyregion_replace(io, n, oldv, newv)
    type(*), dimension(*), intent(inout) :: io
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: oldv
    type(*), intent(in) :: newv
    call polyregion_import("polyregion_import:spectra:spectra.replace")
    error stop 'compiler did not replace'
  end subroutine polyregion_replace

  subroutine polyregion_replace_copy(in, out, n, oldv, newv)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: oldv
    type(*), intent(in) :: newv
    call polyregion_import("polyregion_import:spectra:spectra.replace_copy")
    error stop 'compiler did not replace'
  end subroutine polyregion_replace_copy

  subroutine polyregion_replace_copy_if(in, out, n, new_value, op)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: new_value
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.replace_copy_if")
    error stop 'compiler did not replace'
  end subroutine polyregion_replace_copy_if

  subroutine polyregion_replace_if(data, n, new_value, op)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: new_value
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.replace_if")
    error stop 'compiler did not replace'
  end subroutine polyregion_replace_if

  subroutine polyregion_reverse(data, n)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    call polyregion_import("polyregion_import:spectra:spectra.reverse")
    error stop 'compiler did not replace'
  end subroutine polyregion_reverse

  subroutine polyregion_reverse_copy(in, n, out)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: out
    call polyregion_import("polyregion_import:spectra:spectra.reverse_copy")
    error stop 'compiler did not replace'
  end subroutine polyregion_reverse_copy

  subroutine polyregion_scatter(in, n, map, out, out_n)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    integer(c_int32_t), intent(in) :: map(*)
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: out_n
    call polyregion_import("polyregion_import:spectra:spectra.scatter")
    error stop 'compiler did not replace'
  end subroutine polyregion_scatter

  function polyregion_search(in, n, sub, m, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(in) :: sub
    integer(c_int32_t), intent(in), value :: m
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.search")
    error stop 'compiler did not replace'
  end function polyregion_search

  function polyregion_search_n(in, n, count, value, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    integer(c_int32_t), intent(in), value :: count
    type(*), intent(in) :: value
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.search_n")
    error stop 'compiler did not replace'
  end function polyregion_search_n

  subroutine polyregion_sequence(out, n, init, step)
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: init
    type(*), intent(in) :: step
    call polyregion_import("polyregion_import:spectra:spectra.sequence")
    error stop 'compiler did not replace'
  end subroutine polyregion_sequence

  function polyregion_set_difference(a, na, b, nb, out, op) result(r)
    type(*), dimension(*), intent(in) :: a
    integer(c_int32_t), intent(in), value :: na
    type(*), dimension(*), intent(in) :: b
    integer(c_int32_t), intent(in), value :: nb
    type(*), dimension(*), intent(inout) :: out
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.set_difference")
    error stop 'compiler did not replace'
  end function polyregion_set_difference

  function polyregion_set_intersection(a, na, b, nb, out, out_n, op) result(r)
    type(*), dimension(*), intent(in) :: a
    integer(c_int32_t), intent(in), value :: na
    type(*), dimension(*), intent(in) :: b
    integer(c_int32_t), intent(in), value :: nb
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: out_n
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.set_intersection")
    error stop 'compiler did not replace'
  end function polyregion_set_intersection

  function polyregion_set_union(a, na, b, nb, out, op) result(r)
    type(*), dimension(*), intent(in) :: a
    integer(c_int32_t), intent(in), value :: na
    type(*), dimension(*), intent(in) :: b
    integer(c_int32_t), intent(in), value :: nb
    type(*), dimension(*), intent(inout) :: out
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.set_union")
    error stop 'compiler did not replace'
  end function polyregion_set_union

  subroutine polyregion_sort(data, n, op)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.sort")
    error stop 'compiler did not replace'
  end subroutine polyregion_sort

  subroutine polyregion_sort_by_key(keys, values, n, op)
    type(*), dimension(*), intent(inout) :: keys
    type(*), dimension(*), intent(inout) :: values
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.sort_by_key")
    error stop 'compiler did not replace'
  end subroutine polyregion_sort_by_key

  subroutine polyregion_stable_partition(data, n, op)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.stable_partition")
    error stop 'compiler did not replace'
  end subroutine polyregion_stable_partition

  subroutine polyregion_stable_sort(data, n, op)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.stable_sort")
    error stop 'compiler did not replace'
  end subroutine polyregion_stable_sort

  subroutine polyregion_swap_ranges(a, n, b)
    type(*), dimension(*), intent(inout) :: a
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: b
    call polyregion_import("polyregion_import:spectra:spectra.swap_ranges")
    error stop 'compiler did not replace'
  end subroutine polyregion_swap_ranges

  subroutine polyregion_tabulate(out, n, op)
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.tabulate")
    error stop 'compiler did not replace'
  end subroutine polyregion_tabulate

  subroutine polyregion_transform(in, out, n, op)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.transform")
    error stop 'compiler did not replace'
  end subroutine polyregion_transform

  subroutine polyregion_transform_binary(a, b, out, n, op)
    type(*), dimension(*), intent(in) :: a
    type(*), dimension(*), intent(in) :: b
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.transform_binary")
    error stop 'compiler did not replace'
  end subroutine polyregion_transform_binary

  subroutine polyregion_transform_exclusive_scan(in, out, n, init, map, op)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: init
    procedure() :: map
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.transform_exclusive_scan")
    error stop 'compiler did not replace'
  end subroutine polyregion_transform_exclusive_scan

  subroutine polyregion_transform_inclusive_scan(in, out, n, map, op)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: map
    procedure() :: op
    call polyregion_import("polyregion_import:spectra:spectra.transform_inclusive_scan")
    error stop 'compiler did not replace'
  end subroutine polyregion_transform_inclusive_scan

  subroutine polyregion_transform_reduce(in, n, init, map, op, polyregion_result)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: init
    procedure() :: map
    procedure() :: op
    type(*), intent(inout) :: polyregion_result
    call polyregion_import("polyregion_import:spectra:spectra.transform_reduce")
    error stop 'compiler did not replace'
  end subroutine polyregion_transform_reduce

  function polyregion_unique(data, n, eq) result(r)
    type(*), dimension(*), intent(inout) :: data
    integer(c_int32_t), intent(in), value :: n
    procedure() :: eq
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.unique")
    error stop 'compiler did not replace'
  end function polyregion_unique

  function polyregion_unique_copy(in, n, out, eq) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), dimension(*), intent(inout) :: out
    procedure() :: eq
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.unique_copy")
    error stop 'compiler did not replace'
  end function polyregion_unique_copy

  function polyregion_upper_bound(in, n, value, op) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    type(*), intent(in) :: value
    procedure() :: op
    integer(c_int32_t) :: r
    r = 0
    call polyregion_import("polyregion_import:spectra:spectra.upper_bound")
    error stop 'compiler did not replace'
  end function polyregion_upper_bound

end module spectra_api
