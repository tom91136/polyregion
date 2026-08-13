module callbacks
  use iso_c_binding
  type :: point
    integer(c_int64_t) :: tag
    real(c_float) :: x
    real(c_float) :: y
  end type point
contains
  subroutine twice(value, result)
    real(c_float), intent(in) :: value
    real(c_float) :: result
    result = value * 2
  end subroutine twice

  subroutine seed(result)
    real(c_float) :: result
    result = 1
  end subroutine seed

  subroutine positive(value, result)
    real(c_float), intent(in) :: value
    logical(c_bool) :: result
    result = value > 0
  end subroutine positive

  subroutine equal(lhs, rhs, result)
    real(c_float), intent(in) :: lhs, rhs
    logical(c_bool) :: result
    result = lhs == rhs
  end subroutine equal

  subroutine plus(lhs, rhs, result)
    real(c_float), intent(in) :: lhs, rhs
    real(c_float) :: result
    result = lhs + rhs
  end subroutine plus

  subroutine times(lhs, rhs, result)
    real(c_float), intent(in) :: lhs, rhs
    real(c_float) :: result
    result = lhs * rhs
  end subroutine times

  subroutine magnitude(value, result)
    type(point), intent(in) :: value
    real(c_double) :: result
    result = sqrt(real(value%x, c_double)**2 + real(value%y, c_double)**2)
  end subroutine magnitude

  subroutine combine_point(lhs, rhs, result)
    type(point), intent(in) :: lhs, rhs
    type(point) :: result
    result = lhs
    result%x = lhs%x + rhs%x
    result%y = lhs%y + rhs%y
  end subroutine combine_point

  subroutine increment_i64(value, result)
    integer(c_int64_t), intent(in) :: value
    integer(c_int64_t) :: result
    result = value + 1_c_int64_t
  end subroutine increment_i64
end module callbacks

program check
  use iso_c_binding
  use spectra_api
  use callbacks
  real(c_float) :: input(4), output(4)
  type(point) :: points(4), point_init, point_result
  real(c_double) :: magnitudes(4)
  integer(c_int64_t) :: integers(4), incremented(4)
  real(c_float) :: reduced
  integer(c_int32_t) :: groups
  logical(c_bool) :: accepted
  call polyregion_transform(input, output, 4_c_int32_t, twice)
  call polyregion_transform(points, magnitudes, 4_c_int32_t, magnitude)
  call polyregion_transform(integers, incremented, 4_c_int32_t, increment_i64)
  call polyregion_transform_binary(input, input, output, 4_c_int32_t, plus)
  call polyregion_transform_inclusive_scan(input, output, 4_c_int32_t, twice, plus)
  call polyregion_transform_exclusive_scan(input, output, 4_c_int32_t, 0.0_c_float, twice, plus)
  call polyregion_generate(output, 4_c_int32_t, seed)
  call polyregion_copy(input, 4_c_int32_t, output)
  accepted = polyregion_all_of(input, 4_c_int32_t, positive)
  call polyregion_inner_product(input, output, 4_c_int32_t, 0.0_c_float, plus, times, reduced)
  call polyregion_transform_reduce(input, 4_c_int32_t, 0.0_c_float, twice, plus, reduced)
  call polyregion_reduce(points, 4_c_int32_t, point_init, combine_point, point_result)
  groups = polyregion_reduce_by_key(input, input, output, output, 4_c_int32_t, equal, plus)
end program check
