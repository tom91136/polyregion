module example_ffi
  use iso_c_binding, only: c_bool, c_double, c_float, c_int16_t, c_int32_t, c_int64_t, c_int8_t
  implicit none
  private

  public :: polyregion_count
  public :: polyregion_transform

  interface
    subroutine polyregion_import(identity)
      character(len=*), intent(in) :: identity
    end subroutine polyregion_import
  end interface

  abstract interface
    function polyregion_transform_op_r4(arg0) result(r)
      import :: c_bool, c_double, c_float, c_int16_t, c_int32_t, c_int64_t, c_int8_t
      real(c_float), intent(in), value :: arg0
      real(c_float) :: r
    end function polyregion_transform_op_r4
  end interface

  interface polyregion_count
    module procedure polyregion_count_r4
  end interface polyregion_count

  interface polyregion_transform
    module procedure polyregion_transform_r4
  end interface polyregion_transform

contains

  function polyregion_count_r4(in, n) result(r)
    real(c_float), intent(in) :: in(*)
    integer(c_int32_t), intent(in), value :: n
    integer(c_int32_t) :: r
    call polyregion_import("polyregion_import:example:example.count:r4")
    error stop 'compiler did not replace'
  end function polyregion_count_r4

  subroutine polyregion_transform_r4(in, out, n, op)
    real(c_float), intent(in) :: in(*)
    real(c_float), intent(out) :: out(*)
    integer(c_int32_t), intent(in), value :: n
    procedure(polyregion_transform_op_r4) :: op
    call polyregion_import("polyregion_import:example:example.transform:r4")
    error stop 'compiler did not replace'
  end subroutine polyregion_transform_r4

end module example_ffi
