module example_ffi
  use iso_c_binding, only: c_bool, c_double, c_float, c_int8_t, c_int16_t, c_int32_t, c_int64_t
  implicit none
  private

  public :: polyregion_count
  public :: polyregion_transform

  interface
    subroutine polyregion_interface(identity)
      character(len=*), intent(in) :: identity
    end subroutine polyregion_interface
  end interface

contains

  function polyregion_count(in, n) result(r)
    type(*), dimension(*), intent(in) :: in
    integer(c_int32_t), intent(in), value :: n
    integer(c_int32_t) :: r
    r = 0
    call polyregion_interface("polyregion_interface:example:example.count")
    error stop 'compiler did not replace'
  end function polyregion_count

  subroutine polyregion_transform(in, out, n, op)
    type(*), dimension(*), intent(in) :: in
    type(*), dimension(*), intent(inout) :: out
    integer(c_int32_t), intent(in), value :: n
    procedure() :: op
    call polyregion_interface("polyregion_interface:example:example.transform")
    error stop 'compiler did not replace'
  end subroutine polyregion_transform

end module example_ffi
