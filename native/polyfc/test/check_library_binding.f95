!CHECK case: scalar-package
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK do: {output}
!CHECK requires: 42

module foo_ffi
  use iso_c_binding, only: c_int32_t
  implicit none
  private

  public :: increment

  interface
    subroutine polyregion_interface(identity)
      character(len=*), intent(in) :: identity
    end subroutine polyregion_interface
  end interface

contains

  subroutine ignore(identity)
    character(len=*), intent(in) :: identity
  end subroutine ignore

  function increment(x) result(r)
    integer(c_int32_t), intent(in), value :: x
    integer(c_int32_t) :: r
    r = 0
    call ignore("polyregion_interface:unrelated:declaration")
    call polyregion_interface("polyregion_interface:foo:bar.increment")
    error stop 'compiler did not replace'
  end function increment

end module foo_ffi

program main
  use foo_ffi
  write (*, '(I0)', advance='no') increment(41)
end program main
