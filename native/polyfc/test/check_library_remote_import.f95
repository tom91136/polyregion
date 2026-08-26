!CHECK case: remote-package-import
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK do: {output}
!CHECK requires: 42

module foo_remote_ffi
  use iso_c_binding, only: c_int32_t
  implicit none
  private

  public :: remote_increment

  interface
    subroutine polyregion_interface(identity)
      character(len=*), intent(in) :: identity
    end subroutine polyregion_interface
  end interface

contains

  subroutine remote_increment(x, r)
    integer(c_int32_t), intent(in), value :: x
    type(*), intent(inout) :: r
    call polyregion_interface("polyregion_interface:foo:bar.remote_increment")
    error stop 'compiler did not replace'
  end subroutine remote_increment

end module foo_remote_ffi

program main
  use foo_remote_ffi
  use iso_c_binding, only: c_int32_t
  integer(c_int32_t) :: result
  call remote_increment(41_c_int32_t, result)
  write (*, '(I0)', advance='no') result
end program main
