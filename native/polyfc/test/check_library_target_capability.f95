!CHECK case: target-qualified-capability
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-path={output}.packages -fstdpar-library-capabilities=demo.a@host,demo.a@cuda,demo.a@opencl1_1,demo.a@opencl2_0,demo.a@spirv64_kernel,demo.a@spirv_glcompute,demo.a@metal1_0,demo.b@c11 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 42

!CHECK case: target-qualified-capability-multi-target
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-arch={polyfc_arch};x86_64@native -fstdpar-library-path={output}.packages -fstdpar-library-capabilities=demo.a@host,demo.a@cuda,demo.a@opencl1_1,demo.a@opencl2_0,demo.a@spirv64_kernel,demo.a@spirv_glcompute,demo.a@metal1_0,demo.b@x86_64 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 42

!CHECK case: target-qualified-capability-unknown-target
!CHECK offload-only
!CHECK compile-fails: unknown target `nonesuch` in library capability `demo.a@nonesuch`
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-capabilities=demo.a@nonesuch -o {output} {input}

module foo_targeted_ffi
  use iso_c_binding, only: c_int32_t
  implicit none
  private

  public :: targeted_increment

  interface
    subroutine polyregion_interface(identity)
      character(len=*), intent(in) :: identity
    end subroutine polyregion_interface
  end interface

contains

  subroutine targeted_increment(x, r)
    integer(c_int32_t), intent(in), value :: x
    type(*), intent(inout) :: r
    call polyregion_interface("polyregion_interface:foo:bar.targeted_increment")
    error stop 'compiler did not replace'
  end subroutine targeted_increment

end module foo_targeted_ffi

program main
  use foo_targeted_ffi
  use iso_c_binding, only: c_int32_t
  integer(c_int32_t) :: result
  call targeted_increment(41_c_int32_t, result)
  write (*, '(I0)', advance='no') result
end program main
