!CHECK case: element_in_derived
!CHECK offload-only
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
!CHECK do: {package_fixture} --assert-no-array-value-bindings {output}.polyast

module array_element_in_derived_m
    implicit none

    type :: FooT
        integer(kind = 8) :: x(10000), y(10000), z(10000)
    end type FooT

contains

    integer(kind = 8) function element_sum(foo, i) bind(c, name = "element_sum")
        type(FooT), intent(in) :: foo
        integer, intent(in) :: i
        element_sum = foo%y(i) + foo%z(i)
    end function

    subroutine element_copy(foo, i) bind(c, name = "element_copy")
        type(FooT), intent(inout) :: foo
        integer, intent(in) :: i
        foo%x(i) = foo%y(i)
    end subroutine

end module array_element_in_derived_m
