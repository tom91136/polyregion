!CHECK case: assert-realloc-unallocated
!CHECK offload-only
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: POLYRT_DEBUG=1 {output}
!CHECK requires: 1
!CHECK stderr-count: 1 assignment to an unallocated allocatable is not supported in an offloaded region

program test
    implicit none
    interface
        function fc_assert_raised() bind(C, name = "__polyregion_fc_assert_raised") result(r)
            integer :: r
        end function
    end interface

    integer, allocatable :: x
    integer :: i

    do concurrent (i = 1:1)
        x = i
    end do

    write(*, '(I1)', advance = "no") fc_assert_raised()
end program test
