!CHECK case: assert-element
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1 972976

!CHECK case: assert-element-reduce
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_REDUCE -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1 972976

program test
    implicit none
    interface
        pure subroutine fc_assert() bind(C, name = "__polyregion_fc_assert")
        end subroutine
        function fc_assert_raised() bind(C, name = "__polyregion_fc_assert_raised") result(r)
            integer :: r
        end function
    end interface

    integer, parameter :: N = 1000003
    integer :: i, x
    integer, allocatable :: a(:), done(:)

    allocate(a(N), done(N))
    do i = 1, N
        a(i) = 1
        done(i) = 0
    end do
    do i = 10, N, 37
        a(i) = -1
    end do

    x = 0
#ifdef CHECK_REDUCE
    do concurrent (i = 1:N) reduce(+:x)
        if (a(i) < 0) then
            call fc_assert()
        else
            done(i) = 1
            x = x + 1
        end if
    end do
#else
    do concurrent (i = 1:N)
        if (a(i) < 0) then
            call fc_assert()
        else
            done(i) = 1
        end if
    end do
#endif

    write(*, '(I1,1X,I0)', advance = "no") fc_assert_raised(), sum(done)
end program test
