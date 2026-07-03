!CHECK case: assert-barrier
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1

program test
    ! no `use iso_c_binding`: it makes flang order the module string table non-deterministically (fails the
    ! repro check), and bind(C) does not need it - default integer is C int
    implicit none
    interface
        pure subroutine fc_assert() bind(C, name = "__polyregion_fc_assert")
        end subroutine
        function fc_assert_raised() bind(C, name = "__polyregion_fc_assert_raised") result(r)
            integer :: r
        end function
    end interface

    integer, parameter :: N = 256
    integer :: i, x, a(N)

    do i = 1, N
        a(i) = i
    end do
    a(7) = -1

    x = 0
    do concurrent (i = 1:N) &
        reduce(+:x)
        if (a(i) < 0) call fc_assert()
        x = x + a(i)
    end do

    write(*, '(I1)', advance = "no") fc_assert_raised()
end program test
