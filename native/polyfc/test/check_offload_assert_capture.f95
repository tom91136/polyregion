!CHECK case: assert-capture-msg
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1

program test
    ! no `use iso_c_binding` (it makes flang order the module string table non-deterministically); the message
    ! is NUL-terminated in source via char(0), since a Fortran character is space-padded, not NUL-terminated
    implicit none
    interface
        pure subroutine fc_assert_msg(message) bind(C, name = "__polyregion_fc_assert_msg")
            character(len = *), intent(in) :: message
        end subroutine
        function fc_assert_raised() bind(C, name = "__polyregion_fc_assert_raised") result(r)
            integer :: r
        end function
    end interface

    integer, parameter :: N = 256
    integer :: i, a(N)
    character(len = 16) :: msg

    msg = "fortran" // char(0)
    do i = 1, N
        a(i) = i
    end do
    a(7) = -1

    do concurrent (i = 1:N)
        if (a(i) < 0) call fc_assert_msg(msg)
        a(i) = a(i) + 1
    end do

    write(*, '(I1)', advance = "no") fc_assert_raised()
end program test
