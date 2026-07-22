!CHECK case: host-write-between-launches
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 42 7

program test
    implicit none
    integer :: x(1)

    x = 41
    call bump(x)
    write(*, '(I0,1X)', advance = "no") x(1)

    x = 6
    call bump(x)
    write(*, '(I0)', advance = "no") x(1)

contains
    subroutine bump(a)
        integer, intent(inout) :: a(1)
        integer :: i
        do concurrent (i = 1:1)
            a(i) = a(i) + 1
        end do
    end subroutine bump
end program test
