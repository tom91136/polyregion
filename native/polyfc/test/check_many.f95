!CHECK case: many
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1.2.3.

program test
    implicit none
    integer :: i, x
    x = 1
    write(*, '(F0.0)', advance = "no") real(x)
    do concurrent (i = 1:1)
        x = 2
    end do
    write(*, '(F0.0)', advance = "no") real(x)
    do concurrent (i = 1:1)
        x = 3
    end do
    write(*, '(F0.0)', advance = "no") real(x)
end program test
