!CHECK case: char-compare-literal
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 2

program test
    implicit none
    integer, parameter :: N = 4
    integer :: i, a(N)
    character(len = N), parameter :: word = "poly"

    do concurrent (i = 1:N)
        a(i) = merge(i, 0, word(i:i) == "o")
    end do

    write(*, '(I1)', advance = "no") a(2)
end program test
