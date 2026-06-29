!CHECK case: ichar-scalar-read
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 111

program test
    implicit none
    integer, parameter :: N = 4
    integer :: i, a(N)
    character(len = N) :: word

    word = "poly"
    do concurrent (i = 1:N)
        a(i) = ichar(word(i:i))
    end do

    write(*, '(I3)', advance = "no") a(2)
end program test
