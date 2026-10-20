!CHECK case: static_runtime
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-rt=static -o {output} {input}
!CHECK do: {output}
!CHECK requires: 55

program test
    implicit none
    integer :: i, x
    x = 0
    do concurrent (i = 1:10) reduce(+:x)
        x = x + i
    end do
    write(*, '(I0)', advance = "no") x
end program test
