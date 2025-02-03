!CHECK case: empty
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: done

program test
    implicit none
    integer :: i
    do concurrent (i = 1:1)
    end do
    write(*, '(A)', advance = "no") "done"
end program test
