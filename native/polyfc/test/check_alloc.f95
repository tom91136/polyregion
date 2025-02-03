!CHECK case: alloc
!CHECK using: type=integer,real
!CHECK using: kind=4,8
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -DCHECK_KIND={kind} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 42.

#ifndef CHECK_TYPE
#define CHECK_TYPE integer
#endif

#ifndef CHECK_KIND
#define CHECK_KIND 4
#endif

program test
    implicit none
    CHECK_TYPE, allocatable :: x
    integer :: i
    allocate(x)
    x = -1
    do concurrent (i = 1:1);
        x = 42
    end do
    write(*, '(F0.0)', advance = "no") real(x)
end program test
