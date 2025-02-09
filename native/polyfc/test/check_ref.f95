!CHECK case: ref
!CHECK using: alloc=0,1
!CHECK using: type=integer,real
!CHECK using: kind=2,4,8
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -DCHECK_KIND={kind} -DCHECK_ALLOC={alloc} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 43.

#ifndef CHECK_TYPE
#define CHECK_TYPE integer
#endif

#ifndef CHECK_KIND
#define CHECK_KIND 4
#endif

#ifndef CHECK_ALLOC
#define CHECK_ALLOC 1
#endif

program test
    implicit none
    integer :: i
#if CHECK_ALLOC == 1
    CHECK_TYPE(kind = CHECK_KIND), allocatable :: x
    allocate(x)
#else
    CHECK_TYPE(kind = CHECK_KIND) :: x
#endif
    x = -1
    do concurrent (i = 1:1)
        x = 42 + i
    end do
    write(*, '(F0.0)', advance = "no") real(x)
end program test


