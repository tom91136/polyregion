!CHECK case: reduce
!CHECK using: alloc=0,1
!CHECK using: type=integer,real
!CHECK using: kind=4,8
!CHECK using: size=1,10,1000
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -DCHECK_KIND={kind} -DCHECK_SIZE={size} -DCHECK_ALLOC={alloc} -o {output} {input}
!CHECK do: {output}
!CHECK requires: T

#ifndef CHECK_TYPE
#define CHECK_TYPE integer
#endif

#ifndef CHECK_KIND
#define CHECK_KIND 4
#endif

#ifndef CHECK_SIZE
#define CHECK_SIZE 1
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
    do concurrent (i = 1:CHECK_SIZE) &
        reduce(+:x)
        x = x + i
    end do

    write(*, '(L1)', advance = "no")  (sum([ (i, i = 1, CHECK_SIZE) ]) - 1) == x
end program test


