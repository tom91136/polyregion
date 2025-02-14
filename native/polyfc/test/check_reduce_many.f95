!CHECK case: reduce_many
!CHECK using: alloc=0,1
!CHECK using: type=integer,real
!CHECK using: kind=4,8
!CHECK using: size=1,10,1000
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -DCHECK_KIND={kind} -DCHECK_SIZE={size} -DCHECK_ALLOC={alloc} -o {output} {input}
!CHECK do: {output}
!CHECK requires: T T T T

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
    CHECK_TYPE(kind = CHECK_KIND), allocatable :: a, b, c, d
    allocate(a)
    allocate(b)
    allocate(c)
    allocate(d)
#else
    CHECK_TYPE(kind = CHECK_KIND) :: a, b, c, d
#endif

    a = -1
    b = -1
    c = -1
    d = -1
    do concurrent (i = 1:CHECK_SIZE) &
        reduce(+:a) reduce(*:b)  reduce(min:c) reduce(max:d)
        a = a + i
        b = b * -1
        c = min(c, i)
        d = max(d, i)
    end do

    write(0, *) a, b, c, d
    write(*, '(L1, 1X, L1, 1X, L1, 1X, L1)', advance = "no")  &
            a == (sum([ (i, i = 1, CHECK_SIZE) ]) - 1), &
            b == (product([ (-1, i = 1, CHECK_SIZE) ]) * -1), &
            c == -1, &
            d == CHECK_SIZE
end program test


