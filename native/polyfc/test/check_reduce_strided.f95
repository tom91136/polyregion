!CHECK case: reduce_strided
!CHECK using: type=integer,real
!CHECK using: kind=4,8
!CHECK using: lower=1,4
!CHECK using: size=1,10,1000
!CHECK using: step=2,3
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -DCHECK_KIND={kind} -DCHECK_LOWER={lower} -DCHECK_SIZE={size} -DCHECK_STEP={step} -o {output} {input}
!CHECK do: {output}
!CHECK requires: T T T T

#ifndef CHECK_TYPE
#define CHECK_TYPE integer
#endif

#ifndef CHECK_KIND
#define CHECK_KIND 4
#endif

#ifndef CHECK_LOWER
#define CHECK_LOWER 1
#endif

#ifndef CHECK_SIZE
#define CHECK_SIZE 1
#endif

#ifndef CHECK_STEP
#define CHECK_STEP 2
#endif

program test
    implicit none
    integer :: i
    CHECK_TYPE(kind = CHECK_KIND) :: s, m, lo, e, v(CHECK_SIZE)

    v = [ (mod(i * 7, 13), i = 1, CHECK_SIZE) ]
    s = -1
    m = -1
    lo = 100
    e = 5
    do concurrent (i = CHECK_LOWER:CHECK_SIZE:CHECK_STEP) &
        reduce(+:s) reduce(max:m) reduce(min:lo)
        s = s + i
        m = max(m, v(i))
        lo = min(lo, v(i))
    end do
    do concurrent (i = 1:0:CHECK_STEP) &
        reduce(+:e)
        e = e + i
    end do

    write(0, *) s, m, lo, e
    write(*, '(L1, 1X, L1, 1X, L1, 1X, L1)', advance = "no")  &
            s == (sum([ (i, i = CHECK_LOWER, CHECK_SIZE, CHECK_STEP) ]) - 1), &
            m == max(-1, maxval([ (mod(i * 7, 13), i = CHECK_LOWER, CHECK_SIZE, CHECK_STEP) ])), &
            lo == min(100, minval([ (mod(i * 7, 13), i = CHECK_LOWER, CHECK_SIZE, CHECK_STEP) ])), &
            e == 5
end program test
