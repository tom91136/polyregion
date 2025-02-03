!CHECK case: write
!CHECK using: alloc=0,1
!CHECK using: type=integer,real
!CHECK using: kind=4,8
!CHECK using: size=1,10,100
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -DCHECK_KIND={kind} -DCHECK_SIZE={size} -DCHECK_ALLOC={alloc} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 42.

#ifndef CHECK_TYPE
#define CHECK_TYPE integer
#endif

#ifndef CHECK_KIND
#define CHECK_KIND 4
#endif

#ifndef CHECK_SIZE
#define CHECK_SIZE 100
#endif

#ifndef CHECK_ALLOC
#define CHECK_ALLOC 1
#endif

program test
    implicit none
    integer :: i
#if CHECK_ALLOC == 1
    CHECK_TYPE(kind = CHECK_KIND), allocatable :: x(:)
    allocate(x(CHECK_SIZE))
#else
    CHECK_TYPE(kind = CHECK_KIND) :: x(CHECK_SIZE)
#endif
    x = -1
    do concurrent (i = 1:1);
        x(CHECK_SIZE) = 42
    end do
    write(*, '(F0.0)', advance = "no") real(x(CHECK_SIZE))
    if (CHECK_SIZE > 1 .and. any(x(1:CHECK_SIZE - 1) /= -1)) then
        print*, "fail, init array modified = ", x
    end if
end program test
