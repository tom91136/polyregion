!CHECK case: write_in_derived
!CHECK using: alloc_arr=0,1
!CHECK using: alloc_derived=0,1
!CHECK using: type=integer,real
!CHECK using: kind=4,8
!CHECK using: size=1,10,10000
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -DCHECK_KIND={kind} -DCHECK_SIZE={size} -DCHECK_ALLOC_ARR={alloc_arr} -DCHECK_ALLOC_DERIVED={alloc_derived} -o {output} {input}
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

#ifndef CHECK_ALLOC_ARR
#define CHECK_ALLOC_ARR 1
#endif

#ifndef CHECK_ALLOC_DERIVED
#define CHECK_ALLOC_DERIVED 1
#endif

module mod
    implicit none

    type :: FooT
#if CHECK_ALLOC_ARR == 1
        CHECK_TYPE(kind = CHECK_KIND), allocatable :: x(:)
#else
        CHECK_TYPE(kind = CHECK_KIND) :: x(CHECK_SIZE)
#endif
    end type FooT

end module mod

program test
    use mod
    implicit none
    integer :: i
#if CHECK_ALLOC_DERIVED == 1
    type(FooT), allocatable :: foo
    allocate(foo)
#else
    type(FooT) :: foo
#endif

#if CHECK_ALLOC_ARR == 1
    allocate(foo%x(CHECK_SIZE))
#endif

    foo%x = -1
    do concurrent (i = 1:CHECK_SIZE)
        foo%x(CHECK_SIZE) = 42
    end do

    write(*, '(F0.0)', advance = "no") real(foo%x(CHECK_SIZE))
    if (CHECK_SIZE > 1 .and. any(foo%x(1:CHECK_SIZE - 1) /= -1)) then
        print*, "fail, init array modified = ", foo%x
    end if
end program test
