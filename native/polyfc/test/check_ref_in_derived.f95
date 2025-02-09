! Skip reals here as kind=1|2 aren't implemented yet in flang

!CHECK case: ref_derived
!CHECK using: type=integer
!CHECK using: kind=1,2,4,8
!CHECK using: alloc_member=0,1
!CHECK using: alloc_derived=0,1
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -DCHECK_KIND={kind} -DCHECK_ALLOC_MEMBER={alloc_member} -DCHECK_ALLOC_DERIVED={alloc_derived} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 43.

#ifndef CHECK_TYPE
#define CHECK_TYPE integer
#endif

#ifndef CHECK_KIND
#define CHECK_KIND 4
#endif

#ifndef CHECK_ALLOC_MEMBER
#define CHECK_ALLOC_MEMBER 1
#endif

#ifndef CHECK_ALLOC_DERIVED
#define CHECK_ALLOC_DERIVED 1
#endif

module mod
    implicit none

    type :: FooT
        CHECK_TYPE(kind = CHECK_KIND) :: extra0
#if CHECK_ALLOC_MEMBER == 1
        CHECK_TYPE(kind = CHECK_KIND), allocatable :: bar
#else
    CHECK_TYPE(kind = CHECK_KIND) :: bar
#endif
        CHECK_TYPE(kind = CHECK_KIND) :: extra1
    end type
end module

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

#if CHECK_ALLOC_MEMBER == 1
    allocate(foo%bar)
#endif

    foo%bar = -1
    foo%extra0 = 2
    foo%extra1 = 3
    do concurrent (i = 1:1)
        foo%bar = 42 + i
    end do
    write(*, '(F0.0)', advance = "no") real(foo%bar)

contains

end program test
