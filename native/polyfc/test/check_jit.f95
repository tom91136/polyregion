!CHECK case: capture
!CHECK using: jit=static,dynamic
!CHECK using: type=integer,real
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_TYPE={type} -fstdpar-jit={jit} -o {output} {input}
!CHECK do: POLYRT_JIT_CACHE=off POLYRT_JIT_SPECIALISE=1 POLYRT_JIT_SPECIALISE_HOT=1 {output}
!CHECK requires: 42.0 7.0

#ifndef CHECK_TYPE
#define CHECK_TYPE integer
#endif

program test
    implicit none
    CHECK_TYPE :: x(1), y(1), first, second

    first = 41
    second = 6
    call bump(x, first)
    call bump(y, second)
    write(*, '(F0.1,1X)', advance = "no") real(x(1))
    write(*, '(F0.1)', advance = "no") real(y(1))

contains
    subroutine bump(a, v)
        CHECK_TYPE, intent(inout) :: a(1)
        CHECK_TYPE, intent(in), value :: v
        integer :: i
        do concurrent (i = 1:1)
            a(i) = v + 1
        end do
    end subroutine bump
end program test
