!CHECK case: explicit_shape
!CHECK using: kind=4,8
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_KIND={kind} -o {output} {input}
!CHECK do: {output}
!CHECK requires: pass

#ifndef CHECK_KIND
#define CHECK_KIND 4
#endif

module m
    implicit none
contains

    subroutine runAll(a, b, c, n, times, s)
        integer(kind = 8), intent(in) :: n, times
        real(kind = CHECK_KIND), dimension(n), intent(inout) :: a, b, c
        real(kind = CHECK_KIND), intent(out) :: s
        integer(kind = 8) :: i, k

        do concurrent (i = 1 : n)
            a(i) = real(mod(i, 7_8) + 1, kind = CHECK_KIND)
            b(i) = real(mod(i, 5_8) + 1, kind = CHECK_KIND)
            c(i) = 0.0_CHECK_KIND
        end do

        do k = 1, times
            do concurrent (i = 1 : n); c(i) = a(i); end do
            do concurrent (i = 1 : n); c(i) = a(i) + b(i); end do
            s = 0.0_CHECK_KIND
            do concurrent (i = 1 : n) reduce(+ : s)
                s = s + a(i) * b(i)
            end do
        end do
    end subroutine runAll
end module m

program test
    use m
    implicit none
    real(kind = CHECK_KIND), allocatable :: a(:), b(:), c(:)
    real(kind = CHECK_KIND) :: s, ref
    integer(kind = 8) :: n, i
    logical :: ok

    n = 1024
    allocate(a(n), b(n), c(n))
    call runAll(a, b, c, n, 4_8, s)

    ref = 0.0_CHECK_KIND
    do i = 1, n
        ref = ref + real(mod(i, 7_8) + 1, kind = CHECK_KIND) * real(mod(i, 5_8) + 1, kind = CHECK_KIND)
    end do

    ok = abs(s - ref) <= 1.0e-3_CHECK_KIND * abs(ref)
    if (ok) then
        write(*, '(A)', advance = "no") "pass"
    else
        print *, "fail off=", s, " ref=", ref
    end if
end program test
