!CHECK case: call_expr_arg
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: pass

module m
    implicit none
    integer, parameter :: STRIDE = 16
    integer, parameter :: N = 4096
contains
    pure subroutine pick(data, off, out)
        integer, intent(in) :: data(*), off
        integer, intent(out) :: out
        out = data(off + 1)
    end subroutine pick
end module m

program test
    use m
    implicit none
    integer, allocatable :: data(:), got(:)
    integer :: i
    logical :: ok

    allocate(data(STRIDE * N), got(N))
    do i = 1, STRIDE * N
        data(i) = i
    end do
    got = -1

    do concurrent (i = 1 : N)
        call pick(data, (i - 1) * STRIDE, got(i))
    end do

    ok = .true.
    do i = 1, N
        if (got(i) /= (i - 1) * STRIDE + 1) ok = .false.
    end do

    if (ok) then
        write(*, '(A)', advance = "no") "pass"
    else
        print *, "fail, got = ", got(1:min(N, 16))
    end if
end program test
