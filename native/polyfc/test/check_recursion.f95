!CHECK case: fib
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=0 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 55

!CHECK case: ackermann
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-stack=128 -DCASE=1 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 61

!CHECK case: mutual
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=2 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1

#ifndef CASE
#define CASE 0
#endif

program test
  implicit none
  integer :: out(1), i
  out = 0
  do concurrent (i = 1:1)
#if CASE == 0
    out(i) = fib(10)
#elif CASE == 1
    out(i) = ack(3, 3)
#else
    out(i) = merge(1, 0, is_even(10))
#endif
  end do
  write(*, '(I0)', advance="no") out(1)
contains
  pure recursive function fib(n) result(r)
    integer, intent(in) :: n
    integer :: r
    if (n < 2) then
      r = n
    else
      r = fib(n - 1) + fib(n - 2)
    end if
  end function fib

  pure recursive function ack(m, n) result(r)
    integer, intent(in) :: m, n
    integer :: r
    if (m == 0) then
      r = n + 1
    else if (n == 0) then
      r = ack(m - 1, 1)
    else
      r = ack(m - 1, ack(m, n - 1))
    end if
  end function ack

  pure recursive function is_even(n) result(r)
    integer, intent(in) :: n
    logical :: r
    if (n == 0) then
      r = .true.
    else
      r = is_odd(n - 1)
    end if
  end function is_even
  pure recursive function is_odd(n) result(r)
    integer, intent(in) :: n
    logical :: r
    if (n == 0) then
      r = .false.
    else
      r = is_even(n - 1)
    end if
  end function is_odd
end program test
