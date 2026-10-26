!CHECK case: recur-index-direct
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=0 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1 1 2 3 5 8 13 21

!CHECK case: recur-index-shifted
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=1 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0 1 1 2 3 5 8 13

!CHECK case: recur-index-local-copy
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=2 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1 1 2 3 5 8 13 21

!CHECK case: nonrecur-index-direct
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=3 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1 4 9 16 25 36 49 64

!CHECK case: multiblock-index-direct
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=4 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 1 3 6 10 15 21 28 36

#ifndef CASE
#define CASE 0
#endif

program test
  implicit none
  integer, parameter :: n = 8
  integer :: i, out(n)
  out = 0
  do concurrent (i = 1:n)
#if CASE == 0
    out(i) = fib(i)
#elif CASE == 1
    out(i) = fib(i - 1)
#elif CASE == 2
    block
      integer :: k
      k = i
      out(i) = fib(k)
    end block
#elif CASE == 3
    out(i) = sq(i)
#else
    out(i) = tri(i)
#endif
  end do
  write (*, '(8(I0,:,1X))', advance='no') out
contains
  pure recursive function fib(k) result(r)
    integer, intent(in) :: k
    integer :: r
    if (k < 2) then
      r = k
    else
      r = fib(k - 1) + fib(k - 2)
    end if
  end function fib

  pure function tri(x) result(r)
    integer, intent(in) :: x
    integer :: r
    if (x > 0) then
      r = x * (x + 1) / 2
    else
      r = 0
    end if
  end function tri

  pure function sq(x) result(r)
    integer, intent(in) :: x
    integer :: r
    r = x * x
  end function sq
end program test
