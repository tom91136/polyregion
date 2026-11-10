!CHECK case: and-or
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=0 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: eqv-neqv
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=1 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: compare-operands
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=2 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

#ifndef CASE
#define CASE 0
#endif

program test
  implicit none
  integer, parameter :: n = 4096
  integer :: i, bad
  real :: a(n)
  logical :: p(n), q(n)
  integer :: r1(n), r2(n), ref1(n), ref2(n)

  do i = 1, n
    a(i) = real(i)
    p(i) = mod(i, 2) == 1
    q(i) = mod(i, 3) == 0
  end do
  r1 = 0
  r2 = 0
  ref1 = 0
  ref2 = 0

#if CASE == 0
  do concurrent (i = 1:n)
    r1(i) = merge(1, 0, p(i) .and. q(i))
    r2(i) = merge(1, 0, p(i) .or. q(i))
  end do
  do i = 1, n
    ref1(i) = merge(1, 0, p(i) .and. q(i))
    ref2(i) = merge(1, 0, p(i) .or. q(i))
  end do
#elif CASE == 1
  do concurrent (i = 1:n)
    r1(i) = merge(1, 0, p(i) .eqv. q(i))
    r2(i) = merge(1, 0, p(i) .neqv. q(i))
  end do
  do i = 1, n
    ref1(i) = merge(1, 0, p(i) .eqv. q(i))
    ref2(i) = merge(1, 0, p(i) .neqv. q(i))
  end do
#else
  do concurrent (i = 1:n)
    block
      logical :: lo, hi
      lo = a(i) > 100.0
      hi = a(i) > 300.0
      r1(i) = merge(1, 0, lo .and. hi)
      r2(i) = merge(1, 0, lo .or. hi)
    end block
  end do
  do i = 1, n
    ref1(i) = merge(1, 0, (a(i) > 100.0) .and. (a(i) > 300.0))
    ref2(i) = merge(1, 0, (a(i) > 100.0) .or. (a(i) > 300.0))
  end do
#endif
  bad = count(r1 /= ref1) + count(r2 /= ref2)
  write (*, '(I0)', advance='no') bad
end program test
