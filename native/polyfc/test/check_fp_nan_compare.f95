!CHECK case: nan-relational
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=0 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: nan-unordered
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=1 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

program test
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer, parameter :: n = 8
  integer :: i, bad
  real :: x(n), y(n), nan
  integer :: eq(n), ne(n), lt(n), le(n), gt(n), ge(n), uno(n)
  integer :: req(n), rne(n), rlt(n), rle(n), rgt(n), rge(n), runo(n)

  nan = ieee_value(0.0, ieee_quiet_nan)
  do i = 1, n
    x(i) = real(i)
    y(i) = real(i) + 0.5
  end do

#ifndef CASE
#define CASE 0
#endif

#if CASE == 1
  do concurrent (i = 1:n)
    uno(i) = merge(1, 0, ieee_unordered(x(i), nan))
  end do
  do i = 1, n
    runo(i) = merge(1, 0, ieee_unordered(x(i), nan))
  end do
  bad = count(uno /= runo)
#else
  do concurrent (i = 1:n)
    eq(i) = merge(1, 0, x(i) == nan)
    ne(i) = merge(1, 0, x(i) /= nan)
    lt(i) = merge(1, 0, x(i) < nan)
    le(i) = merge(1, 0, x(i) <= nan)
    gt(i) = merge(1, 0, x(i) > nan)
    ge(i) = merge(1, 0, x(i) >= nan)
  end do
  do i = 1, n
    req(i) = merge(1, 0, x(i) == nan)
    rne(i) = merge(1, 0, x(i) /= nan)
    rlt(i) = merge(1, 0, x(i) < nan)
    rle(i) = merge(1, 0, x(i) <= nan)
    rgt(i) = merge(1, 0, x(i) > nan)
    rge(i) = merge(1, 0, x(i) >= nan)
  end do
  bad = count(eq /= req) + count(ne /= rne) + count(lt /= rlt) &
      + count(le /= rle) + count(gt /= rgt) + count(ge /= rge)
#endif

  write (*, '(I0)', advance='no') bad
end program test
