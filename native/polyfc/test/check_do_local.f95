!CHECK case: real-temp
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=0 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: integer-temp
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=1 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: branch-temp
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=2 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: two-temps-shared-input
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=3 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: temp-across-inner-loop
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=4 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: last-value-after-construct
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=5 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

#ifndef CASE
#define CASE 0
#endif

program test
  implicit none
  integer, parameter :: n = 4000
  integer :: i, j, k, bad, scale
  real :: t, u
  real :: a(n), b(n), ref(n)
  integer :: ia(n), ib(n), iref(n)

  scale = 3
  do i = 1, n
    a(i) = real(i)
    ia(i) = i
  end do
  b = 0
  ib = 0

#if CASE == 0
  do concurrent (i = 1:n)
    t = a(i) * 2
    b(i) = t + 1
  end do
  do i = 1, n
    ref(i) = a(i) * 2 + 1
  end do
  bad = count(b /= ref)
#elif CASE == 1
  do concurrent (i = 1:n)
    k = ia(i) * 2
    ib(i) = k + 1
  end do
  do i = 1, n
    iref(i) = ia(i) * 2 + 1
  end do
  bad = count(ib /= iref)
#elif CASE == 2
  do concurrent (i = 1:n)
    if (mod(i, 2) == 0) then
      t = a(i) * 2
      b(i) = t + 1
    else
      t = a(i) * 3
      b(i) = t - 1
    end if
  end do
  do i = 1, n
    if (mod(i, 2) == 0) then
      ref(i) = a(i) * 2 + 1
    else
      ref(i) = a(i) * 3 - 1
    end if
  end do
  bad = count(b /= ref)
#elif CASE == 3
  do concurrent (i = 1:n)
    t = a(i) * scale
    u = t + a(i)
    b(i) = u * scale
  end do
  do i = 1, n
    ref(i) = (a(i) * 3 + a(i)) * 3
  end do
  bad = count(b /= ref)
#elif CASE == 5
  k = -1
  do concurrent (i = 1:n)
    if (i == n / 2) k = i
  end do
  bad = merge(0, 1, k == n / 2)
#else
  do concurrent (i = 1:n)
    k = 0
    do j = 1, 5
      k = k + ia(i)
    end do
    ib(i) = k
  end do
  do i = 1, n
    iref(i) = 5 * ia(i)
  end do
  bad = count(ib /= iref)
#endif
  write (*, '(I0)', advance='no') bad
end program test
