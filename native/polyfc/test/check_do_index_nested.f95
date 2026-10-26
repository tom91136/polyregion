!CHECK case: alloc2d-inner
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=0 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: explicit2d-inner
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=1 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: alloc2d-block-index
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=2 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

!CHECK case: alloc1d-inner
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCASE=3 -o {output} {input}
!CHECK do: {output}
!CHECK requires: 0

#ifndef CASE
#define CASE 0
#endif
#ifndef N
#define N 1000
#endif
#ifndef M
#define M 50
#endif

program test
  implicit none
  integer, parameter :: n = N, m = M
  integer :: i, j, bad
#if CASE == 1
  integer :: a(n, m), ref(n, m)
#elif CASE == 3
  integer, allocatable :: a(:), ref(:)
#else
  integer, allocatable :: a(:, :), ref(:, :)
#endif

#if CASE == 3
  allocate(a(n), ref(n))
  a = 0
  do i = 1, n
    ref(i) = m * (m + 1) / 2
  end do
  do concurrent (i = 1:n)
    do j = 1, m
      a(i) = a(i) + j
    end do
  end do
  bad = count(a /= ref)
#else
#if CASE != 1
  allocate(a(n, m), ref(n, m))
#endif
  do i = 1, n
    do j = 1, m
      ref(i, j) = 10 * i + j
    end do
  end do
  a = 0
  do concurrent (i = 1:n)
#if CASE == 2
    block
      integer :: j
      do j = 1, m
        a(i, j) = 10 * i + j
      end do
    end block
#else
    do j = 1, m
      a(i, j) = 10 * i + j
    end do
#endif
  end do
  bad = count(a /= ref)
#endif
  write (*, '(I0)', advance='no') bad
end program test
