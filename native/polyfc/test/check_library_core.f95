!CHECK case: core-fallback
!CHECK offload-only
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: 200 10100 28 50 32

!CHECK case: core-package
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK do: {output}
!CHECK requires: 200 10100 28 50 32

program check_library_core
  implicit none
  type :: pair
    integer :: a
    integer :: b
  end type
  integer, parameter :: n = 100
  integer :: xs(n), i, s, m
  integer, allocatable :: ys(:)
  type(pair) :: p
  s = 0
  m = 0
  p = pair(10, 3)
  allocate(ys(n))
  do concurrent (i = 1:n)
    xs(i) = i * 2
  end do
  do concurrent (i = 1:n) reduce(+:s)
    s = s + xs(i)
  end do
  do concurrent (i = 1:n)
    ys(i) = p%a + p%b * mod(i, 7)
  end do
  do concurrent (i = 1:n) reduce(max:m)
    m = max(m, ys(i))
  end do
  do concurrent (i = 2:n:2)
    ys(i) = 0
  end do
  write (*, '(I0,1X,I0,1X,I0,1X,I0,1X,I0)', advance='no') xs(n), s, m, count(ys == 0), ys(7) + ys(11)
end program
