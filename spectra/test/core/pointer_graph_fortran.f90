!CHECK case: pointer_graph
!CHECK using: opt=O0,O3
!CHECK offload-only
!CHECK do: polyfc {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
!CHECK stderr-count: 2 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
!CHECK stderr-count: 2 .core_transform_reduce_w4)
!CHECK stderr-count: 2 .core_transform_reduce_w8)
!CHECK stderr-count: 0 uses the inline body
!CHECK do: {output}
!CHECK requires: 10 19580 10 2061 265998.0 1576408

!CHECK case: pointer_graph-baseline
!CHECK using: opt=O0,O3
!CHECK offload-only
!CHECK do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polyfc {spectra_stdpar} -{opt} -o {output} {input}
!CHECK stderr-count: 0 bound to polyregion.core
!CHECK stderr-count: 0 uses the inline body
!CHECK do: {output}
!CHECK requires: 10 19580 10 2061 265998.0 1576408

program pointer_graph
  implicit none
  type :: pair
    integer :: a
    integer :: b
  end type
  integer, parameter :: n = 1031
  integer :: i, s, lo, hi, t
  integer, allocatable :: xs(:), ys(:)
  real(8), allocatable :: ws(:)
  real(8) :: r
  type(pair) :: p
  p = pair(10, 3)
  allocate(xs(n), ys(n), ws(n))
  do concurrent (i = 1:n)
    xs(i) = i
  end do
  do concurrent (i = 1:n)
    ys(i) = p%a + p%b * mod(xs(i), 7)
    ws(i) = 0.5d0 * xs(i)
  end do
  s = 0
  do concurrent (i = 1:n) reduce(+:s)
    s = s + ys(i)
  end do
  lo = huge(lo)
  hi = 0
  do concurrent (i = 1:n - 1) reduce(min:lo) reduce(max:hi)
    lo = min(lo, ys(i))
    hi = max(hi, xs(i) + xs(i + 1))
  end do
  r = 0d0
  do concurrent (i = 1:n) reduce(+:r)
    r = r + ws(i)
  end do
  t = 0
  do concurrent (i = 1:n) reduce(+:t)
    t = t + xs(i) * p%b - ys(i)
  end do
  write (*, '(I0,1X,I0,1X,I0,1X,I0,1X,F0.1,1X,I0)', advance='no') ys(7), s, lo, hi, r, t
end program
