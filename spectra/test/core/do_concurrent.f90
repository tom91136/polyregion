!CHECK case: do_concurrent
!CHECK using: opt=O0,O3
!CHECK offload-only
!CHECK do: polyfc {spectra_stdpar} -{opt} -fstdpar-library-path={spectra_packages} -fstdpar-library-capabilities={spectra_capability} -o {output} {input}
!CHECK stderr-count: 2 polyregion.core.for_each bound to polyregion.core (spectra.implementation.
!CHECK stderr-count: 3 .core_transform_reduce_w4)
!CHECK stderr-count: 1 .core_transform_reduce_w8)
!CHECK stderr-count: 1 .core_transform_reduce_w16)
!CHECK stderr-count: 1 polyregion.core.transform_reduce uses the inline body: no implementation fits
!CHECK stderr-count: 1 of 72 bytes
!CHECK stderr-count: 1 uses the inline body
!CHECK do: {output}
!CHECK requires: 2062 1063992 0 258.0 1 1031 2060 4120 531996.0 531996.0 1031.0

!CHECK case: do_concurrent-baseline
!CHECK using: opt=O0,O3
!CHECK offload-only
!CHECK do: POLYFRONT_LIBRARY_PATH= POLYFRONT_LIBRARY_CAPABILITIES= polyfc {spectra_stdpar} -{opt} -o {output} {input}
!CHECK stderr-count: 0 bound to polyregion.core
!CHECK stderr-count: 0 uses the inline body
!CHECK do: {output}
!CHECK requires: 2062 1063992 0 258.0 1 1031 2060 4120 531996.0 531996.0 1031.0

program do_concurrent
  implicit none
  integer, parameter :: n = 1031
  integer :: i, xs(n), s, empty, lo, hi, m
  real(8) :: r, half, a1, a2, a3, a4, a5, a6, a7, a8, a9
  integer :: ys(n)
  s = 0
  empty = 0
  lo = huge(lo)
  hi = -huge(hi)
  m = 0
  r = 0d0
  half = 0d0
  a1 = 0d0; a2 = 0d0; a3 = 0d0; a4 = 0d0; a5 = 0d0; a6 = 0d0; a7 = 0d0; a8 = 0d0; a9 = 0d0
  do concurrent (i = 1:n)
    xs(i) = i * 2
  end do
  do concurrent (i = 1:n) reduce(+:s)
    s = s + xs(i)
  end do
  do concurrent (i = 1:0) reduce(+:empty)
    empty = empty + xs(i)
  end do
  do concurrent (i = 1:n) reduce(+:half)
    half = half + 0.25d0 * mod(xs(i), 4)
  end do
  do concurrent (i = 1:n) reduce(+:r) reduce(min:lo) reduce(max:hi)
    r = r + 0.5d0 * xs(i)
    lo = min(lo, i)
    hi = max(hi, i)
  end do
  do concurrent (i = 2:n:2)
    ys(i) = xs(i) * 2
  end do
  do concurrent (i = 2:n:2) reduce(max:m)
    m = max(m, ys(i))
  end do
  do concurrent (i = 1:n) reduce(+:a1, a2, a3, a4, a5, a6, a7, a8, a9)
    a1 = a1 + 0.5d0 * xs(i)
    a2 = a2 + 1
    a3 = a3 + 1
    a4 = a4 + 1
    a5 = a5 + 1
    a6 = a6 + 1
    a7 = a7 + 1
    a8 = a8 + 1
    a9 = a9 + 1
  end do
  write (*, '(I0,1X,I0,1X,I0,1X,F0.1,1X,I0,1X,I0,1X,I0,1X,I0,1X,F0.1,1X,F0.1,1X,F0.1)', advance='no') &
    xs(n), s, empty, half, lo, hi, ys(n - 1) / 2, m, r, a1, a9
end program
