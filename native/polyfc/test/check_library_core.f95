!CHECK case: core-fallback
!CHECK offload-only
!CHECK do: POLYFRONT_LIBRARY_PATH= polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK stderr-count: 0 uses the inline body
!CHECK do: {output}
!CHECK requires: 200 10100 28 50 32 2 5050.0

!CHECK case: core-package
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK stderr-count: 6 bound to polyregion.core
!CHECK do: {output}
!CHECK requires: 200 10100 28 50 32 2 5050.0

!CHECK case: core-package-without-device
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK stderr-count: 6 bound to polyregion.core
!CHECK do: POLYRT_PLATFORM=none POLYRT_HOST_FALLBACK=1 {output}
!CHECK requires: 200 10100 28 50 32 2 5050.0

!CHECK case: core-package-width
!CHECK offload-only
!CHECK do: {package_fixture} --sized-core {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK stderr-count: 5 bound to polyregion.core
!CHECK stderr-count: 1 polyregion.core.transform_reduce uses the inline body: no implementation fits
!CHECK do: {output}
!CHECK requires: 200 10100 28 50 32 2 5050.0

!CHECK case: core-package-character
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHARACTER -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK stderr-count: 1 bound to polyregion.core
!CHECK do: {output}
!CHECK requires: 2

!CHECK case: core-package-assert
!CHECK offload-only
!CHECK do: {package_fixture} {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DASSERT -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK stderr-count: 1 bound to polyregion.core
!CHECK do: {output}
!CHECK requires: 1 7 -1 9 101

!CHECK case: core-device-package
!CHECK offload-only
!CHECK do: {package_fixture} --device-core {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK stderr-count: 6 bound to polyregion.core
!CHECK do: {output}
!CHECK requires: 200 10100 28 50 32 2 5050.0

!CHECK case: core-device-package-allocatable
!CHECK offload-only
!CHECK do: {package_fixture} --device-core {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DALLOCATABLE -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK stderr-count: 4 bound to polyregion.core
!CHECK do: {output}
!CHECK requires: 200 10100 2 199

!CHECK case: core-device-package-recursion
!CHECK offload-only
!CHECK do: {package_fixture} --device-core {output}.packages
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DRECURSION -fstdpar-library-path={output}.packages -o {output} {input}
!CHECK stderr-count: 2 bound to polyregion.core
!CHECK do: {output}
!CHECK requires: 55 108

#if defined(RECURSION)
program check_library_core
  implicit none
  integer, parameter :: n = 16
  integer :: i, s, xs(n)
  do concurrent (i = 1:n)
    xs(i) = fib(i - 1)
  end do
  s = 0
  do concurrent (i = 1:n) reduce(+:s)
    s = s + fib(mod(i, 10))
  end do
  write (*, '(I0,1X,I0)', advance='no') xs(11), s
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
end program
#elif defined(ALLOCATABLE)
program check_library_core
  implicit none
  integer, parameter :: n = 100
  integer :: i, s, lo, hi
  integer, allocatable :: xs(:), ys(:)
  allocate(xs(n), ys(n))
  do concurrent (i = 1:n)
    xs(i) = i
  end do
  do concurrent (i = 1:n)
    ys(i) = xs(i) * 2
  end do
  s = 0
  do concurrent (i = 1:n) reduce(+:s)
    s = s + ys(i)
  end do
  lo = huge(lo)
  hi = 0
  do concurrent (i = 1:n - 1) reduce(min:lo) reduce(max:hi)
    lo = min(lo, ys(i))
    hi = max(hi, xs(i) + ys(i + 1) - xs(i + 1))
  end do
  write (*, '(I0,1X,I0,1X,I0,1X,I0)', advance='no') ys(n), s, lo, hi
end program
#elif defined(CHARACTER)
program check_library_core
  implicit none
  integer, parameter :: n = 4
  integer :: i, a(n)
  character(len = n), parameter :: word = "poly"
  do concurrent (i = 1:n)
    a(i) = merge(i, 0, word(i:i) == "o")
  end do
  write (*, '(I0)', advance='no') a(2)
end program
#elif defined(ASSERT)
program check_library_core
  implicit none
  interface
    pure subroutine fc_assert_msg(message) bind(C, name = "__polyregion_fc_assert_msg")
      character(len = *), intent(in) :: message
    end subroutine
    function fc_assert_raised() bind(C, name = "__polyregion_fc_assert_raised") result(r)
      integer :: r
    end function
  end interface
  integer, parameter :: n = 100
  integer :: i, xs(n)
  character(len = 8) :: message
  message = "core" // char(0)
  do i = 1, n
    xs(i) = i
  end do
  xs(7) = -1
  do concurrent (i = 1:n)
    if (xs(i) < 0) call fc_assert_msg(message)
    xs(i) = xs(i) + 1
  end do
  write (*, '(I0,4(1X,I0))', advance='no') fc_assert_raised(), xs(6), xs(7), xs(8), xs(n)
end program
#else
program check_library_core
  implicit none
  type :: pair
    integer :: a
    integer :: b
  end type
  integer, parameter :: n = 100
  integer :: xs(n), i, s, m, lo
  real(8) :: r
  integer, allocatable :: ys(:)
  type(pair) :: p
  s = 0
  m = 0
  lo = huge(lo)
  r = 0d0
  p = pair(10, 3)
  allocate(ys(n))
  do concurrent (i = 1:n)
    xs(i) = i * 2
  end do
  do concurrent (i = 1:n) reduce(+:s)
    s = s + xs(i)
  end do
  do concurrent (i = 1:n) reduce(min:lo) reduce(+:r)
    lo = min(lo, xs(i))
    r = r + 0.5d0 * xs(i)
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
  write (*, '(I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,F0.1)', advance='no') xs(n), s, m, count(ys == 0), ys(7) + ys(11), lo, r
end program
#endif
