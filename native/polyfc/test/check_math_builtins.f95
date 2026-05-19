!CHECK case: general
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output}
!CHECK requires: sum=44.000000 abs=16.000000 pow=8.000000

program test
    implicit none
    real, dimension(:), allocatable :: v, negv, powv
    real :: total
    integer :: i

    allocate(v(8))
    allocate(negv(1))
    allocate(powv(1))
    v = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
    negv = [-16.0]
    powv = [0.0]

    do concurrent (i = 1:8)
        v(i) = sqrt(abs(v(i))) + sin(v(i)) ** 2 + cos(v(i)) ** 2
    end do
    total = sum(v)

    do concurrent (i = 1:1)
        negv(i) = abs(negv(i))
    end do

    do concurrent (i = 1:1)
        powv(i) = 2.0 ** 3.0
    end do

    write(*, '("sum=",F0.6," abs=",F0.6," pow=",F0.6)', advance = "no") total, negv(1), powv(1)
end program test
