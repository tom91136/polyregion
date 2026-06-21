!CHECK case: microstream
!CHECK using: num_kind=8,4
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DCHECK_NUM_KIND={num_kind} -o {output} {input}
!CHECK do: {output} 1024 10

#ifndef CHECK_NUM_KIND
#define CHECK_NUM_KIND 8
#endif

module stream
    implicit none

    integer, parameter :: NK = CHECK_NUM_KIND
    real(kind = NK), parameter :: startA = 0.1_NK
    real(kind = NK), parameter :: startB = 0.2_NK
    real(kind = NK), parameter :: startC = 0.0_NK
    real(kind = NK), parameter :: startScalar = 0.4_NK
contains

    ! XXX a,b,c are explicit-shape and not assumed-shape dimension(:) as flang lowers box-array access
    ! non-deterministically, and fails repro check even without polyregion at -O3
    subroutine runAll(a, b, c, scalar, dotSum, arrSize, times, timings)
        implicit none
        integer(kind = 8), intent(in) :: arrSize, times
        real(kind = NK), dimension(arrSize), intent(inout) :: a, b, c
        real(kind = NK), intent(in) :: scalar
        real(kind = NK), intent(inout) :: dotSum
        real(kind = 8), dimension(:, :), allocatable, intent(out) :: timings
        integer(kind = 8) :: k, rate, c1, c2, i

        call system_clock(count_rate = rate)
        allocate(timings(5, times))

        do concurrent (i = 1 : arrSize)
            a(i) = startA
            b(i) = startB
            c(i) = startC
        end do

        do k = 1, times
            call system_clock(count = c1)
            do concurrent (i = 1 : arrSize); c(i) = a(i); end do
            call system_clock(count = c2)
            timings(1, k) = real(c2 - c1, kind = 8) / real(rate, kind = 8)

            call system_clock(count = c1)
            do concurrent (i = 1 : arrSize); b(i) = scalar * c(i); end do
            call system_clock(count = c2)
            timings(2, k) = real(c2 - c1, kind = 8) / real(rate, kind = 8)

            call system_clock(count = c1)
            do concurrent (i = 1 : arrSize); c(i) = a(i) + b(i); end do
            call system_clock(count = c2)
            timings(3, k) = real(c2 - c1, kind = 8) / real(rate, kind = 8)

            call system_clock(count = c1)
            do concurrent (i = 1 : arrSize); a(i) = b(i) + scalar * c(i); end do
            call system_clock(count = c2)
            timings(4, k) = real(c2 - c1, kind = 8) / real(rate, kind = 8)

            call system_clock(count = c1)
            dotSum = 0.0_NK
            do concurrent (i = 1 : arrSize) reduce(+ : dotSum)
                dotSum = dotSum + a(i) * b(i)
            end do
            call system_clock(count = c2)
            timings(5, k) = real(c2 - c1, kind = 8) / real(rate, kind = 8)
        end do
    end subroutine runAll

    subroutine run(arrSize, times)
        implicit none
        integer(kind = 8), intent(in) :: arrSize, times
        real(kind = 8) :: bytes
        real(kind = NK) :: dotSum
        real(kind = NK), dimension(:), allocatable :: a, b, c
        real(kind = 8), dimension(:, :), allocatable :: timings
        integer :: kindSize

        allocate(a(arrSize), b(arrSize), c(arrSize))
        kindSize = storage_size(real(0, kind = NK)) / 8
        bytes = real(arrSize, kind = 8) * real(kindSize, kind = 8)

        write(*, "(a,i0,a)") "Running kernels ", times, " times"
        write(*, "(a,i0)") "Number of elements: ", arrSize
        write(*, "(a,i0,a)") "Precision: real(", NK, ")"
        write(*, "(a,f8.1,a,f8.1,a)") "Array size: ", bytes * 1.0E-6, " MB (=", bytes * 1.0E-9, " GB)"
        write(*, "(a,f8.1,a,f8.1,a)") "Total size: ", 3.0 * bytes * 1.0E-6, " MB (=", 3.0 * bytes * 1.0E-9, " GB)"

        call runAll(a, b, c, startScalar, dotSum, arrSize, times, timings)

        block
            character(20) :: buffer(8)
            integer, parameter :: sizes(5) = [2, 2, 3, 3, 2]
            character(5), parameter :: labels(5) = ["Copy ", "Mul  ", "Add  ", "Triad", "Dot  "]
            integer :: i
            real(kind = 8) :: tmin, tmax, tavg
            write(*, "(a)") "Function    Mbytes/s    Min (sec)   Max         Average"
            do i = 1, 5
                tmin = MINVAL(timings(i, 2:times))
                tmax = MAXVAL(timings(i, 2:times))
                tavg = SUM(timings(i, 2:times)) / (times - 1)
                write(buffer(1), '(a)')     labels(i)
                write(buffer(2), '(f12.3)') 1.0d-6 * (kindSize * REAL(arrSize, kind = 8) * sizes(i)) / tmin
                write(buffer(3), '(f12.5)') tmin
                write(buffer(4), '(f12.5)') tmax
                write(buffer(5), '(f12.5)') tavg
                write(*, '(5a12)') ADJUSTL(buffer(1:5))
            end do
        end block

        block
            real(kind = 8) :: eps, goldA, goldB, goldC, goldSum, errSum
            integer :: i
            logical :: failed
            failed = .false.
            goldA = startA
            goldB = startB
            goldC = startC
            do i = 1, times
                goldC = goldA
                goldB = startScalar * goldC
                goldC = goldA + goldB
                goldA = goldB + startScalar * goldC
            end do
            goldSum = goldA * goldB * arrSize
            eps = real(EPSILON(real(1.0, kind = NK)), kind = 8)
            errSum = ABS((dotSum - goldSum) / goldSum)
            if (errSum > eps * real(arrSize, kind = 8)) then
                write (*, '(a,e12.5)') "Validation failed on sum. Error ", errSum
                write (*, '(a,f12.5,a,f12.5)') "Sum was ", dotSum, " but should be ", goldSum
                failed = .true.
            end if
            call checkErr(a, real(goldA, kind = NK), eps * 100.0_8, "a", failed)
            call checkErr(b, real(goldB, kind = NK), eps * 100.0_8, "b", failed)
            call checkErr(c, real(goldC, kind = NK), eps * 100.0_8, "c", failed)
            if (failed) then
                write (*, '(a)') "Some validations failed!"
                stop 1
            end if
        end block
    end subroutine run

    subroutine checkErr(xs, gold, epsi, name, failed)
        implicit none
        real(kind = NK), dimension(:), intent(in) :: xs
        real(kind = NK), intent(in) :: gold
        real(kind = 8), intent(in) :: epsi
        real(kind = 8) :: acc
        character(*), intent(in) :: name
        logical, intent(inout) :: failed
        integer :: i
        acc = 0.0_8
        do i = 1, SIZE(xs)
            acc = acc + ABS(real(xs(i) - gold, kind = 8))
        end do
        if ((acc / SIZE(xs)) > epsi) then
            write (*, '(a,a,a,e12.5)') "Validation failed on ", name, ". Average error ", acc
            failed = .true.
        end if
    end subroutine checkErr

end module stream

program main
    use stream
    implicit none
    integer(kind = 8) :: argc, asize, times
    character(len = 12), dimension(:), allocatable :: argv

    asize = 33554432_8
    times = 100_8
    argc = command_argument_count()
    if (argc > 0) allocate(character(len = 12) :: argv(argc))
    if (argc >= 1) then
        call get_command_argument(1, argv(1))
        read(argv(1), "(i12)") asize
    end if
    if (argc >= 2) then
        call get_command_argument(2, argv(2))
        read(argv(2), "(i8)") times
    end if
    call run(asize, times)
    write(*, "(a)") "Done"
end program main
