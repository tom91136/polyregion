!CHECK case: microstream
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -DUSE_DOCONCURRENT -o {output} {input}
!CHECK do: {output} 1024 10

#define DATA_KIND 8
#define DATA_KIND_NAME "real(8)"

module stream
    implicit none

    real(kind = DATA_KIND), parameter :: startA = 0.1
    real(kind = DATA_KIND), parameter :: startB = 0.2
    real(kind = DATA_KIND), parameter :: startC = 0.0
    real(kind = DATA_KIND), parameter :: startScalar = 0.4
contains

    subroutine runAll(a, b, c, h_a, h_b, h_c, initA, initB, initC, scalar, dotSum, arrSize, times, timings)
        implicit none
        real(kind = DATA_KIND), dimension(:), intent(inout) :: a, b, c
        real(kind = DATA_KIND), dimension(:), intent(out) :: h_a, h_b, h_c
        real(kind = DATA_KIND), intent(in) :: initA, initB, initC, scalar
        real(kind = DATA_KIND), intent(inout) :: dotSum
        integer(kind = 8), intent(in) :: arrSize, times
        real(kind = 8), dimension(:, :), allocatable, intent(out) :: timings
        integer(kind = 8) :: k, rate, c1, c2

        call system_clock(count_rate = rate)

        allocate(timings(5, times))

        call init()

        do k = 1, times
            call system_clock(count = c1)
            call copy()
            call system_clock(count = c2)
            timings(1, k) = real(c2 - c1) / real(rate)

            call system_clock(count = c1)
            call mul()
            call system_clock(count = c2)
            timings(2, k) = real(c2 - c1) / real(rate)

            call system_clock(count = c1)
            call add()
            call system_clock(count = c2)
            timings(3, k) = real(c2 - c1) / real(rate)

            call system_clock(count = c1)
            call triad()
            call system_clock(count = c2)
            timings(4, k) = real(c2 - c1) / real(rate)

            call system_clock(count = c1)
            call dot()
            call system_clock(count = c2)
            timings(5, k) = real(c2 - c1) / real(rate)
        end do

        call read()

    contains
        subroutine init()
            integer :: i
#ifdef USE_OMP
            !$omp parallel do
#endif
#ifdef USE_DOCONCURRENT
            do concurrent (i = 1:arrSize)
#else
            do i = 1, arrSize
#endif
                a(i) = initA
                b(i) = initB
                c(i) = initC
            end do
        end subroutine init

        subroutine read()
            integer :: i
#ifdef USE_OMP
            !$omp parallel do
#endif
#ifdef USE_DOCONCURRENT
            do concurrent (i = 1:arrSize)
#else
            do i = 1, arrSize
#endif
                h_a(i) = a(i)
                h_b(i) = b(i)
                h_c(i) = c(i)
            end do
        end subroutine read

        subroutine copy()
            integer :: i
#ifdef USE_OMP
            !$omp parallel do
#endif
#ifdef USE_DOCONCURRENT
            do concurrent (i = 1:arrSize)
#else
            do i = 1, arrSize
#endif
                c(i) = a(i)
            end do
        end subroutine copy

        subroutine mul()
            integer :: i
#ifdef USE_OMP
            !$omp parallel do
#endif
#ifdef USE_DOCONCURRENT
            do concurrent (i = 1:arrSize)
#else
            do i = 1, arrSize
#endif
                b(i) = scalar * c(i)
            end do
        end subroutine mul

        subroutine add()
            integer :: i
#ifdef USE_OMP
            !$omp parallel do
#endif
#ifdef USE_DOCONCURRENT
            do concurrent (i = 1:arrSize)
#else
            do i = 1, arrSize
#endif
                c(i) = a(i) + b(i)
            end do
        end subroutine add

        subroutine triad()
            integer :: i
#ifdef USE_OMP
            !$omp parallel do
#endif
#ifdef USE_DOCONCURRENT
            do concurrent (i = 1:arrSize)
#else
            do i = 1, arrSize
#endif
                a(i) = b(i) + scalar * c(i)
            end do
        end subroutine triad

        subroutine dot()
            integer :: i
            dotSum = 0.0
#ifdef USE_OMP
            !$omp parallel do reduction(+:dotSum)
#endif
#ifdef USE_DOCONCURRENT
            do concurrent (i = 1:arrSize) reduce(+:dotSum)
#else
            do i = 1, arrSize
#endif
                dotSum = dotSum + a(i) * b(i)
            end do
        end subroutine dot
    end subroutine runAll

    subroutine run(arrSize, times)
        implicit none
        integer(kind = 8), intent(in) :: arrSize, times
        real(kind = 8) :: bytes
        real(kind = DATA_KIND) :: dotSum
        real(kind = DATA_KIND), dimension(:), allocatable :: h_a, h_b, h_c, a, b, c
        real(kind = 8), dimension(:, :), allocatable :: timings

        allocate(a(arrSize), b(arrSize), c(arrSize))
        allocate(h_a(arrSize), h_b(arrSize), h_c(arrSize))

        bytes = arrSize * 4.0

#if defined(USE_OMP)
        write(*, "(a)") "Implementation: OpenMP"
#elif defined(USE_DOCONCURRENT)
        write(*, "(a)") "Implementation: DoConcurrent"
#else
            write(*, "(a)") "Implementation: Serial"
#endif

        write(*, "(a,i0,a)") "Running kernels ", times, " times"
        write(*, "(a,i0)") "Number of elements: ", arrSize
        write(*, "(a,a)") "Precision: ", DATA_KIND_NAME

        write(*, "(a,f8.1,a,f8.1,a)") "Array size: ", bytes * 1.0E-6, " MB (=", bytes * 1.0E-9, " GB)"
        write(*, "(a,f8.1,a,f8.1,a)")  "Total size: ", 3.0 * bytes * 1.0E-6, " MB (=", 3.0 * bytes * 1.0E-9, " GB)"

        call runAll(a, b, c, h_a, h_b, h_c, startA, startB, startC, startScalar, dotSum, arrSize, times, timings)

        block
            character(20) :: buffer(8)
            integer, parameter :: sizes(5) = [2, 2, 3, 3, 2]
            character(5), parameter :: labels(5) = ["Copy ", "Mul  ", "Add  ", "Triad", "Dot  "]
            integer :: kindSize = storage_size(real(0, kind = DATA_KIND)) / 8
            integer :: i
            real(kind = 8) :: tmin, tmax, tavg
            write(*, "(a)")   "Function    Mbytes/s    Min (sec)   Max         Average"
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
            enddo
        end block

        block
            real(kind = 8) :: epsi, goldA, goldB, goldC, goldSum, errSum
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
            epsi = EPSILON(REAL(1.0, kind = DATA_KIND)) * 100.0
            errSum = ABS((dotSum - goldSum) / goldSum)
            if (errSum > 1.0E-8) then
                write (*, '(a,f12.5)')  "Validation failed on sum. Error ", errSum
                write (*, '(a,f12.5,a,f12.5)')  "Sum was ", dotSum, " but should be ", goldSum
                failed = .true.
            end if

            call checkErr(h_a, goldA, epsi, "a", failed)
            call checkErr(h_b, goldB, epsi, "b", failed)
            call checkErr(h_c, goldC, epsi, "c", failed)

            if (failed) then
                write (*, '(a)')  "Some validations failed!"
                stop 1
            end if

        end block
    end subroutine run

    subroutine checkErr(xs, gold, epsi, name, failed)
        implicit none
        real(kind = DATA_KIND), dimension(:), intent(in) :: xs
        real(kind = 8), intent(in) :: gold, epsi
        real(kind = 8) :: acc, err
        character(*), intent(in) :: name
        logical, intent(inout) :: failed
        integer :: i
        acc = 0.0
        do i = 1, SIZE(xs)
            acc = acc + ABS(xs(i) - gold)
        end do
        if ((acc / SIZE(xs)) > epsi) then
            write (*, '(a,a,f12.5)') "Validation failed on ", name, ". Average error ", acc
            failed = .true.
        end if
    end subroutine checkErr

end module stream

program main
    use iso_fortran_env
    use stream
    implicit none
    integer(kind = 8) :: argc, size, times
    character(len = 12), dimension(:), allocatable :: argv

    size = 33554432
    times = 100
    argc = command_argument_count()
    if (argc > 0) then
        allocate(character(len = 12) :: argv(argc))
    end if
    if (argc >= 1) then
        call get_command_argument(1, argv(1))
        read(argv(1), "(i12)") size
    endif
    if (argc >= 2) then
        call get_command_argument(2, argv(2))
        read(argv(2), "(i8)") times
    endif
    call run(size, times)
end program main
