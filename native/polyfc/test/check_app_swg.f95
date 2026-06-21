!CHECK case: swg
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output} 128

module swg
    implicit none

    integer, parameter :: MAXLEN = 64
    integer, parameter :: MATCH = 1
    integer, parameter :: MISMATCH = -2
    integer, parameter :: GAP = -1
    character(len = *), parameter :: ALPHABET = "ACGT"

contains

    pure subroutine swg(query, qlen, target, tlen, score)
        character(len = MAXLEN), intent(in) :: query, target
        integer, intent(in) :: qlen, tlen
        real(kind = 8), intent(out) :: score
        integer :: v0(MAXLEN), v1(MAXLEN)
        integer :: m, j, ii

        m = max(0, max(GAP, merge(MATCH, MISMATCH, query(1:1) == target(1:1))))
        v0(1) = m
        do j = 2, tlen
            v0(j) = max(0, max(v0(j - 1) + GAP, merge(MATCH, MISMATCH, query(1:1) == target(j:j))))
        end do
        do j = 2, tlen
            m = max(m, v0(j))
        end do

        do ii = 2, qlen
            v1(1) = max(0, max(v0(1) + GAP, merge(MATCH, MISMATCH, query(ii:ii) == target(1:1))))
            m = max(m, v1(1))
            do j = 2, tlen
                v1(j) = max(max(0, v0(j) + GAP), &
                            max(v1(j - 1) + GAP, v0(j - 1) + merge(MATCH, MISMATCH, query(ii:ii) == target(j:j))))
            end do
            do j = 2, tlen
                m = max(m, v1(j))
            end do
            do j = 1, tlen
                v0(j) = v1(j)
            end do
        end do

        score = real(m, kind = 8) / real(min(qlen, tlen) * max(MATCH, GAP), kind = 8)
    end subroutine swg

    subroutine makeDatabase(n, db, lens)
        implicit none
        integer, intent(in) :: n
        character(len = MAXLEN), intent(out) :: db(n)
        integer, intent(out) :: lens(n)
        integer :: i, j, slen, a
        integer(kind = 8) :: s

        do i = 1, n
            db(i) = repeat('A', MAXLEN)
            s = mod(int(i - 1, kind = 8) * 2654435761_8 + 12345_8, 2147483648_8)
            s = mod(s * 1103515245_8 + 12345_8, 2147483648_8)
            slen = 8 + int(mod(s / 65536_8, int(MAXLEN - 8 + 1, kind = 8)))
            do j = 1, slen
                s = mod(s * 1103515245_8 + 12345_8, 2147483648_8)
                a = int(mod(s / 256_8, 4_8)) + 1
                db(i)(j:j) = ALPHABET(a:a)
            end do
            lens(i) = slen
        end do
    end subroutine makeDatabase

end module swg

program main
    use swg
    implicit none
    character(len = MAXLEN), allocatable :: db(:)
    character(len = MAXLEN) :: needle
    integer, allocatable :: lens(:)
    real(kind = 8), allocatable :: scores(:)
    real(kind = 8) :: total, err
    integer :: argc, n, i, nlen
    character(len = 12) :: arg
    logical :: ok

    n = 8192
    argc = command_argument_count()
    if (argc >= 1) then
        call get_command_argument(1, arg)
        read(arg, "(i12)") n
    end if

    needle = "GATTACAGATTACA"
    nlen = 14

    allocate(db(n), lens(n), scores(n))
    call makeDatabase(n, db, lens)

    do concurrent (i = 1 : n)
        call swg(needle, nlen, db(i), lens(i), scores(i))
    end do

    err = 0.0d0
    block
        real(kind = 8) :: ref
        do i = 1, n
            call swg(needle, nlen, db(i), lens(i), ref)
            err = err + abs(scores(i) - ref)
        end do
    end block
    total = sum(scores)

    ok = err / n <= 1.0d-9
    if (.not. ok) write(*, '(a,e12.5)') "Validation failed, average error ", err / n

    write(*, '(a,i0)') "Entries: ", n
    write(*, '(a,f14.6)') "Total similarity: ", total
    write(*, '(a)') "Done"
    if (.not. ok) stop 1
end program main
