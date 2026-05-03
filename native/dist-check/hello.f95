program hello
    implicit none
    integer :: i, s
    s = 0
    do i = 1, 10
        s = s + i
    end do
    if (s /= 55) call exit(1)
    print *, "sum=", s
end program hello
