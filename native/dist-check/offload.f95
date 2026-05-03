program test
    implicit none
    integer :: i, x
    x = 0
    do concurrent (i = 1:10) reduce(+:x)
        x = x + i
    end do
    if (x /= 55) call exit(1)
    print *, "sum=", x
end program test
