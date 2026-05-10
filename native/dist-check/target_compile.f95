program test
    implicit none
    integer :: i
    integer :: xs(10)
    do concurrent (i = 1:10)
        xs(i) = i
    end do
end program test
