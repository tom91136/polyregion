!CHECK case: unlowered-present-in-do-concurrent
!CHECK offload-only
!CHECK compile-fails: unhandled op fir.is_present with used results
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}

module unlowered_present_m
  implicit none
contains
  subroutine add_optional(a, opt)
    integer, intent(inout) :: a(:)
    integer, intent(in), optional :: opt
    integer :: i
    do concurrent(i=1:size(a))
      if (present(opt)) a(i) = a(i) + opt
    end do
  end subroutine add_optional
end module unlowered_present_m

program main
  use unlowered_present_m
  integer :: a(8)
  a = 1
  call add_optional(a, 2)
  print '(I0)', sum(a)
end program main
