!CHECK case: wide-division
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output} 384
!CHECK requires: 0

!CHECK case: wide-division-large-divisor
!CHECK do: polyfc {polyfc_defaults} {polyfc_stdpar} -o {output} {input}
!CHECK do: {output} 6000000000
!CHECK requires: 0

program test
  implicit none
  integer, parameter :: n = 4099
  integer(8) :: d, qs(n), rs(n), qu(n), ru(n), eqs, ers, equ, eru, s, u
  character(len=32) :: text
  integer :: i, k, bad

  call get_command_argument(1, text)
  read (text, *) d
  qs = 0
  rs = 0
  qu = 0
  ru = 0
  do concurrent (i = 1:n)
    block
      integer :: kk
      integer(8) :: uu, ss
      do kk = 1, 6
        uu = (kk - 1_8) * (kk - 1_8) * (kk - 1_8) * 1500000001_8 - 1_8 + i
        ss = merge(-1_8, 1_8, mod(i, 2) == 1) * ((kk - 1_8) * (kk - 1_8) * (kk - 1_8) * 1500000001_8 / 4 + i)
        qu(i) = qu(i) + uu / d
        ru(i) = ru(i) + mod(uu, d)
        qs(i) = qs(i) + ss / d
        rs(i) = rs(i) + mod(ss, d)
      end do
    end block
  end do

  bad = 0
  do i = 1, n
    equ = 0
    eru = 0
    eqs = 0
    ers = 0
    do k = 1, 6
      u = (k - 1_8) * (k - 1_8) * (k - 1_8) * 1500000001_8 - 1_8 + i
      s = merge(-1_8, 1_8, mod(i, 2) == 1) * ((k - 1_8) * (k - 1_8) * (k - 1_8) * 1500000001_8 / 4 + i)
      equ = equ + u / d
      eru = eru + mod(u, d)
      eqs = eqs + s / d
      ers = ers + mod(s, d)
    end do
    if (qu(i) /= equ .or. ru(i) /= eru .or. qs(i) /= eqs .or. rs(i) /= ers) bad = bad + 1
  end do
  write (*, '(I0)', advance='no') bad
end program test
