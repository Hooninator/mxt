module utils
use, intrinsic :: iso_fortran_env, dp=>real64
implicit none
contains

    subroutine print_mat(X)
        real(dp), intent(in) :: X(:,:)
        integer :: i, j


        do j=1, size(X, 2)*15
            write(*, '(A)', advance='no') "-"
        end do

        do i = 1, size(X, 1)
            print *
            do j = 1, size(X, 2)
                write(*, "(A)", advance='no') "|"
                write(*, '(F12.6)', advance='no') X(i,j)
                write(*, "(A)", advance='no') "|"
            end do
            print *
            do j=1, size(X, 2)*15
                write(*, '(A)', advance='no') "-"
            end do
        end do
        print *

    end subroutine


    subroutine write_mat(X, path)
        real(dp), intent(in) :: X(:,:)
        character(256), intent(in) :: path

        integer :: i, j

        integer :: fd = 1

        open(fd, file=path)

        do i = 1, size(X,1)
            do j = 1, size(X,2)
                write(fd, '(I0,1X,I0,1X,SP, ES13.6)') i, j, X(i,j)
            end do
        end do

        close(fd)

    end subroutine


    subroutine write_tensor(X, ranks, path)
        real(dp), intent(in) :: X(:,:)
        integer, intent(in) :: ranks(:)
        character(256), intent(in) :: path

        integer :: i, j, k
        integer :: N, idx
        integer, allocatable :: ind(:)

        integer :: fd = 1

        N = size(ranks, 1)
        allocate(ind(N))

        open(fd, file=path)

        do i = 1, size(X,1)
            ind(N) = i
            do j = 1, size(X, 2)
                idx = j
                do k = N - 1, 1, -1
                    ind(k) = mod(idx, ranks(k)) + 1
                    idx = idx / ranks(k)
                end do
                do k = 1, N
                    write(fd, '(I0, 1X)', advance="no") ind(k)
                end do
                write(fd, "(SP, ES13.6)") X(i,j)
            end do
        end do

        close(fd)

    end subroutine

end module utils
