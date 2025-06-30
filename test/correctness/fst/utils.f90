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
                write(fd, *) X(i,j)
            end do
        end do

        close(fd)

    end subroutine

end module utils
