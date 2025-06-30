module kron
use utils
use matrix
implicit none

contains

    subroutine kron_prod_rows(matrices, Y, N)
        type(matrix_t), intent(in) :: matrices(:)
        real(dp), intent(inout) :: Y(:)
        integer, intent(in) :: N

        select case (N)
            case (3)
                call kron_prod_rows2(matrices, Y)
            case (4)
                call kron_prod_rows3(matrices, Y)
            !case (5)
            !    call kron_prod_rows4(matrices, Y)
        end select

    end subroutine


    subroutine kron_prod_rows2(matrices, Y)
        type(matrix_t), intent(in) :: matrices(:)
        real(dp), intent(inout) :: Y(:)

        integer :: i, j
        integer :: r1, r2

        r1 = mat_size(matrices(1), 2)
        r2 = mat_size(matrices(2), 2)

        do i=1, r1
            do j=1, r2
                Y(j + (i - 1) * r2) = Y(j + (i - 1) * r2) + matrices(1)%data(1, i) * matrices(2)%data(1, j)
            end do
        end do

    end subroutine


    subroutine kron_prod_rows3(matrices, Y)
        type(matrix_t), intent(in) :: matrices(:)
        real(dp), intent(inout) :: Y(:)

        integer :: i, j, k
        integer :: r1, r2, r3

        r1 = mat_size(matrices(1), 2)
        r2 = mat_size(matrices(2), 2)
        r3 = mat_size(matrices(3), 2)

        do i=1, r1
            do j=1, r2
                do k = 1, r3
                    Y(k + (j - 1) * r3 + (i - 1) * r2 * r3) = Y(k + (j - 1) * r3 + (i - 1) * r2 * r3) + matrices(1)%data(1, i) &
                    * matrices(2)%data(1, j) * matrices(3)%data(1, k)
                end do
            end do
        end do
    end subroutine


    !subroutine kron_prod_rows4(matrices, Y)
    !    type(matrix_t), intent(in) :: matrices(:)
    !    real(dp), intent(inout) :: Y(:)
    !end subroutine

end module kron



