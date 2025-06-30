module matrix
use utils
implicit none

    type matrix_t
        real(dp), pointer :: data(:, :)
    end type

contains

    function mat_size(A, dim) result(n)
        type(matrix_t), intent(in) :: A
        integer, intent(in) :: dim
        integer :: n
        n = size(A%data, dim)
    end function


    subroutine free_matrix(A)
        type(matrix_t), intent(inout) :: A
        deallocate(A%data)
    end subroutine


end module matrix
