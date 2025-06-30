module sptensor
use utils
implicit none

type sptensor_t

    integer :: nnz
    integer :: N

    integer, pointer :: inds(:, :)
    real(dp), pointer :: vals(:)
    integer, pointer :: modes(:)

end type sptensor_t

contains

    subroutine free_sptensor(X)
        type(sptensor_t) :: X
        deallocate(X%inds)
        deallocate(X%vals)
        deallocate(X%modes)
    end subroutine

end module
