module frostt_io
use utils
use sptensor
implicit none

contains

    function read_frostt(fpath) result(X)
        type(sptensor_t) :: X
        character(256), intent(in) :: fpath

        integer :: fd = 1
        integer :: nnz = 0
        integer :: N, i

        open(fd, file=fpath)

        read(fd, *) N

        X%N = N
        allocate(X%modes(N))

        read(fd, *) X%modes
        read(fd, *) nnz

        X%nnz = nnz
        allocate(X%vals(nnz))
        allocate(X%inds(nnz, N))

        print*, "[Tensor NNZ]: ", nnz
        print*, "[Tensor Modes]: ", X%modes 

        rewind(fd)

        read(fd, *)
        read(fd, *)
        read(fd, *) 
        do i = 1, nnz
            read(fd, *) X%inds(i, :), X%vals(i)
        end do

        close(fd)

    end function


end module frostt_io
