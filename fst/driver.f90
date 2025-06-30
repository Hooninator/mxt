program main

    use sparse_tucker
    use frostt_io
    use sptensor
    use utils
    use matrix

    implicit none

    character(256) :: fpath
    character(10) :: r

    integer :: N
    integer :: maxiters = 5
    
    integer :: i

    real(dp) :: stime, etime
    real(dp) :: err

    integer, allocatable :: ranks(:)
    real(dp), allocatable :: core(:, :)
    type(matrix_t), allocatable :: factors(:)

    type(sptensor_t) :: X


    call get_command_argument(1, fpath)
    
    print*, "Reading tensor ",fpath
    X = read_frostt(fpath)
    print*, "Done"

    N = X%N
    allocate(ranks(N))

    do i=1, N
        call get_command_argument(i + 1, r)
        read(r, *) ranks(i)
    end do

    print*, "[Tucker Ranks]: ", ranks

    print*, "Beginning Tucker"
    call cpu_time(stime)
    call sparse_hooi(X, core, factors, ranks, maxiters)
    call cpu_time(etime)
    print*, "Done"
    print*, "Elapsed time: ", etime - stime, "s"

    err = compute_err(X, core, factors, ranks)
    print*, "Reconstruction error: ", err

    call write_tensor(core, ranks, "core_final.tns")

    call free_sptensor(X)

    do i=1, N
        call free_matrix(factors(i))
    end do

end program
