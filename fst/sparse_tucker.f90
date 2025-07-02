module sparse_tucker

use sptensor
use utils
use matrix
use kron
use omp_lib
implicit none

contains 

    subroutine sparse_hooi(X, core, factors, ranks, maxiters)
        type(sptensor_t), intent(in) :: X
        integer, intent(in) :: ranks(:)
        real(dp), intent(out), allocatable :: core(:, :)
        type(matrix_t), intent(out), allocatable :: factors(:) 
        integer, intent(in) :: maxiters

        integer :: Rn, nnz, N
        integer :: i,  k, iter

        character(256) :: factor_name
        character(7) :: factor_ = "factor"
        character(9) :: tns = "iter0.tns"

        integer :: lwork, info
        real(dp), pointer :: work(:)
        real(dp), pointer :: S(:)
        real(dp), pointer :: U(:, :)
        real(dp), allocatable :: Vt(:,:)

        real(dp) :: alpha = 1.0
        real(dp) :: beta = 0.0

        real(dp), pointer :: vals(:)
        integer, pointer :: inds(:, :)
        integer, pointer :: modes(:)

        real(dp), pointer :: Y_n(:, :)
        real(dp), pointer :: Y_n2(:, :)

        vals => X%vals
        inds => X%inds
        modes => X%modes
        N = X%N
        nnz = X%nnz

        Rn = product(ranks)
        allocate(core(ranks(N), Rn / ranks(N)))

        allocate(factors(N))
        allocate(Vt(1, 1))

        ! Init factors
        do i = 1, N
            iter = i
            call init_factor(factors(i), modes(i), ranks(i))
            !write(factor_name, '(A,"_",I0,"_",A)') trim(factor_), iter-1 ,trim(tns)
            !call write_mat(factors(i)%data, factor_name)
        end do

        ! Main Loop
        do i = 1, maxiters
            print*, "Iteration: ", i
            do k = 1, N
                
                ! SpTTMC
                allocate(Y_n(modes(k), Rn / ranks(k)))
                Y_n = 0

                call spttmc(X, factors, k, ranks, Y_n)

                ! SVD
                if (k == N) then
                    allocate(Y_n2(size(Y_n, 1), size(Y_n, 2)))
                    Y_n2 = Y_n
                end if

                allocate(S(min(modes(k), Rn / ranks(k))))
                allocate(U(modes(k), min(modes(k), Rn / ranks(k))))

                lwork = -1
                allocate(work(1))
                call DGESVD('S', 'N', modes(k), Rn / ranks(k), Y_n, modes(k), &
                            S, U, modes(k), Vt, 1, work, lwork, info)
                lwork = int(work(1))

                deallocate(work)
                allocate(work(lwork))

                call DGESVD('S', 'N', modes(k), Rn / ranks(k), Y_n, modes(k), &
                            S, U, modes(k), Vt, 1, work, lwork, info)

                factors(k)%data = U(:, 1:ranks(k))

                deallocate(work)
                deallocate(S)
                deallocate(U)
                deallocate(Y_n)
            end do

            ! make the core
            call DGEMM('T', 'N', ranks(N), Rn / ranks(N), modes(N), alpha, factors(N)%data, modes(N), &
                        Y_n2, modes(N), beta, core, ranks(N))
            deallocate(Y_n2)
        end do
    end subroutine


    subroutine init_factor(U, m, n)
        type(matrix_t), intent(out) :: U
        integer, intent(in) :: m, n
        integer :: i, j, sz
        integer, allocatable :: seed(:)

        call random_seed(size = sz)
        allocate(seed(sz))
        do i=1,sz
            seed(i)=1
        end do
        call random_seed(put=seed)

        allocate(U%data(m,n))
        
        do i = 1, m
            do j = 1, n
                call random_number(U%data(i,j))
            end do
        end do
    end subroutine


    subroutine spttmc(X, factors, k, ranks, Y_n)
        type(sptensor_t), intent(in) :: X
        type(matrix_t), intent(in) :: factors(:)
        integer, intent(in) :: k
        integer, intent(in) :: ranks(:)
        real(dp), intent(inout) :: Y_n(:,:)

        integer :: Rn, I

        type(matrix_t), allocatable :: U_rows(:)

        integer :: j, l, h
        real(dp) :: val
        integer, allocatable :: ind(:)
        integer :: ridx

        I = X%modes(k)
        Rn = product(ranks) / ranks(k)

        allocate(ind(I))
        allocate(U_rows(X%N - 1))

        l = 1
        do j = 1, X%N
            if (j /= k) then
                allocate(U_rows(l)%data(1, ranks(j)))
                l = l + 1
            end if
        end do


        do j = 1, X%nnz
            val = X%vals(j)
            ind = X%inds(j, :)
            ridx = ind(k)

            h = 1
            do l = 1, X%N
                if (l /= k) then 
                    U_rows(h)%data(1, :) = factors(l)%data(ind(l), :)
                    h = h + 1
                end if
            end do

            call kron_prod_rows(U_rows, Y_n(ridx, :), val, X%N)

        end do

        do j = 1, X%N-1
            call free_matrix(U_rows(j))
        end do

    end subroutine spttmc


    function compute_err(X, core, factors, ranks) result(err)

        type(sptensor_t), intent(in) :: X
        real(dp), intent(in) :: core(:,:)
        type(matrix_t), intent(in) :: factors(:)
        integer, intent(in) :: ranks(:)
        real(dp) :: err

        real(dp) :: val, val_approx, val_tmp
        integer, allocatable :: ind(:)
        integer, allocatable :: r_ind(:)
        real(dp), allocatable :: core_f(:)

        integer :: i, j, k, l
        integer :: Rn, idx, nnz

        allocate(ind(X%N))
        allocate(r_ind(X%N))

        Rn = product(ranks)
        nnz = X%nnz

        err = 0.0

        !$OMP PARALLEL DEFAULT(SHARED) FIRSTPRIVATE(nnz, Rn, val_approx, val, ind, r_ind, idx, val_tmp)
        !$OMP DO REDUCTION(+:err)
        do i=1, nnz
            val_approx = 0.0
            val = X%vals(i)
            ind = X%inds(i, :)

            if (mod(i, 1000)==0.and.OMP_get_thread_num()==0) then
                print*, i, " values reconstructed..."
            end if

            do j=1, ranks(X%N)
                r_ind(X%N) = j
                do l = 1, Rn / ranks(X%N)
                    val_tmp = 1
                    idx = l - 1
                    do k = X%N - 1, 1, -1
                        r_ind(k) = mod(idx, ranks(k)) + 1
                        idx = idx / ranks(k)
                    end do
                    do k = 1, X%N
                        val_tmp = val_tmp * factors(k)%data(ind(k), r_ind(k))
                    end do
                    val_approx = val_approx + val_tmp * core(j, l)
                end do
            end do

            val_approx = (val_approx - val)**2

            err = err + val_approx
        end do
        !$OMP END DO
        !$OMP END PARALLEL

        err = sqrt(err) / norm2(X%vals)

    end function compute_err


end module
