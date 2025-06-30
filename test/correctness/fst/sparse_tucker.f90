module sparse_tucker

use sptensor
use utils
use matrix
use kron
implicit none

contains 

    subroutine sparse_hooi(X, core, factors, ranks, maxiters)
        type(sptensor_t), intent(in) :: X
        integer, intent(in) :: ranks(:)
        real(dp), intent(out), allocatable :: core(:, :)
        type(matrix_t), intent(out), allocatable :: factors(:) 
        integer, intent(in) :: maxiters

        integer :: Rn, nnz, N
        integer :: i,  k

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
            call init_factor(factors(i), modes(i), ranks(i))
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

                !print*, Y_n
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
        integer :: i, j

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


            call kron_prod_rows(U_rows, Y_n(ridx, :), X%N)
        end do

        do j = 1, X%N-1
            call free_matrix(U_rows(j))
        end do

    end subroutine spttmc


    function err(X, core, factors) result(err)

        type(sptensor_t), intent(in) :: X
        real(dp), intent(in) :: core(:,:)
        type(matrix_t), intent(in) :: factors(:)
        real(dp) :: err = 0.0
        



    end function err


end module
