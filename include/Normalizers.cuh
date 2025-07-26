#ifndef NORMALIZERS_CUH
#define NORMALIZERS_CUH

#include "common.cuh"
#include "utils.cuh"
#include "kernels/unfolding_max.cuh"
#include "kernels/normalization.cuh"
#include "linalg/dgmm.cuh"

#include <thrust/transform.h>


namespace mxt
{


template <typename T, size_t NT>
class NormalizerNull
{

public:

    static constexpr size_t N = NT;

    NormalizerNull()
    {}

    void normalize_tensor(T * d_X, T theta, std::array<size_t, N>& modes, const size_t In, const size_t mode)
    {
        //Do nothing
    }


    void normalize_matrix(T * d_U, T theta, const size_t m, const size_t n, const size_t mode)
    {
        //Do nothing
    }


    void recover_tensor(T * d_X, T theta, std::array<size_t, N>& modes, const size_t In, const size_t mode)
    {
        //Do nothing
    }

};


/* X_tilde = RXS --- U_tilde = DUR^-1 */
template <typename T, size_t N>
class NormalizerTwoSided
{
public:


    NormalizerTwoSided():
        R_vec(N), S_vec(N), D_vec(N),
        R_vec_n(N), S_vec_n(N), D_vec_n(N)
    {}


    void normalize_tensor(T * d_X, T theta, std::array<size_t, N>& modes, const size_t In, const size_t mode)
    {
        assert(mode < N);

        size_t R_size = modes[mode];
        size_t S_size = In / R_size;

        /* Now, set S */
        T * d_S = kernels::unfolding_colmax<N>(d_X, modes, In, mode);
        utils::write_d_arr(globals::logfile, d_S, S_size, "d_S_init");

        /* Apply S */
        kernels::tensor_apply_diag_normalization_right<N>(d_X, d_S, modes, In, mode, true);

        /* First, set R */
        T * d_R = kernels::unfolding_rowmax<N>(d_X, modes, In, mode);

        /* Apply R */
        kernels::tensor_apply_diag_normalization_left<N>(d_X, d_R, modes, In, mode, true);

        /* Theta */
        CUBLAS_CHECK(cublasScalEx(globals::cublas_handle,
                                  In, &theta, utils::to_cuda_dtype<T>(),
                                  d_X, utils::to_cuda_dtype<T>(), 1,
                                  (std::is_same<T, double>::value) ? CUDA_R_64F : CUDA_R_32F));

        //utils::write_d_arr(globals::logfile, d_X, In, "Normalized Tensor");
        //utils::write_d_arr(globals::logfile, d_R, R_size, "d_R");
        //utils::write_d_arr(globals::logfile, d_S, S_size, "d_S");

        /* Add matrices to vectors */
        R_vec[mode] = d_R;
        R_vec_n[mode] = R_size;
        S_vec[mode] = d_S;
        S_vec_n[mode] = S_size;
    }


    void normalize_matrix(T * d_U, T theta, const size_t m, const size_t n, const size_t mode)
    {
        assert(mode < N);

        /* First, apply R^-1 -- since the normalization matrices are stored as \|Xi\|_inf, we can just use dgmm */
        linalg::dgmm_inplace(d_U, R_vec[mode], m, n, CUBLAS_SIDE_RIGHT);

        utils::write_d_arr(globals::logfile, d_U, m*n, "Pseudo-Normalized Matrix");

        /* Now, set D */
        std::array<size_t, 2> modes = {m, n};
        T * d_D = kernels::unfolding_rowmax<2>(d_U, modes, m*n, 0);
        utils::write_d_arr(globals::logfile, d_D, m, "d_D");

        invert_t op{};
        auto d_D_ptr = thrust::device_pointer_cast<T>(d_D);
        thrust::transform(d_D_ptr, d_D_ptr + m, d_D_ptr, op);

        /* Apply D */
        linalg::dgmm_inplace(d_U, d_D, m, n, CUBLAS_SIDE_LEFT);

        /* Theta */
        CUBLAS_CHECK(cublasScalEx(globals::cublas_handle,
                                  m*n, &theta, utils::to_cuda_dtype<T>(),
                                  d_U, utils::to_cuda_dtype<T>(), 1,
                                  (std::is_same<T, double>::value) ? CUDA_R_64F : CUDA_R_32F));

        utils::write_d_arr(globals::logfile, d_U, m*n, "Normalized Matrix");

        /* Add matrices to vectors */
        D_vec[mode] = d_D;
        D_vec_n[mode] = m;

    }


    void recover_tensor(T * d_X, T theta, std::array<size_t, N>& modes, const size_t In, const size_t mode)
    {
        assert(mode < N);

        /* Apply D^-1 */
        T * d_D = D_vec[mode];
        kernels::tensor_apply_diag_normalization_left<N>(d_X, d_D, modes, In, mode, true);

        /* Apply S^-1 */
        T * d_S = S_vec[mode];
        kernels::tensor_apply_diag_normalization_right<N>(d_X, d_S, modes, In, mode, false);

        /* Theta */
        theta = 1/(theta*theta); 
        CUBLAS_CHECK(cublasScalEx(globals::cublas_handle,
                                  In, &theta, utils::to_cuda_dtype<T>(),
                                  d_X, utils::to_cuda_dtype<T>(), 1,
                                  (std::is_same<T, double>::value) ? CUDA_R_64F : CUDA_R_32F));

        
    }


    ~NormalizerTwoSided()
    {
        for (size_t i=0; i<N; i++)
        {
            CUDA_FREE(R_vec[i]);
            CUDA_FREE(S_vec[i]);
            CUDA_FREE(D_vec[i]);
        }
    }


    struct invert_t
    {
        __device__ T operator()(T& x)
        {
            return 1/x;
        }
    };

private:
    std::vector<T*> R_vec; //R^-1
    std::vector<T*> S_vec; //S^-1
    std::vector<T*> D_vec; //D

    std::vector<size_t> R_vec_n;
    std::vector<size_t> S_vec_n;
    std::vector<size_t> D_vec_n;

};


} //mxt

#endif
