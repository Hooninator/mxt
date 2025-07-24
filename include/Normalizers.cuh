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

    template <typename Tensor_t>
    void normalize_tensor(T * d_X, size_t * modes, const size_t In, const size_t mode)
    {
        //Do nothing
    }


    template <typename Matrix_t>
    void normalize_matrix(Matrix_t* U, const size_t m, const size_t n)
    {
        //Do nothing
    }


};


/* X_tilde = RXS --- U_tilde = DUR^-1 */
template <typename T, size_t NT>
class NormalizerTwoSided
{
public:

    static constexpr size_t N = NT;

    NormalizerTwoSided():
        R_vec(N), S_vec(N), D_vec(N),
        R_vec_n(N), S_vec_n(N), D_vec_n(N)
    {}

    void normalize_tensor(T * d_X, size_t * modes, const size_t In, const size_t mode)
    {
        assert(mode < N);

        size_t R_size = modes[mode];
        size_t S_size = In / R_size;

        /* First, set R */
        T * d_R = kernels::unfolding_rowmax<N>(d_X, modes, In, mode);

        /* Apply R */
        kernels::tensor_apply_diag_normalization_left<N>(d_X, d_R, modes, In, mode);

        /* Now, set S */
        T * d_S = kernels::unfolding_colmax<N>(d_X, modes, In, mode);

        /* Apply S */
        kernels::tensor_apply_diag_normalization_right<N>(d_X, d_S, modes, In, mode);

        /* Add matrices to vectors */
        R_vec[mode] = d_R;
        R_vec_n[mode] = R_size;
        S_vec[mode] = d_S;
        S_vec_n[mode] = S_size;
    }


    void normalize_matrix(T * d_U, const size_t m, const size_t n, const size_t mode)
    {
        assert(mode < N);

        /* First, apply R^-1 -- since the normalization matrices are stored as \|Xi\|_inf, we can just use dgmm */
        linalg::dgmm_inplace(d_U, R_vec[mode], m, n, CUBLAS_SIDE_RIGHT);

        /* Now, set D */
        size_t modes[2] = {m, n};
        T * d_D = kernels::unfolding_rowmax<2>(d_U, modes, m*n, 0);

        invert_t op{};
        auto d_D_ptr = thrust::device_pointer_cast<T>(d_D);
        thrust::transform(d_D_ptr, d_D_ptr + m, d_D_ptr, op);

        /* Apply D */
        linalg::dgmm_inplace(d_U, d_D, m, n, CUBLAS_SIDE_LEFT);

        /* Add matrices to vectors */
        D_vec[mode] = d_D;
        D_vec_n[mode] = m;

    }


    void recover_tensor(T * d_X, size_t * modes, const size_t In, const size_t mode)
    {
        assert(mode < N);

        /* Apply D */
        T * d_D = D_vec[mode];
        kernels::tensor_apply_diag_normalization_left<N>(d_X, d_D, modes, In, mode);

        /* Apply S^-1 */
        T * d_S = S_vec[mode];
        invert_t op{};
        auto d_S_ptr = thrust::device_pointer_cast<T>(d_S);
        thrust::transform(d_S_ptr, d_S_ptr + (In / mode), d_S_ptr, op);

        kernels::tensor_apply_diag_normalization_right<N>(d_X, d_S, modes, In, mode);

        thrust::transform(d_S_ptr, d_S_ptr + (In / mode), d_S_ptr, op);
        
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
    std::vector<T*> R_vec; //R
    std::vector<T*> S_vec; //S
    std::vector<T*> D_vec; //D^-1

    std::vector<size_t> R_vec_n;
    std::vector<size_t> S_vec_n;
    std::vector<size_t> D_vec_n;

};


} //mxt

#endif
