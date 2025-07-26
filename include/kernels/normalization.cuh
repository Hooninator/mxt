#ifndef NORMALIZATION_CUH
#define NORMALIZATION_CUH

#include "common.cuh"
#include "kernel_utils.cuh"

#include <cub/cub.cuh>


namespace mxt
{
namespace kernels
{


template <size_t N, typename T>
__global__ void tensor_apply_diag_normalization_left_kernel(T * d_X, T * d_R, std::array<size_t, N> modes, const size_t In, const size_t mode, bool inv=false)
{
    size_t idx = kernel_utils::tid_1d();

    if (idx < In)
    {
        size_t row_idx;
        size_t idx2 = idx;
        
        for (int i=0; i<N; i++)
        {
            if (i==mode)
            {
                row_idx = idx % modes[i];
                break;
            }
            idx /= modes[i];
        }

        T val = d_R[row_idx];

        d_X[idx2] *= ( (inv) ? 1/val : val);
    }
}





template <size_t N, typename T>
void tensor_apply_diag_normalization_left(T * d_X, T * d_R, std::array<size_t, N>& modes, const size_t In, const size_t mode, bool inv=false)
{
    size_t nthreads = 1024;
    size_t nblocks = std::ceil((double)In / nthreads);
    tensor_apply_diag_normalization_left_kernel<N>
        <<<nthreads, nblocks>>>
        (d_X, d_R, modes, In, mode, inv);
    CUDA_CHECK(cudaDeviceSynchronize());
}


template <size_t N, typename T>
__global__ void tensor_apply_diag_normalization_right_kernel(T * d_X, T * d_S, std::array<size_t, N> dims, const size_t In, const size_t mode, bool inv=false)
{
    size_t idx = kernel_utils::tid_1d();

    if (idx < In)
    {
        size_t u_idx[2];
        utils::unfolding_idx<N>(idx, dims, In, mode, u_idx);
        T val = d_S[u_idx[1]];
        d_X[idx] *= ( (inv) ? 1/val : val);
    }
}


template <size_t N, typename T>
__global__ void tensor_apply_diag_normalization_right_mode0_kernel(T * d_X, T * d_S, const size_t m, const size_t n, bool inv=false)
{
    size_t idx = kernel_utils::tid_1d();
    if (idx < m * n)
    {
        T val = d_S[idx / m];
        d_X[idx] *= ( (inv) ? 1/val : val);
    }
}


template <size_t N, typename T>
void tensor_apply_diag_normalization_right(T * d_X, T * d_S, std::array<size_t, N>& modes, const size_t In, const size_t mode, bool inv=false)
{
    size_t nthreads = 1024;
    size_t nblocks = std::ceil((double)In / nthreads);

    if (mode == 0)
    {
        tensor_apply_diag_normalization_right_mode0_kernel<N>
            <<<nthreads, nblocks>>>
            (d_X, d_S, modes[mode], In / modes[mode], inv);
    }
    else
    {
        /* Dimensions of reshaped tensor */
        size_t m = 1;
        size_t p = 1;
        size_t n = modes[mode];
        for (int j=0; j < mode; j++)
        {
            m *= modes[j];
        }

        for (int j=mode + 1; j<N; j++)
        {
            p *= modes[j];
        }
        tensor_apply_diag_normalization_right_kernel<N>
            <<<nthreads, nblocks>>>
            (d_X, d_S, modes, In, mode, inv);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}


} //kernels
} //mxt



#endif
