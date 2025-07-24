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
__global__ void tensor_apply_diag_normalization_left_kernel(T * d_X, T * d_R, size_t * modes, const size_t mode)
{
    size_t idx = kernel_utils::tid_1d();
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

    d_X[idx2] /= d_R[row_idx];
}





template <size_t N, typename T>
void tensor_apply_diag_normalization_left(T * d_X, T * d_R, size_t * modes, const size_t In, const size_t mode)
{
    size_t nthreads = 1024;
    size_t nblocks = std::ceil((double)In / nthreads);
    tensor_apply_diag_normalization_left_kernel<N>
        <<<nthreads, nblocks>>>
        (d_X, d_R, modes, mode);
    CUDA_CHECK(cudaDeviceSynchronize());
}



template <size_t N, typename T>
__global__ void tensor_apply_diag_normalization_right_kernel(T * d_X, T * d_S, const size_t m, const size_t n, const size_t p)
{
    size_t idx = kernel_utils::tid_1d();

    /* Which 3-slice am I in? */
    size_t l = idx / (n * m);

    /* Which row of that 3-slice am I in? */
    size_t i = l % m;
    
    d_X[idx] /= d_S[l * (n * m) + i];
}



template <size_t N, typename T>
void tensor_apply_diag_normalization_right(T * d_X, T * d_S, size_t * modes, const size_t In, const size_t mode)
{
    size_t nthreads = 1024;
    size_t nblocks = std::ceil((double)In / nthreads);

    /* Dimensions of reshaped tensor */
    size_t m = 1;
    size_t p = 1;
    size_t n = modes[mode];

    for (int j=0; j<mode; j++)
    {
        m *= modes[j];
    }

    for (int j=mode + 1; j<N; j++)
    {
        p *= modes[j];
    }

    tensor_apply_diag_normalization_right_kernel<N>
        <<<nthreads, nblocks>>>
        (d_X, d_S, m, n, p);
    CUDA_CHECK(cudaDeviceSynchronize());
}


} //kernels
} //mxt



#endif
