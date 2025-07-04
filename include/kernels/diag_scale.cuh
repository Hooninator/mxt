#ifndef DIAG_SCALE_CUH
#define DIAG_SCALE_CUH

#include "common.cuh"
#include "kernel_utils.cuh"

namespace mxt
{
namespace kernels
{

template <typename T>
__global__ void scale_diag_inplace_left_kernel(T * d_A, T * d_X, size_t m, size_t n)
{
    const uint32_t tid = kernel_utils::tid_1d();
    const uint32_t wid = tid / warpSize;
    const uint32_t lid = tid % warpSize;

    if (wid < m)
    {
        for (uint32_t i = lid; i < n; i += warpSize)
        {
            d_A[wid * n + i] *= d_X[wid];
        }
    }
}


template <typename T>
void scale_diag_inplace_left(T * d_A, T * d_X, size_t m, size_t n)
{

    const size_t threads = 1024;
    const size_t blocks = std::ceil( (double)m / (double)(threads / 32));

    scale_diag_inplace_left_kernel<T>
        <<<threads, blocks>>>
        (d_A, d_X, m, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}


} //kernels
} //mxt
#endif
