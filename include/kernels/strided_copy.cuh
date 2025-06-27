#ifndef STRIDED_COPY_CUH
#define STRIDED_COPY_CUH

#include "common.cuh"
#include "utils.cuh"



namespace mxt
{
namespace kernels
{

template <typename T1, typename T2, size_t N, size_t M, size_t Stride>
__global__ void strided_copy_kernel(T1 * d_arr1, T2 * d_arr2)
{
    for (uint32_t i = threadIdx.x; i < N; i += blockDim.x)
    {
        T1 elem = d_arr1[i + Stride * blockIdx.x];
        d_arr2[i + N * blockIdx.x] = kernel_utils::convert<T1, T2>(elem);
    }
}


template <typename T1, typename T2, size_t N, size_t M, size_t Stride>
void strided_copy(T1 * d_arr1, T2 * d_arr2)
{
    static_assert( Stride >= N );
    const uint32_t nblocks = M;
    const uint32_t nthreads = 128;
    strided_copy_kernel<T1, T2, N, M, Stride>
        <<<nblocks, nthreads>>>
        (d_arr1, d_arr2);
    CUDA_CHECK(cudaDeviceSynchronize());
}
    
} //kernels
} //mxt




#endif
