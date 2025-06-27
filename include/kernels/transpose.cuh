#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH

#include "common.cuh"
#include "kernel_utils.cuh"

namespace mxt
{
namespace kernels
{


template<typename T1, typename T2, size_t TileM, size_t TileN>
__global__ void transpose_outplace_kernel(T1 * d_A, T2 * d_A_trans, const size_t M, const size_t N)
{
    __shared__ T1 tile[TileM][TileN]; //TODO: avoid bank conflict?
    
    const uint32_t bx = blockIdx.x;
    const uint32_t by = blockIdx.y;
    const uint32_t j = threadIdx.x + bx * blockDim.x;
    const uint32_t i = threadIdx.y + by * blockDim.y;

    //TODO: Why doesn't this work with shared memory?
    if (j < N && i < M)
    {
        //tile[threadIdx.y][threadIdx.x] = d_A[i * N + j];
    }

    __syncthreads();

    if (i < M && j < N)
    {
        //d_A_trans[i + j * M] = kernel_utils::convert<T1, T2>(tile[threadIdx.y][threadIdx.x]);
        d_A_trans[i + j * M] = kernel_utils::convert<T1, T2>(d_A[i*N + j]);
    }

}


template<typename T1, typename T2>
void transpose_outplace(T1 * d_A, T2 * d_A_trans, const size_t M, const size_t N)
{
    static constexpr size_t threadsx = 32;
    static constexpr size_t threadsy = 32;
    static constexpr size_t tile_size = threadsx * threadsy;
    size_t blocky = std::max((size_t)1, (size_t)std::ceil((double)M / threadsy)); //TODO: Max blocks
    size_t blockx = std::max((size_t)1, (size_t)std::ceil((double)N / threadsx)); //TODO: Max blocks
    static_assert(tile_size * sizeof(T1) <= MAX_SMEM);
    transpose_outplace_kernel<T1, T2, threadsx, threadsy>
        <<<dim3(blockx, blocky), dim3(threadsx, threadsy)>>>
        (d_A, d_A_trans, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} //kernels
} //mxt

#endif
