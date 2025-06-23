#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH

#include "common.cuh"
#include "kernel_utils.cuh"

namespace mxt
{
namespace kernels
{


template<typename T1, typename T2, size_t M, size_t N, size_t TileM, size_t TileN>
__global__ void transpose_outplace_kernel(T1 * d_A, T2 * d_A_trans)
{
    __shared__ T1 tile[TileM][TileN + 1]; //avoid bank conflict
    
    const uint32_t bx = blockIdx.x;
    const uint32_t by = blockIdx.y;
    const uint32_t j = threadIdx.x + bx * blockDim.x;
    const uint32_t i = threadIdx.y + by * blockDim.y;

    if (j < N && i < M)
    {
        tile[threadIdx.x][threadIdx.y] = d_A[i * N + j];
    }

    __syncthreads();

    if (i < N && j < M)
    {
        d_A_trans[j + i * M] = kernel_utils::convert<T1, T2>(tile[threadIdx.y][threadIdx.x]);
    }

}


template<typename T1, typename T2, size_t M, size_t N>
void transpose_outplace(T1 * d_A, T2 * d_A_trans)
{
    static constexpr size_t threadsx = 32;
    static constexpr size_t threadsy = 32;
    static constexpr size_t tile_size = threadsx * threadsy;
    static constexpr size_t blockx = std::max((size_t)1, (size_t)std::ceil((double)M / tile_size)); //TODO: Max blocks
    static constexpr size_t blocky = std::max((size_t)1, (size_t)std::ceil((double)N / tile_size)); //TODO: Max blocks
    static_assert(tile_size * sizeof(T1) <= MAX_SMEM);
    transpose_outplace_kernel<T1, T2, M, N, threadsx, threadsy>
        <<<dim3(blockx, blocky), dim3(threadsx, threadsy)>>>
        (d_A, d_A_trans);
}

} //kernels
} //mxt

#endif
