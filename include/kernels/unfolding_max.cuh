#ifndef UNFOLDING_MAX_CUH
#define UNFOLDING_MAX_CUH

#include "kernel_utils.cuh"
#include "common.cuh"

#include "DenseTensor.cuh"

#include <cub/cub.cuh>



template <typename T, size_t Threads>
__global__ void unfolding_rowmax_kernel(T * d_data, T * d_result, const size_t m, const size_t n)
{
    using BlockReduce = cub::BlockReduce<T, Threads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T max = 0;
    T data;
    for (size_t t = threadIdx.x; t < n; t += blockDim.x)
    {

    }
}


template <typename Tensor_t>
Tensor_t::ValueType_t * unfolding_rowmax(Tensor_t& X, const size_t mode)
{

    using T = Tensor_t::ValueType_t;

    T * d_data = X.d_data;
    size_t m = Tensor_t::Modes[mode];
    size_t n = Tensor_t::In / m;

    static constexpr size_t nthreads = 256;
    size_t nblocks = m;

    T * d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T) * m));

    unfolding_rowmax_kernel<T, nthreads><<<nblocks, nthreads>>>
        (d_data, d_result, m, n);

}



#endif
