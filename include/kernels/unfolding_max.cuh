#ifndef UNFOLDING_MAX_CUH
#define UNFOLDING_MAX_CUH

#include "kernel_utils.cuh"
#include "common.cuh"

#include "DenseTensor.cuh"

#include <cub/cub.cuh>
#include <thrust/functional.h>

namespace mxt
{
namespace kernels 
{
template <size_t N, typename T>
__global__ void unfold_kernel(T * d_in, T * d_out, const size_t In, const size_t mode, size_t * dims, const bool col_maj)
{
    const size_t tid = kernel_utils::tid_1d();
    if (tid < In)
    {
        size_t linear_idx = tid;
        std::array<size_t, N> midx = utils::multidx_natural<N>(linear_idx, dims);
        size_t row_idx = midx[mode];
        size_t col_idx = 0;
        size_t stride = 1;
        for (int i=0; i<N; i++)
        {
            if (i==In)
                continue;
            col_idx += midx[i]*stride;
            stride += dims[i];
        }
        linear_idx = (col_maj) ? row_idx + col_idx * dims[mode] : col_idx + row_idx * (In/dims[mode]);
        d_out[linear_idx] = d_in[tid];
    }

}



/* NOTE: Assumes that d_data is the unfolding in row major order */
template <size_t Threads, typename T>
__global__ void unfolding_rowmax_kernel(T * d_data, T * d_result, const size_t m, const size_t n)
{
    using BlockReduce = cub::BlockReduce<T, Threads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T global_max = 0;
    T curr_max = 0;
    T data;
    for (size_t t = threadIdx.x; t < n; t += blockDim.x)
    {
        data = d_data[blockIdx.x * n + t];
        curr_max = BlockReduce(temp_storage).Reduce(data, thrust::maximum<T>{});

        if (threadIdx.x==0)
        {
            curr_max = (curr_max < 0) ? -curr_max : curr_max;
            global_max = (curr_max > global_max) ? curr_max : global_max;
        }
    }

    __syncthreads();

    if (threadIdx.x==0)
    {
        d_result[blockIdx.x] = global_max;
    }
}



template <size_t N, typename T>
T * unfolding_rowmax(T * d_data, size_t * modes, const size_t In, const size_t mode)
{

    size_t m = modes[mode];
    size_t n = In / m;

    static constexpr size_t nthreads = 256;
    size_t nblocks = m;

    T * d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T) * m));

    /* TODO: Avoid explicit unfolding */
    T * d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(T) * In));

    unfold_kernel<N><<< ((double)In / (double)1024), 1024 >>>
        (d_data, d_tmp, In, mode, modes, false);

    CUDA_CHECK(cudaDeviceSynchronize());

    unfolding_rowmax_kernel<nthreads><<<nblocks, nthreads>>>
        (d_tmp, d_result, m, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_FREE(d_tmp);

    return d_result;
}


template <size_t N, typename T>
T * unfolding_colmax(T * d_data, size_t * modes, const size_t In, const size_t mode)
{
    size_t m = modes[mode];
    size_t n = In / m;

    static constexpr size_t nthreads = 256;
    size_t nblocks = n;

    T * d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T) * n));

    /* TODO: Avoid explicit unfolding */
    T * d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, sizeof(T) * In));

    unfold_kernel<N><<< ((double)In / (double)1024), 1024 >>>
        (d_data, d_tmp, In, mode, modes, true);

    CUDA_CHECK(cudaDeviceSynchronize());

    unfolding_rowmax_kernel<nthreads><<<nblocks, nthreads>>>
        (d_tmp, d_result, n, m);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_FREE(d_tmp);

    return d_result;
}

} //kernels
} //mxt


#endif
