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
__global__ void unfold_kernel(T * d_in, T * d_out, const size_t In, const size_t mode, std::array<size_t,N> dims, const bool col_maj)
{
    const size_t tid = kernel_utils::tid_1d();
    if (tid < In)
    {
        size_t u_idx[2];
        utils::unfolding_idx(tid, dims, In, mode, u_idx);
        size_t linear_idx = (col_maj) ? u_idx[0] + u_idx[1] * dims[mode] : u_idx[1] + u_idx[0] * (In/dims[mode]);
        d_out[linear_idx] = d_in[tid];
    }

}


template <typename T>
struct max_t
{
    __device__ T operator()(const T& a, const T& b)
    {
        return (a < b) ? b : a;
    }
};



/* NOTE: Assumes that d_data is the unfolding in row major order */
template <size_t Threads, typename T>
__global__ void unfolding_rowmax_kernel(T * d_data, T * d_result, const size_t m, const size_t n)
{
    using BlockReduce = cub::BlockReduce<T, Threads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T curr_max = 0;
    T data;
    for (size_t t = threadIdx.x; t < n; t += blockDim.x)
    {
        data = d_data[blockIdx.x * n + t];
        data = (data < 0) ? -data : data;
        curr_max = max(data, curr_max);
    }

    __syncthreads();

    T global_max = BlockReduce(temp_storage).Reduce(curr_max, max_t<T>{});

    if (threadIdx.x==0)
    {
        d_result[blockIdx.x] = global_max;
    }
}



template <size_t N, typename T>
T * unfolding_rowmax(T * d_data, std::array<size_t, N>& modes, const size_t In, const size_t mode)
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

    utils::write_d_arr(globals::logfile, d_data, In, "Original Tensor");

    unfold_kernel<N><<< std::ceil((double)In / (double)1024), 1024 >>>
        (d_data, d_tmp, In, mode, modes, false);

    CUDA_CHECK(cudaDeviceSynchronize());

    utils::write_d_arr(globals::logfile, d_tmp, In, "Unfolded Tensor");

    unfolding_rowmax_kernel<nthreads><<<nblocks, nthreads>>>
        (d_tmp, d_result, m, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_FREE(d_tmp);

    return d_result;
}


template <size_t N, typename T>
T * unfolding_colmax(T * d_data, std::array<size_t, N>& modes, const size_t In, const size_t mode)
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

    unfold_kernel<N><<< std::ceil((double)In / (double)1024), 1024 >>>
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
