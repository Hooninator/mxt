
#ifndef KERNEL_UTILS_CUH 
#define KERNEL_UTILS_CUH 

#include "common.cuh"

#define MAX_SMEM 164 * 1024 //TODO: Condition this on GPU type


#if DEBUG_SPTTMC_KERNEL > 0
#define SPTTMC_PRINT_T0(msg, ...) \
    do { \
        const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; \
        if (tid == 0) \
        { \
            printf(msg "\n", ##__VA_ARGS__); \
        } \
    } while (0);

#define SPTTMC_PRINT_BLOCK(msg, ...) \
    do { \
        if (threadIdx.x == 0) \
        { \
            printf(msg "\n", ##__VA_ARGS__); \
        } \
    } while (0);
#else
#define SPTTMC_PRINT_T0(msg, ...)
#define SPTTMC_PRINT_BLOCK(msg, ...)
#endif

namespace mxt
{
namespace kernel_utils
{

__device__ __forceinline__
uint32_t tid_1d()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}


__device__ __forceinline__
uint8_t lid()
{
    return threadIdx.x % warpSize;
}



template <typename I, size_t Dim, size_t N>
__device__ 
void block_multidx(I (&midx)[N])
{
    uint32_t tid = threadIdx.x;

    #pragma unroll
    for (size_t i = 0; i<N; i++)
    {
        midx[N - i - 1] = tid % Dim;
        tid /= Dim;
    }

}


template <typename I, size_t N>
__device__
void bdims(I (&arr)[N])
{
    unsigned int bdim_total = blockDim.x;
    bdim_total = 8*sizeof(unsigned int) - __clz(bdim_total);
    I val = 2 << ((bdim_total / N) - 1);

    #pragma unroll
    for (size_t i=0; i<N; i++)
    {
        arr[N - i - 1] = val;
    }
}




__device__ __forceinline__
uint8_t wid()
{
    return threadIdx.x / warpSize;
}


template <typename T1, typename T2>
__device__ __forceinline__
T2 convert(T1 x) 
{
    if constexpr(std::is_same<T1, T2>::value)
    {
        return x;
    }
    if constexpr(std::is_same<T1, double>::value)
    {
        if constexpr(std::is_same<T2, half>::value)
        {
            return __double2half(x);
        }
        if constexpr(std::is_same<T2, float>::value)
        {
            return (float)(x);
        }
    }
    if constexpr(std::is_same<T1, float>::value)
    {
        if constexpr(std::is_same<T2, half>::value)
        {
            return __float2half(x);
        }
        if constexpr(std::is_same<T2, double>::value)
        {
            return (double)(x);
        }
    }
    if constexpr(std::is_same<T1, half>::value)
    {
        if constexpr(std::is_same<T2, float>::value)
        {
            return __half2float(x);
        }
        if constexpr(std::is_same<T2, double>::value)
        {
            return (double)(__half2float(x));
        }
    }
}



} //kernel_utils
} //mxt


#endif
