
#ifndef KERNEL_UTILS_CUH 
#define KERNEL_UTILS_CUH 

#include "common.cuh"


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
