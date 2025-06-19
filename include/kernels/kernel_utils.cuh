
#ifndef KERNEL_UTILS_CUH 
#define KERNEL_UTILS_CUH 

#include "common.cuh"

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





} //kernel_utils
} //mxt


#endif
