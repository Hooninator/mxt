#ifndef DEVICE_WORKSPACE_CUH
#define DEVICE_WORKSPACE_CUH

#include "common.cuh"
#include "utils.cuh"


template <typename T>
struct DeviceWorkspace
{

    DeviceWorkspace(const size_t _n)
    {
        n = _n;
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * n));
    }


    void zero()
    {
        CUDA_CHECK(cudaMemset(d_data, 0, sizeof(T) * n));
    }
    

    ~DeviceWorkspace()
    {
        CUDA_FREE(d_data);
    }

    size_t n;
    T * d_data;

};




#endif
