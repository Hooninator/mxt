#ifndef DEVICE_WORKSPACE_CUH
#define DEVICE_WORKSPACE_CUH

#include "common.cuh"
#include "utils.cuh"


namespace mxt
{
template <typename T>
struct DeviceWorkspace
{

    DeviceWorkspace()
    {
        d_data = nullptr;
        n = 0;
    }

    DeviceWorkspace(const size_t _n)
    {
        n = _n;
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * n));
    }


    void zero()
    {
        CUDA_CHECK(cudaMemset(d_data, 0, sizeof(T) * n));
    }


    void alloc(const size_t _n)
    {
        ASSERT( (n==0), "Tried to alloc on a device workspace of size %zu", n);
        n = _n;
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(T) * n));
    }


    void h2d_cpy(T * h_arr, const size_t _n, const size_t offset=0)
    {
        ASSERT( (_n <= (n - offset)), "Tried to copy %zu entries to a device workspace starting at offset %zu with enough space for only %zu entries", _n, offset, n);
        utils::h2d_cpy(d_data + offset, h_arr, _n);
    }


    T * d2h_cpy(const size_t _n, const size_t offset = 0)
    {
        ASSERT( (_n <= (n - offset)), "Tried to copy %zu entries from a device workspace starting at offset %zu with enough space for only %zu entries", _n, offset, n);
        return utils::d2h_cpy(d_data + offset, _n);
    }


    void dump(std::ofstream& ofs, const char * prefix = nullptr)
    {
        if (prefix != nullptr)
        {
            ofs<<prefix<<"\n";
        }
        T * h_data = d2h_cpy(n);
        for (size_t i=0; i<n; i++)
        {
            ofs<<h_data[i]<<"\n";
        }
        std::flush(ofs);
        delete[] h_data;
    }


    ~DeviceWorkspace()
    {
        CUDA_FREE(d_data);
    }


    size_t n;
    T * d_data;

};

} //mxt



#endif
