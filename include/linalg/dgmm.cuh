#ifndef DGMM_CUH
#define DGMM_CUH

#include "common.cuh"
#include "kernels/diag_scale.cuh"


namespace mxt
{
namespace linalg
{


template <typename T>
void dgmm_inplace(T * d_A, T * d_X, int m, int n, cublasSideMode_t side)
{

    if constexpr(std::is_same<T, float>::value)
    {
        CUBLAS_CHECK(cublasSdgmm(globals::cublas_handle,
                                 side,
                                 m, n,
                                 d_A, m, 
                                 d_X, 1,
                                 d_A, m));
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        CUBLAS_CHECK(cublasDdgmm(globals::cublas_handle,
                                 side,
                                 m, n,
                                 d_A, m, 
                                 d_X, 1,
                                 d_A, m));
    }
    else if constexpr(std::is_same<T, __half>::value)
    {
        // Swap dims because row major
        kernels::scale_diag_inplace_left(d_A, d_A, n, m);
    }

}



} //linalg
} //mxt










#endif
