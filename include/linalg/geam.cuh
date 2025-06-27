#ifndef GEAM_CUH
#define GEAM_CUH

#include "common.cuh"
#include "utils.cuh"
#include "kernels/transpose.cuh"

namespace mxt
{
namespace linalg
{

template <typename T, typename U>
void transpose(T * d_in, U * d_out, const size_t m, const size_t n)
{

    T alpha = 1.0;
    T beta = 0.0;

    if constexpr (!std::is_same<T, U>::value)
    {
        kernels::transpose_outplace<T, U>(d_in, d_out, n, m);
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        CUBLAS_CHECK(cublasDgeam(globals::cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, m, &alpha, d_in, m, 
                                    &beta, d_out, n,
                                    d_out, n));
    }
    else if constexpr(std::is_same<T, float>::value)
    {
        CUBLAS_CHECK(cublasSgeam(globals::cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, m, &alpha, d_in, m, 
                                    &beta, d_out, n,
                                    d_out, n));
    }
    else if constexpr(std::is_same<T, __half>::value)
    {
        kernels::transpose_outplace<T, U>(d_in, d_out, n, m);
    }
    else
    {
        NOT_REACHABLE();
    }
}

} //linalg
} //mxt







#endif 
