#ifndef GEMM_CUH
#define GEMM_CUH

#include "common.cuh"
#include "utils.cuh"

#include <cutensor.h>

namespace mxt
{

namespace linalg
{

template <typename In, typename Out>
cublasComputeType_t get_compute_type()
{
    if constexpr (std::is_same<In, __half>::value)
    {
        if constexpr (std::is_same<Out, __half>::value)
        {
            return CUBLAS_COMPUTE_16F;
        }
        if constexpr (std::is_same<Out, float>::value)
        {
            return CUBLAS_COMPUTE_32F;
        }
    }
    if constexpr (std::is_same<In, float>::value)
    {
        return CUBLAS_COMPUTE_32F;
    }
    if constexpr (std::is_same<In, double>::value)
    {
        return CUBLAS_COMPUTE_64F;
    }

}


template <typename ValueTypeIn_t, typename ValueTypeOut_t, typename IndexType_t>
void gemm(ValueTypeIn_t * d_Y, ValueTypeIn_t * d_U, ValueTypeOut_t * d_G, const IndexType_t m, const IndexType_t n, const IndexType_t k)
{
    ValueTypeIn_t alpha = 1.0;
    ValueTypeIn_t beta = 0.0;
    CUBLAS_CHECK(cublasGemmEx(globals::cublas_handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha, d_U,
                              utils::to_cuda_dtype<ValueTypeIn_t>(),
                              m,
                              d_Y, utils::to_cuda_dtype<ValueTypeIn_t>(),
                              k,
                              &beta,
                              d_G, utils::to_cuda_dtype<ValueTypeOut_t>(), m,
                              get_compute_type<ValueTypeIn_t, ValueTypeOut_t>(),
                              CUBLAS_GEMM_DEFAULT));
}


} //linalg

} //mxt


#endif
