#ifndef GEMM_CUH
#define GEMM_CUH

#include "common.cuh"
#include "utils.cuh"

namespace mxt
{

namespace linalg
{


template <typename ValueTypeIn_t, typename ValueTypeOut_t, typename IndexType_t>
void gemm(ValueTypeIn_t * d_A, ValueTypeIn_t * d_B, ValueTypeOut_t * d_C, const IndexType_t m, const IndexType_t n, const IndexType_t k, bool transA, bool transB)
{
    ValueTypeIn_t alpha = 1.0;
    ValueTypeIn_t beta = 0.0;
    CUBLAS_CHECK(cublasGemmEx(globals::cublas_handle,
                              (transA) ? CUBLAS_OP_T : CUBLAS_OP_N, 
                              (transB) ? CUBLAS_OP_T : CUBLAS_OP_N,
                              (transA) ? k : m, 
                              (transB) ? k : n, 
                              (transA) ? m : k,
                              &alpha, d_A,
                              utils::to_cuda_dtype<ValueTypeIn_t>(),
                              (transA) ? k : m,
                              d_B, utils::to_cuda_dtype<ValueTypeIn_t>(),
                              (transB) ? n : k,
                              &beta,
                              d_C, utils::to_cuda_dtype<ValueTypeOut_t>(), 
                              (transA) ? k : m,
                              utils::get_compute_type<ValueTypeIn_t, ValueTypeOut_t>(),
                              CUBLAS_GEMM_DEFAULT));
}


} //linalg

} //mxt


#endif
