#ifndef NORMS_CUH
#define NORMS_CUH


#include "utils.cuh"
#include "common.cuh"


namespace mxt
{

namespace linalg
{


template <typename T>
T relative_frob_norm(T * d_correct, T * d_computed, const size_t n)
{

    /* ||d_correct|| */
    T nrm_correct;
    CUBLAS_CHECK(cublasNrm2Ex(globals::cublas_handle,
                              n, d_correct, utils::to_cuda_dtype<T>(),
                              1, 
                              &nrm_correct, utils::to_cuda_dtype<T>(),
                              ((std::is_same<T, __half>::value) ? CUDA_R_32F : utils::to_cuda_dtype<T>())
                              ));


    /*  d_correct - d_computed, overwrites d_correct */
    if constexpr(std::is_same<T, __half>::value)
    {
        const float alpha = -1.0;
        CUBLAS_CHECK(cublasAxpyEx(globals::cublas_handle,
                                  n, &alpha, 
                                  CUDA_R_32F,
                                  d_computed,
                                  utils::to_cuda_dtype<T>(),
                                  1,
                                  d_correct,
                                  utils::to_cuda_dtype<T>(),
                                  1,
                                  CUDA_R_32F));
    }
    else 
    {
        const T alpha = -1.0;
        CUBLAS_CHECK(cublasAxpyEx(globals::cublas_handle,
                                  n, &alpha, 
                                  utils::to_cuda_dtype<T>(),
                                  d_computed,
                                  utils::to_cuda_dtype<T>(),
                                  1,
                                  d_correct,
                                  utils::to_cuda_dtype<T>(),
                                  1,
                                  utils::to_cuda_dtype<T>()
                                  ));
    }


    /* ||d_correct - d_computed|| */
    T nrm_diff;
    CUBLAS_CHECK(cublasNrm2Ex(globals::cublas_handle,
                              n, d_correct, utils::to_cuda_dtype<T>(),
                              1, 
                              &nrm_diff, utils::to_cuda_dtype<T>(),
                              ((std::is_same<T, __half>::value) ? CUDA_R_32F : utils::to_cuda_dtype<T>())));

    return nrm_diff / nrm_correct;
}


}

}






#endif
