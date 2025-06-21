#ifndef SVD_CUH
#define SVD_CUH

#include "common.cuh"
#include "utils.cuh"

namespace mxt
{
namespace linalg
{


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, int P, int Iters>
void llsv_randsvd(cusolverDnHandle_t& handle, ValueTypeIn * d_A, ValueTypeOut * d_U, const IndexType m, const IndexType n, const IndexType k)
{

    //TODO: Move these
    ValueTypeIn * d_S;
    CUDA_CHECK(cudaMalloc(&d_S, sizeof(ValueTypeIn) * m));

    ValueTypeIn * d_U_tmp;
    CUDA_CHECK(cudaMalloc(&d_U, sizeof(ValueTypeIn) * m * k));

    size_t d_bytes = 0;
    size_t h_bytes = 0;

    void * d_workspace = nullptr;
    void * h_workspace = nullptr;

    int * d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
                    handle, nullptr,
                    'S', 'N',
                    m, n, k, P,
                    Iters, 
                    utils::to_cuda_dtype<ValueTypeIn>(), d_A, m,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_S,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_U_tmp, k,
                    utils::to_cuda_dtype<ValueTypeIn>(), nullptr, 0,
                    utils::to_cuda_dtype<ValueTypeIn>(), &d_bytes, 
                    &h_bytes));

    CUDA_CHECK(cudaMalloc(&d_workspace, d_bytes));
    h_workspace = (void*)(new char[h_bytes]);


    CUSOLVER_CHECK(cusolverDnXgesvdr(
                    handle, nullptr,
                    'S', 'N',
                    m, n, k, P,
                    Iters, 
                    utils::to_cuda_dtype<ValueTypeIn>(), d_A, m,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_S,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_U_tmp, k,
                    utils::to_cuda_dtype<ValueTypeIn>(), nullptr, 0,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_workspace, d_bytes, 
                    h_workspace, h_bytes, d_info));

    utils::d_to_u(d_U_tmp, d_U, m * k);

    free(h_workspace);
    CUDA_FREE(d_info);
    CUDA_FREE(d_workspace);
    CUDA_FREE(d_S);
    CUDA_FREE(d_U_tmp);
}




} //linalg

} //mxt



#endif
