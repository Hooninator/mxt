#ifndef SVD_CUH
#define SVD_CUH

#include "common.cuh"
#include "utils.cuh"
#include "kernels/transpose.cuh"

namespace mxt
{
namespace linalg
{


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, size_t Iters, size_t M, size_t N, size_t K, bool KeepLra>
void llsv_randsvd_cusolver(ValueTypeIn * d_A, ValueTypeOut * d_U, ValueTypeIn * d_U_lra)
{
    // Since the SpTTMc output is in row-major order but cusolver only accepts column major, 
    // we treat d_A as transposed and compute the right singular vectors, 
    // this should give us the left singular vectors of non-transposed d_A
    size_t m = M;
    size_t n = N;
    size_t k = K;

    //TODO: Move these allocations
    ValueTypeIn * d_S;
    CUDA_CHECK(cudaMalloc(&d_S, sizeof(ValueTypeIn) * std::min(m, n)));

    ValueTypeIn * d_V_tmp; // I think this can remain uninitialized, since we set jobu='N'

    signed char jobu = 'N';
    signed char jobv = 'S';

    size_t d_bytes = 0;
    size_t h_bytes = 0;

    void * d_workspace = nullptr;
    void * h_workspace = nullptr;

    int * d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    cusolverDnParams_t params = NULL;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
                    globals::cusolverdn_handle, params,
                    jobu, jobv,
                    n, m, k, 2,
                    Iters, 
                    utils::to_cuda_dtype<ValueTypeIn>(), d_A, n,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_S,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_V_tmp, n,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_U_lra, m,
                    utils::to_cuda_dtype<ValueTypeIn>(), &d_bytes, 
                    &h_bytes));

    CUDA_CHECK(cudaMalloc(&d_workspace, d_bytes));
    h_workspace = (void*)(new char[h_bytes]);

    CUSOLVER_CHECK(cusolverDnXgesvdr(
                    globals::cusolverdn_handle, params,
                    jobu, jobv,
                    n, m, k, 2,
                    Iters, 
                    utils::to_cuda_dtype<ValueTypeIn>(), d_A, n,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_S,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_V_tmp, n,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_U_lra, m,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_workspace, d_bytes, 
                    h_workspace, h_bytes, d_info));

    // Convert the output to row-major order and change precisions
    kernels::transpose_outplace<ValueTypeIn, ValueTypeOut, M, K>(d_U_lra, d_U);
    CUDA_CHECK(cudaDeviceSynchronize());

    // If we want the output of the lra, convert the thing we just transposed to the right precision
    if constexpr (KeepLra)
    {
        utils::d_to_u<ValueTypeOut, ValueTypeIn>(d_U, d_U_lra, m * k);
    }

    free(h_workspace);

    CUDA_FREE(d_info);
    CUDA_FREE(d_workspace);
    CUDA_FREE(d_S);
}




} //linalg

} //mxt



#endif
