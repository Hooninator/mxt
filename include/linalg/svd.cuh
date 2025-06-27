#ifndef SVD_CUH
#define SVD_CUH

#include "common.cuh"
#include "utils.cuh"
#include "kernels/transpose.cuh"
#include "kernels/strided_copy.cuh"
#include "linalg/geam.cuh"

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
    ValueTypeIn * d_U_tmp;
    CUDA_CHECK(cudaMalloc(&d_U_tmp, sizeof(ValueTypeIn) * n * n));

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
                    m, n, k, 3,
                    Iters, 
                    utils::to_cuda_dtype<ValueTypeIn>(), d_A, m,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_S,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_V_tmp, m,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_U_tmp, n,
                    utils::to_cuda_dtype<ValueTypeIn>(), &d_bytes, 
                    &h_bytes));

    CUDA_CHECK(cudaMalloc(&d_workspace, d_bytes));
    h_workspace = (void*)(new char[h_bytes]);

    CUSOLVER_CHECK(cusolverDnXgesvdr(
                    globals::cusolverdn_handle, params,
                    jobu, jobv,
                    m, n, k, 3,
                    Iters, 
                    utils::to_cuda_dtype<ValueTypeIn>(), d_A, m,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_S,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_V_tmp, m,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_U_tmp, n,
                    utils::to_cuda_dtype<ValueTypeIn>(), d_workspace, d_bytes, 
                    h_workspace, h_bytes, d_info));

    // Convert the output to row-major order and change precisions
    CUDA_CHECK(cudaMemcpy(d_U_lra, d_U_tmp, sizeof(ValueTypeIn) * N * K, cudaMemcpyDeviceToDevice));
    kernels::transpose_outplace<ValueTypeIn, ValueTypeOut, K, N>(d_U_lra, d_U);
    //utils::write_d_arr(globals::logfile, d_U, N * K, "Temporary");

    // If we want the output of the lra, convert the thing we just transposed to the right precision
    if constexpr (KeepLra)
    {
        utils::d_to_u<ValueTypeOut, ValueTypeIn>(d_U, d_U_lra, n * k);
    }

    free(h_workspace);

    CUDA_FREE(d_info);
    CUDA_FREE(d_workspace);
    CUDA_FREE(d_S);
    CUDA_FREE(d_U_tmp);
}


template <typename ValueTypeIn, typename ValueTypeOut, typename IndexType, size_t M, size_t N, size_t K, bool KeepLra>
void llsv_jacobi_cusolver(ValueTypeIn * d_A, ValueTypeOut * d_U, ValueTypeIn * d_U_lra)
{
    // This allows any combination of m and n

    // Since the SpTTMc output is in row-major order but cusolver only accepts column major, 
    // we treat d_A as transposed and compute the right singular vectors, 
    // this should give us the left singular vectors of non-transposed d_A
    size_t m = M;
    size_t n = N;
    size_t k = K;

    //TODO: Move these allocations
    ValueTypeIn * d_S;
    CUDA_CHECK(cudaMalloc(&d_S, sizeof(ValueTypeIn) * std::min(m, n)));

    ValueTypeIn * d_V_tmp; 
    ValueTypeIn * d_U_tmp;
    CUDA_CHECK(cudaMalloc(&d_U_tmp, sizeof(ValueTypeIn) * std::min(m, n) * m));
    CUDA_CHECK(cudaMalloc(&d_V_tmp, sizeof(ValueTypeIn) * std::min(m, n) * n));

    int lwork = 0;
    void * d_workspace = nullptr;
    int * d_info;
    gesvdjInfo_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());

    if constexpr (std::is_same<ValueTypeIn, float>::value)
    {
        CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(
                        globals::cusolverdn_handle, 
                        CUSOLVER_EIG_MODE_VECTOR, 1,
                        m, n, 
                        d_A, m,
                        d_S,
                        d_U_tmp, m,
                        d_V_tmp, n,
                        &lwork,
                        params));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMalloc(&d_workspace, sizeof(float) * lwork));

        CUSOLVER_CHECK(cusolverDnSgesvdj(
                        globals::cusolverdn_handle, 
                        CUSOLVER_EIG_MODE_VECTOR, 1,
                        m, n, 
                        d_A, m,
                        d_S,
                        d_U_tmp, m,
                        d_V_tmp, n,
                        (float *)d_workspace, lwork, d_info, params));
    }
    else if constexpr (std::is_same<ValueTypeIn, double>::value)
    {

        CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
                        globals::cusolverdn_handle, 
                        CUSOLVER_EIG_MODE_VECTOR, 1,
                        m, n, 
                        d_A, m,
                        d_S,
                        d_U_tmp, m,
                        d_V_tmp, n,
                        &lwork,
                        params));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMalloc(&d_workspace, sizeof(double) * lwork));

        CUSOLVER_CHECK(cusolverDnDgesvdj(
                        globals::cusolverdn_handle, 
                        CUSOLVER_EIG_MODE_VECTOR, 1,
                        m, n, 
                        d_A, m,
                        d_S,
                        d_U_tmp, m,
                        d_V_tmp, n,
                        (double *)d_workspace, lwork, d_info, params));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Convert the output to row-major order and change precisions
    CUDA_CHECK(cudaMemcpy(d_U_lra, d_V_tmp, sizeof(ValueTypeIn) * N * K, cudaMemcpyDeviceToDevice));
    kernels::transpose_outplace<ValueTypeIn, ValueTypeOut, K, N>(d_U_lra, d_U);

    // If we want the output of the lra, convert the thing we just transposed to the right precision
    if constexpr (KeepLra)
    {
        utils::d_to_u<ValueTypeOut, ValueTypeIn>(d_U, d_U_lra, N * K);
    }

    CUDA_FREE(d_info);
    CUDA_FREE(d_workspace);
    CUDA_FREE(d_S);
    CUDA_FREE(d_U_tmp);
    CUDA_FREE(d_V_tmp);

    CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(params));
}


template <typename ValueTypeIn, typename ValueTypeCore, typename ValueTypeOut, typename IndexType, size_t M, size_t N, size_t K, bool KeepLra>
void llsv_svd_cusolver(ValueTypeIn * d_A, ValueTypeOut * d_U, ValueTypeCore * d_U_core)
{
    //TODO: if N > M, transpose
    constexpr bool transpose = (N > M);
    signed char jobu = transpose ? 'S' : 'N';
    signed char jobvt = transpose ? 'N' : 'S';

    ValueTypeIn * d_A_active = d_A;
    ValueTypeIn * d_A_t;

    if constexpr (transpose)
    {
        CUDA_CHECK(cudaMalloc(&d_A_t, sizeof(ValueTypeIn) * M * N));
        linalg::transpose<ValueTypeIn, ValueTypeIn>(d_A, d_A_t, M, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        d_A_active = d_A_t;
    }

    utils::write_d_arr(globals::logfile, d_A_active, M * N, "d_A_active");

    size_t m = transpose ? N : M;
    size_t n = transpose ? M : N;

    //TODO: Move these allocations
    ValueTypeIn * d_S;
    CUDA_CHECK(cudaMalloc(&d_S, sizeof(ValueTypeIn) * std::min(m, n)));

    ValueTypeIn * d_V_tmp; 
    ValueTypeIn * d_U_tmp;

    if constexpr (transpose)
    {
        CUDA_CHECK(cudaMalloc(&d_U_tmp, sizeof(ValueTypeIn) * std::min(m, n) * m));
    }
    else
    {
        CUDA_CHECK(cudaMalloc(&d_V_tmp, sizeof(ValueTypeIn) * std::min(m, n) * n));
    }

    cudaDataType dtype = utils::to_cuda_dtype<ValueTypeIn>();

    size_t h_worksize, d_worksize;
    void * d_workspace = nullptr;
    void * h_workspace = nullptr;
    int * d_info;
    cusolverDnParams_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUSOLVER_CHECK(cusolverDnXgesvd_bufferSize(
                    globals::cusolverdn_handle, params,
                    jobu, jobvt,
                    m, n, 
                    dtype, d_A_active, m,
                    dtype, d_S,
                    dtype, d_U_tmp, m,
                    dtype, d_V_tmp, n,
                    dtype, &d_worksize, &h_worksize));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMalloc(&d_workspace, d_worksize));
    h_workspace = malloc(h_worksize);

    CUSOLVER_CHECK(cusolverDnXgesvd(
                    globals::cusolverdn_handle, params,
                    jobu, jobvt,
                    m, n, 
                    dtype, d_A_active, m,
                    dtype, d_S,
                    dtype, d_U_tmp, m,
                    dtype, d_V_tmp, n, dtype,
                    d_workspace, d_worksize,
                    h_workspace, h_worksize, d_info));
    CUDA_CHECK(cudaDeviceSynchronize());

    if constexpr (transpose)
    {
        utils::d_to_u<ValueTypeIn, ValueTypeCore>(d_U_tmp, d_U_core, N * K);
        linalg::transpose<ValueTypeCore, ValueTypeOut>(d_U_core, d_U, N, K);
    }
    else
    {
        kernels::strided_copy<ValueTypeIn, ValueTypeOut, K, N, std::min(M, N)>(d_V_tmp, d_U);
    }

    if constexpr (KeepLra)
    {
        utils::d_to_u<ValueTypeOut, ValueTypeCore>(d_U, d_U_core, N * K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    free(h_workspace);

    CUDA_FREE(d_info);
    CUDA_FREE(d_workspace);
    CUDA_FREE(d_S);
    
    if constexpr (transpose)
    {
        CUDA_FREE(d_U_tmp);
        CUDA_FREE(d_A_t);
    }
    else
    {
        CUDA_FREE(d_V_tmp);
    }

}

} //linalg

} //mxt



#endif
