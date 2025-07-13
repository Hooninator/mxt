#ifndef TTMC_CUH
#define TTMC_CUH

#include "common.cuh"
#include "utils.cuh"

#include "DenseTensor.cuh"
#include "MatrixCollection.cuh"


namespace mxt
{

template <typename InputType1, typename InputType2, typename OutputType>
void ttm_modek(InputType1 * d_X, InputType2 * d_U, OutputType * d_Y, 
                const size_t m, const size_t n, const size_t k,
                const size_t p, const size_t i,
                cublasComputeType_t compute_type)
{
    cudaDataType IT1 = utils::to_cuda_dtype<InputType1>();
    cudaDataType IT2 = utils::to_cuda_dtype<InputType2>();
    cudaDataType OT = utils::to_cuda_dtype<OutputType>();


    InputType1 alpha = 1.0;
    OutputType beta = 0.0;


    /* Setup arrays of matrices */
    void ** h_X_arr = new void * [p];
    void ** h_U_arr = new void * [p];
    void ** h_Y_arr = new void * [p];


    for (size_t i=0; i<p; i++)
    {
        h_X_arr[i] = (void *) (d_X + (i * m * n));
        h_U_arr[i] = (void *) (d_U);
        h_Y_arr[i] = (void *) (d_Y + (i * m * k));
    }

    void ** d_X_arr = utils::h2d_cpy(h_X_arr, p);
    void ** d_U_arr = utils::h2d_cpy(h_U_arr, p);
    void ** d_Y_arr = utils::h2d_cpy(h_Y_arr, p);

    CUBLAS_CHECK(cublasGemmBatchedEx(globals::cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_T,
                                     m, k, n,
                                     &alpha,
                                     d_X_arr, IT1, m,
                                     d_U_arr, IT2, k,
                                     &beta, d_Y_arr, OT,
                                     m, p, compute_type,
                                     CUBLAS_GEMM_DEFAULT));


    CUDA_FREE(d_X_arr);
    CUDA_FREE(d_U_arr);
    CUDA_FREE(d_Y_arr);

    delete[] h_X_arr;
    delete[] h_U_arr;
    delete[] h_Y_arr;
}


template <typename InputType1, typename InputType2, typename OutputType>
void ttm_mode1(InputType1 * d_X, InputType2 * d_U, OutputType * d_Y, 
                const size_t m, const size_t n, const size_t k,
                cublasComputeType_t compute_type)
{

    cudaDataType IT1 = utils::to_cuda_dtype<InputType1>();
    cudaDataType IT2 = utils::to_cuda_dtype<InputType2>();
    cudaDataType OT = utils::to_cuda_dtype<OutputType>();

    InputType1 alpha = 1.0;
    OutputType beta = 0.0;

    CUBLAS_CHECK(cublasGemmEx(globals::cublas_handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, k,
                              &alpha, d_U, IT2, m,
                              d_X, IT1, k,
                              &beta, d_Y, OT,
                              m, compute_type,
                              CUBLAS_GEMM_DEFAULT));

}


template <typename InputTensor_t, typename MatrixCollection_t, typename OutputTensor_t>
OutputTensor_t ttmc_mixed(InputTensor_t& X, MatrixCollection_t& matrices)
{

    using TensorType_t = InputTensor_t::ValueType_t;
    using MatrixType_t = MatrixCollection_t::ValueType_t;

    static constexpr auto MatrixRows = MatrixCollection_t::Rows; 
    static constexpr auto MatrixCols = MatrixCollection_t::Cols; 
    auto TensorModes = InputTensor_t::Modes; 

    static constexpr size_t N = MatrixCollection_t::N;

    /* Scaling */
    //TODO
    TensorType_t * d_X = X.d_data;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
     

    /* Set up output tensor */
    TensorType_t * d_Y_prev, * d_Y_curr;
    //TODO: Replace sequence of mallocs and frees with one allocation up here

    /* TTM chain */
    for (int i = 0; i < N; i++)
    {

        MatrixType_t * d_U = matrices.get_matrix(i);

        size_t y_size;

        if (i == 0)
        {
            /* Special case of mode 1 */
            const size_t m = MatrixRows[0];
            const size_t n = X.unfolding_cols(0);
            const size_t k = MatrixCols[0];

            y_size = m * n;

            CUDA_CHECK(cudaMalloc(&d_Y_curr, sizeof(TensorType_t) * y_size));


            ttm_mode1(d_X, d_U, d_Y_curr, 
                      m, n, k, 
                      compute_type);


            d_Y_prev = d_Y_curr;
            d_Y_curr = nullptr;

            TensorModes[0] = MatrixRows[0];
        }
        else
        {
            size_t m = 1;
            size_t p = 1;

            for (int j=0; j<i; j++)
            {
                m *= TensorModes[j];
            }

            for (int j=i+1; j<N; j++)
            {
                p *= TensorModes[j]; 
            }

            size_t n = MatrixCols[i];
            size_t k = MatrixRows[i];

            y_size = m * k * p;
            CUDA_CHECK(cudaMalloc(&d_Y_curr, sizeof(TensorType_t) * y_size));


            ttm_modek(d_Y_prev, d_U, d_Y_curr, 
                      m, n, k, 
                      p, i, 
                      compute_type);


            CUDA_FREE(d_Y_prev);
            d_Y_prev = d_Y_curr;

            TensorModes[i] = MatrixRows[i];
        }

        CUDA_CHECK(cudaDeviceSynchronize());

    }


    OutputTensor_t Y(d_Y_prev);
    return Y;
}



} //mxt

#endif
